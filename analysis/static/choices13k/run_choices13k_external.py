from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import qmc

from rfdt_power import (
    POWER_BOUNDS,
    power_D_and_derivative,
    equilibrium_and_derivatives,
    unpack_params,
    free_bounds,
    deterministic_starts,
    predict as rfdt_predict,
)

FOLD_SEED = 20260911
N_FOLDS = 10
MODELS = [
    'RFDT-P',
    'RFDT-L',
    'RFDT-P kappa=0',
    'RFDT-P alpha=0',
    'RFDT-P alpha=1',
    'Power-EU logit',
    'Linear-EU logit',
    'TrainMean',
    'Half',
]

RFDT_MODEL_MAP = {
    'RFDT-P': 'power',
    'RFDT-L': 'rho1',
    'RFDT-P kappa=0': 'kappa0',
    'RFDT-P alpha=0': 'alpha0',
    'RFDT-P alpha=1': 'alpha1',
}


def extract_member_by_suffix(zpath, suffix, outdir):
    zpath = Path(zpath)
    with zipfile.ZipFile(zpath) as z:
        hits = [n for n in z.namelist() if n.endswith(suffix)]
        if len(hits) != 1:
            raise RuntimeError('Expected exactly one %s in %s; found %r' % (suffix, zpath, hits))
        target = Path(outdir) / Path(hits[0]).name
        target.write_bytes(z.read(hits[0]))
        return target


def load_choices13k(zpath, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    csvp = extract_member_by_suffix(zpath, 'c13k_selections.csv', work)
    jsonp = extract_member_by_suffix(zpath, 'c13k_problems.json', work)
    df = pd.read_csv(csvp)
    with open(jsonp, 'r') as f:
        problems = json.load(f)

    required = ['Problem','Feedback','n','Block','Ha','pHa','La','Hb','pHb','Lb',
                'LotShapeB','LotNumB','Amb','Corr','bRate','bRate_std']
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError('Missing required columns: %r' % miss)

    # Frozen external-validation subset from the RFDT checkpoint.
    keep = (~df['Feedback'].astype(bool)) & (~df['Amb'].astype(bool)) & (df['Corr'].astype(int) == 0)
    sub = df.loc[keep].copy()
    sub['source_row'] = sub.index.astype(int)
    sub = sub.reset_index(drop=True)

    if len(sub) != 1766 or sub['Problem'].nunique() != 1766:
        raise RuntimeError('Frozen subset mismatch: rows=%d unique problems=%d, expected 1766/1766' %
                           (len(sub), sub['Problem'].nunique()))
    if not ((sub['Block'] == 1).all() and (~sub['Feedback']).all() and (~sub['Amb']).all() and (sub['Corr'] == 0).all()):
        raise RuntimeError('Frozen subset predicates failed after filtering')

    # Build exact displayed payoff distributions from JSON.  The JSON key is
    # the original CSV row index, as documented by Choices13k.
    A_out, A_prob, B_out, B_prob = [], [], [], []
    max_prod = 0
    max_A = 0
    max_B = 0
    max_ev_err = 0.0
    for r in sub.itertuples(index=False):
        key = str(int(r.source_row))
        if key not in problems:
            raise RuntimeError('Missing JSON problem for source row %s' % key)
        pr = problems[key]
        A = pr['A']; B = pr['B']
        ap = [float(v[0]) for v in A]; ao = [float(v[1]) for v in A]
        bp = [float(v[0]) for v in B]; bo = [float(v[1]) for v in B]
        if not np.isclose(sum(ap), 1.0, atol=1e-10) or not np.isclose(sum(bp), 1.0, atol=1e-10):
            raise RuntimeError('Problem probabilities do not sum to one: source row %s' % key)
        evA_json = float(np.dot(ap, ao)); evB_json = float(np.dot(bp, bo))
        evA_tab = float(r.pHa*r.Ha + (1.0-r.pHa)*r.La)
        evB_tab = float(r.pHb*r.Hb + (1.0-r.pHb)*r.Lb)
        max_ev_err = max(max_ev_err, abs(evA_json-evA_tab), abs(evB_json-evB_tab))
        A_out.append(ao); A_prob.append(ap); B_out.append(bo); B_prob.append(bp)
        max_A = max(max_A, len(ao)); max_B = max(max_B, len(bo)); max_prod = max(max_prod, len(ao)*len(bo))

    if max_ev_err > 1e-8:
        raise RuntimeError('JSON/tabular expected-value mismatch max error %g' % max_ev_err)

    # Exact independent/product hypothetical sampling.  Product support reaches 18 states here.
    nprob = len(sub)
    XA = np.zeros((nprob, max_prod), float)
    XB = np.zeros((nprob, max_prod), float)
    W = np.zeros((nprob, max_prod), float)
    for j, (ao, ap, bo, bp) in enumerate(zip(A_out,A_prob,B_out,B_prob)):
        k = 0
        for xa, pa in zip(ao, ap):
            for xb, pb in zip(bo, bp):
                XA[j,k] = float(xa); XB[j,k] = float(xb); W[j,k] = float(pa)*float(pb); k += 1
        if not np.isclose(W[j].sum(), 1.0, atol=1e-10):
            raise RuntimeError('Product support probability mismatch at row %d' % j)

    # Outcome-blind deterministic 10-fold problem split.
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(nprob)
    fold = np.empty(nprob, dtype=int)
    fold[order] = (np.arange(nprob) % N_FOLDS) + 1
    sub['cv_fold'] = fold

    # Full-dataset stimulus-only normalization sanity checks at rho=1 and rho=.25.
    _, _, scale1, mids1 = power_D_and_derivative(XA,XB,W,1.0,norm_XA=XA,norm_XB=XB,norm_W=W)
    _, _, scale025, mids025 = power_D_and_derivative(XA,XB,W,0.25,norm_XA=XA,norm_XB=XB,norm_W=W)

    audit = {
        'all_rows': int(len(df)),
        'frozen_subset_rows': int(len(sub)),
        'frozen_subset_unique_problems': int(sub['Problem'].nunique()),
        'subset_rule': 'Feedback=False, Amb=False, Corr=0',
        'all_subset_Block_1': bool((sub['Block']==1).all()),
        'n_min': int(sub['n'].min()), 'n_median': float(sub['n'].median()), 'n_max': int(sub['n'].max()),
        'bRate_min': float(sub['bRate'].min()), 'bRate_mean': float(sub['bRate'].mean()), 'bRate_max': float(sub['bRate'].max()),
        'max_A_support': int(max_A), 'max_B_support': int(max_B), 'max_product_support': int(max_prod),
        'max_json_tabular_EV_error': float(max_ev_err),
        'rho1_scale': float(scale1), 'rho025_scale': float(scale025),
        'rho1_median_indices': [int(x) for x in mids1],
        'rho025_median_indices': [int(x) for x in mids025],
        'fold_seed': int(FOLD_SEED), 'n_folds': int(N_FOLDS),
        'fold_counts': {str(int(k)): int(v) for k,v in sub['cv_fold'].value_counts().sort_index().items()},
        'mean_se2_from_bRate_std_over_n': float(np.mean((sub['bRate_std'].to_numpy(float)**2)/sub['n'].to_numpy(float))),
    }
    return sub, XA, XB, W, audit


def _rfdt_mse_and_grad(x, XA, XB, W, rate, model, norm_XA, norm_XB, norm_W):
    a,b,k,lam,rho = unpack_params(x, model)
    D,dD,_,_ = power_D_and_derivative(XA,XB,W,rho,norm_XA,norm_XB,norm_W)
    want_rho = model != 'rho1'
    m, dm = equilibrium_and_derivatives(D,W,a,b,k,dD if want_rho else None)
    eta = lam*m
    p = expit(eta)
    e = p - rate
    val = float(np.mean(e*e))
    common = (2.0/len(rate))*e*p*(1.0-p)
    ga = float(np.sum(common*lam*dm[:,0]))
    gb = float(np.sum(common*lam*dm[:,1]))
    gk = float(np.sum(common*lam*dm[:,2]))
    gl = float(np.sum(common*m))
    if want_rho:
        gr = float(np.sum(common*lam*dm[:,3]))
    if model == 'power': g = np.array([ga,gb,gk,gl,gr])
    elif model == 'rho1': g = np.array([ga,gb,gk,gl])
    elif model == 'kappa0': g = np.array([ga,gb,gl,gr])
    elif model in ('alpha0','alpha1'): g = np.array([gb,gk,gl,gr])
    else: raise ValueError(model)
    return val, g


def fit_rfdt_mse(XA,XB,W,rate,model,n_halton,seed,norm_XA,norm_XB,norm_W):
    bounds = free_bounds(model)
    best = None
    recs = []
    for j,x0 in enumerate(deterministic_starts(model,n_halton,seed)):
        res = minimize(lambda x: _rfdt_mse_and_grad(x,XA,XB,W,rate,model,norm_XA,norm_XB,norm_W),
                       x0,jac=True,method='L-BFGS-B',bounds=bounds,
                       options={'ftol':1e-13,'gtol':1e-8,'maxiter':1500,'maxls':100})
        recs.append({'start':j,'fun':float(res.fun),'success':bool(res.success),'nit':int(res.nit),'message':str(res.message)})
        if best is None or res.fun < best.fun:
            best = res
    return best,recs


def _eu_unpack(x, power):
    if power:
        return float(x[0]), float(x[1])
    return float(x[0]), 1.0


def _eu_mse_grad(x,XA,XB,W,rate,power,norm_XA,norm_XB,norm_W):
    lam,rho = _eu_unpack(x,power)
    D,dD,_,_ = power_D_and_derivative(XA,XB,W,rho,norm_XA,norm_XB,norm_W)
    v = np.sum(W*D,axis=1)
    dv = np.sum(W*dD,axis=1)
    p = expit(lam*v)
    e = p-rate
    val = float(np.mean(e*e))
    common = (2.0/len(rate))*e*p*(1-p)
    gl = float(np.sum(common*v))
    if power:
        gr = float(np.sum(common*lam*dv))
        return val,np.array([gl,gr])
    return val,np.array([gl])


def fit_eu_mse(XA,XB,W,rate,power,n_halton,seed,norm_XA,norm_XB,norm_W):
    if power:
        bounds=[POWER_BOUNDS['lam'],POWER_BOUNDS['rho']]
        central=np.array([[2.0,0.22],[1.0,0.85]],float)
    else:
        bounds=[POWER_BOUNDS['lam']]
        central=np.array([[1.0],[2.5]],float)
    b=np.asarray(bounds,float)
    sampler=qmc.Halton(d=len(bounds),scramble=True,seed=seed)
    pts=qmc.scale(sampler.random(n=n_halton),b[:,0],b[:,1]) if n_halton>0 else np.empty((0,len(bounds)))
    starts=np.vstack([central,pts])
    best=None; recs=[]
    for j,x0 in enumerate(starts):
        res=minimize(lambda x:_eu_mse_grad(x,XA,XB,W,rate,power,norm_XA,norm_XB,norm_W),
                     x0,jac=True,method='L-BFGS-B',bounds=bounds,
                     options={'ftol':1e-13,'gtol':1e-8,'maxiter':1500,'maxls':100})
        recs.append({'start':j,'fun':float(res.fun),'success':bool(res.success),'nit':int(res.nit),'message':str(res.message)})
        if best is None or res.fun < best.fun: best=res
    return best,recs


def predict_eu(XA,XB,W,x,power,norm_XA,norm_XB,norm_W):
    lam,rho = _eu_unpack(x,power)
    D,_,scale,_ = power_D_and_derivative(XA,XB,W,rho,norm_XA,norm_XB,norm_W)
    v=np.sum(W*D,axis=1)
    return expit(lam*v),v,scale


def prediction_metrics(rate,pred,bRate_std,n):
    rate=np.asarray(rate,float); pred=np.asarray(pred,float)
    se2=(np.asarray(bRate_std,float)**2)/np.asarray(n,float)
    mse=float(np.mean((pred-rate)**2)); rmse=float(np.sqrt(mse)); mae=float(np.mean(np.abs(pred-rate)))
    naive05=float(np.mean((0.5-rate)**2)); irr=float(np.mean(se2))
    completeness=float((naive05-mse)/(naive05-irr)) if naive05>irr else np.nan
    pc=np.clip(pred,1e-12,1-1e-12)
    ce=float(np.mean(-(rate*np.log(pc)+(1-rate)*np.log1p(-pc))))
    corr=float(np.corrcoef(rate,pred)[0,1]) if np.std(pred)>0 and np.std(rate)>0 else np.nan
    if len(rate)>=2 and np.std(pred)>0:
        slope,intercept=np.polyfit(pred,rate,1)
    else:
        slope,intercept=np.nan,np.nan
    return dict(mse=mse,rmse=rmse,mae=mae,rate_cross_entropy=ce,correlation=corr,
                naive05_mse=naive05,irreducible_se2=irr,completeness05=completeness,
                calibration_intercept=float(intercept),calibration_slope=float(slope))



def gradient_self_test(XA,XB,W):
    """Finite-difference checks using synthetic target rates only.

    This validates both the even-sample median-scale derivative and the MSE
    gradients before any human outcome is used for model fitting.
    """
    idx=np.arange(min(73,len(XA)))
    xa,xb,w=XA[idx],XB[idx],W[idx]
    rate=np.linspace(0.17,0.83,len(idx))

    x=np.array([0.43,1.35,0.57,1.8,0.61],float)
    val,g=_rfdt_mse_and_grad(x,xa,xb,w,rate,'power',XA,XB,W)
    gn=np.zeros_like(g)
    for j in range(len(x)):
        h=1e-6*max(1.0,abs(x[j]))
        xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h
        vp,_=_rfdt_mse_and_grad(xp,xa,xb,w,rate,'power',XA,XB,W)
        vm,_=_rfdt_mse_and_grad(xm,xa,xb,w,rate,'power',XA,XB,W)
        gn[j]=(vp-vm)/(2*h)
    den=np.maximum(1e-7,np.maximum(np.abs(g),np.abs(gn)))
    err1=float(np.max(np.abs(g-gn)/den))

    xe=np.array([1.7,0.64],float)
    ve,ge=_eu_mse_grad(xe,xa,xb,w,rate,True,XA,XB,W)
    gen=np.zeros_like(ge)
    for j in range(len(xe)):
        h=1e-6*max(1.0,abs(xe[j]))
        xp=xe.copy();xm=xe.copy();xp[j]+=h;xm[j]-=h
        vp,_=_eu_mse_grad(xp,xa,xb,w,rate,True,XA,XB,W)
        vm,_=_eu_mse_grad(xm,xa,xb,w,rate,True,XA,XB,W)
        gen[j]=(vp-vm)/(2*h)
    dene=np.maximum(1e-7,np.maximum(np.abs(ge),np.abs(gen)))
    err2=float(np.max(np.abs(ge-gen)/dene))
    return {'rfdt_power_max_relative_gradient_error':err1,'power_eu_max_relative_gradient_error':err2}

def fit_one(args):
    work, fold, label, nstarts = args
    work=Path(work)
    df=pd.read_pickle(work/'subset.pkl')
    XA=np.load(work/'XA.npy'); XB=np.load(work/'XB.npy'); W=np.load(work/'W.npy')
    train=np.where(df.cv_fold.to_numpy(int)!=fold)[0]
    test=np.where(df.cv_fold.to_numpy(int)==fold)[0]
    rtrain=df.iloc[train].bRate.to_numpy(float)

    if label in RFDT_MODEL_MAP:
        mdl=RFDT_MODEL_MAP[label]
        res,recs=fit_rfdt_mse(XA[train],XB[train],W[train],rtrain,mdl,nstarts,
                              FOLD_SEED+fold,norm_XA=XA,norm_XB=XB,norm_W=W)
        pred,latent,scale=rfdt_predict(XA[test],XB[test],W[test],res.x,mdl,
                                      norm_XA=XA,norm_XB=XB,norm_W=W)
        a,b,k,lam,rho=unpack_params(res.x,mdl)
        params=dict(alpha=a,beta=b,kappa=k,lambda_=lam,rho=rho)
        train_loss=float(res.fun)
    elif label in ('Power-EU logit','Linear-EU logit'):
        power=label.startswith('Power')
        res,recs=fit_eu_mse(XA[train],XB[train],W[train],rtrain,power,nstarts,
                            FOLD_SEED+fold,norm_XA=XA,norm_XB=XB,norm_W=W)
        pred,latent,scale=predict_eu(XA[test],XB[test],W[test],res.x,power,
                                     norm_XA=XA,norm_XB=XB,norm_W=W)
        lam,rho=_eu_unpack(res.x,power)
        params=dict(alpha=np.nan,beta=np.nan,kappa=np.nan,lambda_=lam,rho=rho)
        train_loss=float(res.fun)
    else:
        raise ValueError(label)

    fitrow=dict(model=label,fold=int(fold),train_mse=train_loss,success=bool(res.success),
                n_train=int(len(train)),n_test=int(len(test)),normalization_scale=float(scale),**params)
    out=df.iloc[test][['Problem','source_row','cv_fold','n','bRate','bRate_std','Ha','pHa','La','Hb','pHb','Lb','LotShapeB','LotNumB']].copy()
    out['model']=label; out['pred']=pred; out['latent']=latent
    return fitrow,out,recs


def full_data_fit(df,XA,XB,W,label,nstarts):
    rate=df.bRate.to_numpy(float)
    if label in RFDT_MODEL_MAP:
        mdl=RFDT_MODEL_MAP[label]
        res,recs=fit_rfdt_mse(XA,XB,W,rate,mdl,nstarts,FOLD_SEED+999,norm_XA=XA,norm_XB=XB,norm_W=W)
        a,b,k,lam,rho=unpack_params(res.x,mdl)
        return dict(model=label,full_data_mse=float(res.fun),success=bool(res.success),alpha=a,beta=b,kappa=k,lambda_=lam,rho=rho)
    power=label.startswith('Power')
    res,recs=fit_eu_mse(XA,XB,W,rate,power,nstarts,FOLD_SEED+999,norm_XA=XA,norm_XB=XB,norm_W=W)
    lam,rho=_eu_unpack(res.x,power)
    return dict(model=label,full_data_mse=float(res.fun),success=bool(res.success),alpha=np.nan,beta=np.nan,kappa=np.nan,lambda_=lam,rho=rho)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--choices',required=True,help='choices13k-main.zip')
    ap.add_argument('--out',required=True)
    ap.add_argument('--jobs',type=int,default=10)
    ap.add_argument('--starts',type=int,default=32,help='Halton starts per fitted model in addition to central starts')
    a=ap.parse_args()

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    work=out/'_work'; work.mkdir(exist_ok=True)

    # Write the analysis specification before loading human outcomes for fitting.
    spec={
        'analysis':'Choices13k external static validation of frozen RFDT-P',
        'subset':{'Feedback':False,'Amb':False,'Corr':0,'expected_n_problems':1766},
        'choice_orientation':'bRate = aggregate rate of choosing Gamble B',
        'stimulus_representation':'exact displayed A/B distributions from c13k_problems.json; independent/product hypothetical sampling',
        'utility':'u_rho(x)=sign(x)|x|^rho',
        'utility_scale':'median_j sqrt(E[D_j(rho)^2])=1 over the entire frozen stimulus subset; for even n, standard median = mean of two central order statistics; stimulus-only and outcome-blind',
        'rfdt_bounds':POWER_BOUNDS,
        'cv':{'type':'10-fold problem CV','fold_seed':FOLD_SEED,'fold_assignment':'seeded permutation of problem rows, round-robin 1..10; no outcomes used'},
        'training_objective':'unweighted problem-level MSE between predicted P(B) and aggregate bRate',
        'primary_metric':'held-out problem-level MSE on bRate',
        'secondary_metrics':['RMSE','MAE','rate cross-entropy (proper score, NOT treated as a likelihood)','correlation','completeness vs 0.5 using bRate_std^2/n only as an estimated sampling-noise floor'],
        'likelihood_warning':'bRate*n is NOT treated as a binomial count; n is the number of subjects and each subject made 5 choices before aggregation',
        'predeclared_models':MODELS,
        'model_status':'RFDT-P form and parameter domain frozen after HAB22; no Choices13k outcomes are used to modify the model',
    }
    (out/'analysis_spec.json').write_text(json.dumps(spec,indent=2,sort_keys=True))

    df,XA,XB,W,audit=load_choices13k(a.choices,work)
    gradcheck=gradient_self_test(XA,XB,W)
    audit['gradient_self_test']=gradcheck
    if max(gradcheck.values()) > 5e-4:
        raise RuntimeError('Gradient self-test failed: %r' % gradcheck)
    (out/'choices13k_subset_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True))
    df[['Problem','source_row','cv_fold','n','bRate','bRate_std','Ha','pHa','La','Hb','pHb','Lb','LotShapeB','LotNumB']].to_csv(out/'choices13k_folds_and_subset.csv',index=False)
    df.to_pickle(work/'subset.pkl'); np.save(work/'XA.npy',XA); np.save(work/'XB.npy',XB); np.save(work/'W.npy',W)

    fitted=[m for m in MODELS if m not in ('TrainMean','Half')]
    todo=[(str(work),fold,m,a.starts) for fold in range(1,N_FOLDS+1) for m in fitted]
    fitrows=[]; oof=[]; opt=[]
    print('Choices13k external CV: %d fitted fold-model jobs; jobs=%d; starts=%d (+ central starts)' % (len(todo),a.jobs,a.starts),flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(fit_one,x) for x in todo]
        for i,f in enumerate(as_completed(futs),1):
            fr,pr,recs=f.result(); fitrows.append(fr); oof.append(pr)
            for rr in recs:
                z=dict(model=fr['model'],fold=fr['fold']);z.update(rr);opt.append(z)
            if i%10==0 or i==len(todo): print('  completed %d/%d' % (i,len(todo)),flush=True)

    fitdf=pd.DataFrame(fitrows).sort_values(['model','fold'])
    fitdf.to_csv(out/'choices13k_fit_parameters.csv',index=False)
    pd.DataFrame(opt).to_csv(out/'choices13k_optimizer_audit.csv',index=False)
    pred=pd.concat(oof,ignore_index=True)

    # Add outcome-safe baselines fold by fold.
    base=[]
    for fold in range(1,N_FOLDS+1):
        tr=df[df.cv_fold!=fold]; te=df[df.cv_fold==fold]
        tm=float(tr.bRate.mean())
        for label,pv in [('TrainMean',tm),('Half',0.5)]:
            q=te[['Problem','source_row','cv_fold','n','bRate','bRate_std','Ha','pHa','La','Hb','pHb','Lb','LotShapeB','LotNumB']].copy()
            q['model']=label;q['pred']=pv;q['latent']=np.nan;base.append(q)
    pred=pd.concat([pred]+base,ignore_index=True)
    pred.to_csv(out/'choices13k_oof_predictions.csv',index=False)

    # Per-fold metrics and global OOF ranking.
    fm=[]
    for (model,fold),g in pred.groupby(['model','cv_fold']):
        d=prediction_metrics(g.bRate,g.pred,g.bRate_std,g.n);d.update(model=model,fold=int(fold),n_test=int(len(g)));fm.append(d)
    fm=pd.DataFrame(fm).sort_values(['model','fold']);fm.to_csv(out/'choices13k_fold_metrics.csv',index=False)

    ranks=[]
    for model,g in pred.groupby('model'):
        d=prediction_metrics(g.bRate,g.pred,g.bRate_std,g.n);d.update(model=model,n_problems=int(len(g)));ranks.append(d)
    rank=pd.DataFrame(ranks).sort_values('mse').reset_index(drop=True);rank.insert(0,'rank_mse',np.arange(1,len(rank)+1));rank.to_csv(out/'choices13k_model_ranking.csv',index=False)

    # Foldwise nested comparisons against the frozen full RFDT-P.
    full=fm[fm.model=='RFDT-P'][['fold','mse','rate_cross_entropy']].rename(columns={'mse':'full_mse','rate_cross_entropy':'full_ce'})
    comps=[]
    for other in ['RFDT-L','RFDT-P kappa=0','RFDT-P alpha=0','RFDT-P alpha=1','Power-EU logit','Linear-EU logit']:
        oo=fm[fm.model==other][['fold','mse','rate_cross_entropy']].rename(columns={'mse':'other_mse','rate_cross_entropy':'other_ce'})
        j=full.merge(oo,on='fold')
        for r in j.itertuples(index=False):
            comps.append(dict(other=other,fold=int(r.fold),delta_mse_other_minus_full=float(r.other_mse-r.full_mse),delta_ce_other_minus_full=float(r.other_ce-r.full_ce)))
    comp=pd.DataFrame(comps);comp.to_csv(out/'choices13k_nested_fold_comparisons.csv',index=False)
    summary=[]
    for other,g in comp.groupby('other'):
        summary.append(dict(other=other,mean_delta_mse=float(g.delta_mse_other_minus_full.mean()),median_delta_mse=float(g.delta_mse_other_minus_full.median()),
                            full_better_mse_folds=int((g.delta_mse_other_minus_full>0).sum()),
                            mean_delta_ce=float(g.delta_ce_other_minus_full.mean()),full_better_ce_folds=int((g.delta_ce_other_minus_full>0).sum()),n_folds=int(len(g))))
    pd.DataFrame(summary).sort_values('mean_delta_mse',ascending=False).to_csv(out/'choices13k_nested_comparison_summary.csv',index=False)

    # Descriptive all-data fits are performed only after OOF predictions are complete; never used for scoring.
    fd=[]
    for m in fitted:
        print('Full-data descriptive fit:',m,flush=True)
        fd.append(full_data_fit(df,XA,XB,W,m,max(a.starts,40)))
    pd.DataFrame(fd).to_csv(out/'choices13k_full_data_descriptive_fits.csv',index=False)

    freeze={
        'status':'Choices13k completed as external validation of the HAB22-frozen RFDT-P specification',
        'no_model_selection_from_test_outcomes':True,
        'rfdt_p_utility':'sign(x)*abs(x)^rho',
        'rfdt_p_bounds':POWER_BOUNDS,
        'normalization':'within-dataset outcome-blind median RMS comparative evidence = 1',
        'interpretation_caution':'Choices13k was previously shown to be weaker for kappa recovery than HAB22; use it primarily for predictive external generalization, not precise kappa inference.',
    }
    (out/'CHOICES13K_EXTERNAL_VALIDATION_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True))
    print('\nOOF ranking:\n',rank[['rank_mse','model','mse','completeness05','rate_cross_entropy','correlation']].to_string(index=False),flush=True)
    print('\nDone:',out,flush=True)

if __name__=='__main__':
    main()
