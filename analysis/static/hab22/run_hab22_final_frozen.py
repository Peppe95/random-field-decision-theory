from __future__ import annotations
import argparse, json, math, os, shutil, time, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import qmc

from read_rdata_min import load_rdata, top_env, attrs_to_dict, vec_values, STRSXP, INTSXP, LGLSXP, REALSXP
from rfdt_power import (fit as fit_rfdt, predict as predict_rfdt, unpack_params,
                        POWER_BOUNDS, build_raw_pair_arrays, power_D_and_derivative,
                        gradient_self_test)

NA_INT=-2147483648

# ---------------- RData + exact R 4.x folds ----------------
def r_dataframe_to_pandas(obj):
    ad=attrs_to_dict(obj.attrs); names=vec_values(ad['names']); out={}
    for nm,col in zip(names,obj.data):
        vals=vec_values(col)
        if col.type==STRSXP: out[nm]=pd.Series(vals,dtype='object')
        elif col.type in (INTSXP,LGLSXP):
            arr=np.array(vals,dtype=np.int64)
            if np.any(arr==NA_INT):
                a=arr.astype(float); a[arr==NA_INT]=np.nan; out[nm]=a
            else: out[nm]=arr
        elif col.type==REALSXP: out[nm]=np.array(vals,dtype=float)
        else: out[nm]=vals
    return pd.DataFrame(out)

class RRNG:
    def __init__(self,seed):
        self.state=np.zeros(624,dtype=np.uint32); s=int(seed)&0xFFFFFFFF
        for _ in range(50): s=(69069*s+1)&0xFFFFFFFF
        seeds=[]
        for _ in range(625): s=(69069*s+1)&0xFFFFFFFF; seeds.append(s)
        self.mti=624; self.state[:]=np.asarray(seeds[1:625],dtype=np.uint32)
    def unif(self):
        N,M=624,397
        if self.mti>=N:
            mt=self.state; mag=np.array([0,0x9908b0df],dtype=np.uint32)
            for kk in range(0,N-M):
                y=(mt[kk]&np.uint32(0x80000000))|(mt[kk+1]&np.uint32(0x7fffffff))
                mt[kk]=mt[kk+M]^(y>>np.uint32(1))^mag[int(y&np.uint32(1))]
            for kk in range(N-M,N-1):
                y=(mt[kk]&np.uint32(0x80000000))|(mt[kk+1]&np.uint32(0x7fffffff))
                mt[kk]=mt[kk+(M-N)]^(y>>np.uint32(1))^mag[int(y&np.uint32(1))]
            y=(mt[N-1]&np.uint32(0x80000000))|(mt[0]&np.uint32(0x7fffffff))
            mt[N-1]=mt[M-1]^(y>>np.uint32(1))^mag[int(y&np.uint32(1))]; self.mti=0
        y=np.uint32(self.state[self.mti]); self.mti+=1
        y^=y>>np.uint32(11); y^=(y<<np.uint32(7))&np.uint32(0x9d2c5680)
        y^=(y<<np.uint32(15))&np.uint32(0xefc60000); y^=y>>np.uint32(18)
        u=float(np.uint64(y))*2.3283064365386963e-10; eps=2.328306437080797e-10
        if u<=0:return .5*eps
        if 1-u<=0:return 1-.5*eps
        return u
    def rbits(self,bits):
        v=0;n=0
        while n<=bits: v=65536*v+int(math.floor(self.unif()*65536.0));n+=16
        return v&((1<<bits)-1)
    def unif_index(self,n):
        bits=int(math.ceil(math.log2(n)))
        while True:
            dv=self.rbits(bits)
            if dv<n:return dv
    def sample_perm(self,n):
        x=list(range(n));out=[];cur=n
        for _ in range(n):
            j=self.unif_index(cur);out.append(x[j]+1);cur-=1;x[j]=x[cur]
        return np.array(out,dtype=int)

def r_cut_5(vals):
    vals=np.asarray(vals,float);lo,hi=float(vals.min()),float(vals.max());dx=hi-lo;br=np.linspace(lo,hi,6)
    if dx==0:
        dx=abs(lo)/1000 if lo!=0 else 1/1000;br=np.linspace(lo-dx,hi+dx,6)
    else: br[0]=lo-dx/1000;br[-1]=hi+dx/1000
    return np.searchsorted(br[1:],vals,side='left')+1

def make_subject_folds(hab):
    u=hab[['dataset','Type','subject']].drop_duplicates(keep='first').copy();u['fold']=np.nan;rng=RRNG(123)
    for ds in pd.unique(u.dataset):
        idx=np.flatnonzero(u.dataset.to_numpy()==ds);u.loc[u.index[idx],'fold']=r_cut_5(rng.sample_perm(len(idx)))
    u['fold']=u['fold'].astype(int);return u

def extract_member(zpath,suffix,outdir):
    with zipfile.ZipFile(zpath) as z:
        ms=[n for n in z.namelist() if n.endswith(suffix) and '/__MACOSX/' not in ('/'+n) and not Path(n).name.startswith('._')]
        if len(ms)!=1:raise RuntimeError(f'Expected one {suffix!r}, found {ms}')
        dest=outdir/Path(ms[0]).name
        with z.open(ms[0]) as fi,open(dest,'wb') as fo:shutil.copyfileobj(fi,fo)
    return dest

def parse_nums(s):
    return [float(v) for v in str(s).split(';')]

def prepare(plonsky_zip,recovery_zip,work):
    work.mkdir(parents=True,exist_ok=True)
    hab_r=extract_member(plonsky_zip,'HAB22 all data with Stew1C_Uni.RData',work)
    gb_r=extract_member(plonsky_zip,'HAB22 R1 preds 5X10cv.RData',work)
    raw_csv=extract_member(recovery_zip,'hab22_standardized_design.csv',work)
    lin_csv=extract_member(recovery_zip,'HAB22_phase3b_rescaled_design.csv',work)
    old_rec=extract_member(recovery_zip,'HAB22_recovery_raw.csv',work)

    top,_,_=load_rdata(str(hab_r));hab=r_dataframe_to_pandas(top_env(top)['HAB22'])
    top2,_,_=load_rdata(str(gb_r));gb=r_dataframe_to_pandas(top_env(top2)['all_preds'])
    original_cols=hab.columns.tolist(); behavioral_cols=original_cols[12:72]
    if len(original_cols)!=90 or len(behavioral_cols)!=60 or original_cols[89]!='BEASTpred':
        raise RuntimeError('Unexpected HAB22 column layout')
    hab=hab[hab.dataset!='Stewart15_1C_uniform'].copy(); folds=make_subject_folds(hab)
    hab=hab.merge(folds[['dataset','subject','fold']],on=['dataset','subject'],how='left',validate='many_to_one')

    obs=hab.groupby(['task_id','crossValidation_id','fold'],as_index=False).agg(rate=('choice','mean'))
    gbt=gb.copy();gbt['cvid']=gbt.cvid.astype(int);gbt['cvsubjs']=gbt.cvsubjs.astype(int)
    cmp=gbt.merge(obs,left_on=['task_id','cvid','cvsubjs'],right_on=['task_id','crossValidation_id','fold'],how='left')
    max_diff=float(np.nanmax(np.abs(cmp.B_rate-cmp.rate)))
    if cmp.rate.isna().any() or max_diff>1e-12:raise RuntimeError(f'Fold reproduction failed: {max_diff}')

    raw=pd.read_csv(raw_csv).rename(columns={'problem_id':'task_id'});lin=pd.read_csv(lin_csv).rename(columns={'problem_id':'task_id'})
    raw=raw.sort_values('task_id').reset_index(drop=True);lin=lin.sort_values('task_id').reset_index(drop=True)
    if raw.task_id.tolist()!=lin.task_id.tolist() or set(raw.task_id)!=set(hab.task_id.unique()):raise RuntimeError('Task mismatch')
    Aout=[parse_nums(s) for s in raw.A_outcomes];Aprob=[parse_nums(s) for s in raw.A_probabilities]
    Bout=[parse_nums(s) for s in raw.B_outcomes];Bprob=[parse_nums(s) for s in raw.B_probabilities]
    XA,XB,W=build_raw_pair_arrays(Aout,Aprob,Bout,Bprob,max_support=4)
    raw['_row']=np.arange(len(raw))

    # Frozen normalization audit: rho=1 must exactly recreate the Phase-3b linear D design.
    D1,_,scale1,_=power_D_and_derivative(XA,XB,W,1.0)
    Dlin=[];Wlin=[]
    for ds,ps in zip(lin.D_support.astype(str),lin.D_probabilities.astype(str)):
        d=parse_nums(ds);w=parse_nums(ps);Dlin.append(d+[0.]*(4-len(d)));Wlin.append(w+[0.]*(4-len(w)))
    Dlin=np.asarray(Dlin,float);Wlin=np.asarray(Wlin,float)
    # Product pairs can appear in a different order than combined stored support. Compare task moments + sorted support weighted pairs.
    # In HAB22 the stored design keeps all product states; use moments, which are permutation-invariant.
    momdiff=max(float(np.max(np.abs(np.sum(W*D1,axis=1)-np.sum(Wlin*Dlin,axis=1)))),
                float(np.max(np.abs(np.sum(W*D1*D1,axis=1)-np.sum(Wlin*Dlin*Dlin,axis=1)))))
    stored_scale=float(pd.unique(lin.phase3b_D_scale)[0])
    if abs(scale1-stored_scale)>1e-8 or momdiff>1e-8:
        raise RuntimeError(f'Power normalization rho=1 does not reproduce Phase3b: scale {scale1} vs {stored_scale}, momentdiff {momdiff}')
    grad=gradient_self_test(XA,XB,W)
    if grad['max_relative_gradient_error']>2e-4:raise RuntimeError(f'Gradient self-test failed: {grad}')

    oldrec=pd.read_csv(old_rec)
    checks={'rows':int(len(hab)),'tasks':int(hab.task_id.nunique()),'participants':int(hab[['dataset','subject']].drop_duplicates().shape[0]),
            'contexts':int(hab.dataset.nunique()),'max_stored_B_rate_difference':max_diff,
            'rho1_normalization_scale':scale1,'stored_phase3b_scale':stored_scale,'rho1_moment_reproduction_maxdiff':momdiff,
            'gradient_self_test':grad,'behavioral_model_count':60}
    return hab,gb,raw,XA,XB,W,behavioral_cols,oldrec,checks

# ---------------- task aggregation ----------------
def aggregate_cell(hab,cvid,sfold,train):
    x=hab[(hab.crossValidation_id!=cvid)&(hab['fold']!=sfold)] if train else hab[(hab.crossValidation_id==cvid)&(hab['fold']==sfold)]
    g=x.groupby('task_id',as_index=False).agg(y=('choice','sum'),n=('choice','size'),rate=('choice','mean'),sd=('choice','std'))
    g['se2']=(g.sd**2)/g.n;return g

def attach(g,design,XA,XB,W):
    z=g.merge(design[['task_id','_row','dataset']],on='task_id',how='left',validate='one_to_one');idx=z._row.astype(int).to_numpy()
    return z,XA[idx],XB[idx],W[idx]

def metrics(te):
    r=te.rate.to_numpy(float);p=te.pred.to_numpy(float);y=te.y.to_numpy(float);n=te.n.to_numpy(float)
    mse=float(np.mean((p-r)**2));naive=float(np.mean((.5-r)**2));irr=float(np.nanmean(te.se2.to_numpy(float)))
    comp=float((naive-mse)/(naive-irr)) if naive>irr else np.nan;pc=np.clip(p,1e-12,1-1e-12)
    ll=float(-np.sum(y*np.log(pc)+(n-y)*np.log1p(-pc))/np.sum(n));return mse,naive,irr,comp,ll

# ---------------- stored comparators ----------------
def eval_stored(hab,gb,behavioral_cols):
    rows=[];predrows=[];cols=behavioral_cols+['BEASTpred']
    for cvid in range(1,11):
        tt=hab[hab.crossValidation_id==cvid]
        for sf in range(1,6):
            ins=tt[tt['fold']!=sf];outs=tt[tt['fold']==sf];pred=ins.groupby('task_id')[cols].mean()
            obs=outs.groupby('task_id').choice.agg(['sum','mean','std','size']).rename(columns={'sum':'y','mean':'rate','size':'n'})
            j=obs.join(pred,how='inner');j['se2']=(j['std']**2)/j.n
            for model in cols:
                te=j.reset_index().rename(columns={model:'pred'});mse,naive,irr,comp,ll=metrics(te)
                rows.append(dict(model=model,cvid=cvid,subject_fold=sf,mse=mse,naive=naive,irreducible=irr,completeness=comp,logloss=ll))
                for r in te.itertuples(index=False):predrows.append(dict(model=model,cvid=cvid,subject_fold=sf,task_id=r.task_id,obs_rate=r.rate,pred=getattr(r,'pred'),n=int(r.n)))
    for (cvid,sf),j in gb.groupby(['cvid','cvsubjs']):
        cvid=int(cvid);sf=int(sf);obs=hab[(hab.crossValidation_id==cvid)&(hab['fold']==sf)].groupby('task_id').choice.agg(['sum','mean','std','size']).rename(columns={'sum':'y','mean':'rate','size':'n'})
        jj=j.set_index('task_id').join(obs,how='inner');te=jj.reset_index().rename(columns={'pred':'pred'});te['se2']=(te['std']**2)/te.n
        mse,naive,irr,comp,ll=metrics(te);rows.append(dict(model='BEAST-GB',cvid=cvid,subject_fold=sf,mse=mse,naive=naive,irreducible=irr,completeness=comp,logloss=ll))
        for r in te.itertuples(index=False):predrows.append(dict(model='BEAST-GB',cvid=cvid,subject_fold=sf,task_id=r.task_id,obs_rate=r.rate,pred=r.pred,n=int(r.n)))
    return pd.DataFrame(rows),pd.DataFrame(predrows)

# ---------------- pooled power fits ----------------
def fit_one_pooled(args):
    work,cvid,sf,model,nstarts=args;work=Path(work);hab=pd.read_pickle(work/'hab.pkl');design=pd.read_pickle(work/'design.pkl')
    XA=np.load(work/'XA.npy');XB=np.load(work/'XB.npy');W=np.load(work/'W.npy')
    tr=aggregate_cell(hab,cvid,sf,True);tr,xa,xb,w=attach(tr,design,XA,XB,W)
    res,starts=fit_rfdt(xa,xb,w,tr.y.to_numpy(float),tr.n.to_numpy(float),model=model,n_halton=nstarts,seed=20260910,norm_XA=XA,norm_XB=XB,norm_W=W)
    te=aggregate_cell(hab,cvid,sf,False);te,xat,xbt,wt=attach(te,design,XA,XB,W);p,m,scale=predict_rfdt(xat,xbt,wt,res.x,model,norm_XA=XA,norm_XB=XB,norm_W=W)
    te=te.copy();te['pred']=p;te['m_star']=m;a,b,k,l,r=unpack_params(res.x,model)
    return cvid,sf,model,res,starts,a,b,k,l,r,scale,te

# ---------------- individual power fits ----------------
def fit_subject_one(args):
    work,cvid,dataset,subject,nstarts=args;work=Path(work);hab=pd.read_pickle(work/'hab.pkl');design=pd.read_pickle(work/'design.pkl')
    XA=np.load(work/'XA.npy');XB=np.load(work/'XB.npy');W=np.load(work/'W.npy')
    s=hab[(hab.dataset==dataset)&(hab.subject==subject)&(hab.crossValidation_id!=cvid)].copy()
    if len(s)<20:return None
    g=s.groupby('task_id',as_index=False).agg(y=('choice','sum'),n=('choice','size'));g,xa,xb,w=attach(g,design,XA,XB,W)
    res,_=fit_rfdt(xa,xb,w,g.y.to_numpy(float),g.n.to_numpy(float),model='power',n_halton=nstarts,seed=20260910,norm_XA=XA,norm_XB=XB,norm_W=W)
    a,b,k,l,r=unpack_params(res.x,'power')
    tt=hab[(hab.dataset==dataset)&(hab.crossValidation_id==cvid)][['task_id']].drop_duplicates();tt,xat,xbt,wt=attach(tt,design,XA,XB,W);p,_,_=predict_rfdt(xat,xbt,wt,res.x,'power',norm_XA=XA,norm_XB=XB,norm_W=W)
    preds=[(dataset,int(subject),int(cvid),tid,float(pp)) for tid,pp in zip(tt.task_id,p)]
    fr=dict(dataset=dataset,subject=int(subject),cvid=int(cvid),alpha=a,beta=b,kappa=k,lambda_=l,rho=r,train_nll=float(res.fun),success=bool(res.success),n_train_tasks=int(len(g)))
    return fr,preds

def run_individual(work,hab,jobs,nstarts,out):
    people=hab[['dataset','subject','fold']].drop_duplicates();todo=[(str(work),cv,r.dataset,int(r.subject),nstarts) for cv in range(1,11) for r in people.itertuples(index=False)]
    fitrows=[];predrows=[];print(f'FINAL individual RFDT-P: {len(todo)} subject x task-fold fits',flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs=[ex.submit(fit_subject_one,x) for x in todo]
        for i,f in enumerate(as_completed(futs),1):
            ans=f.result()
            if ans is not None:fr,pr=ans;fitrows.append(fr);predrows.extend(pr)
            if i%250==0:print(f'  completed {i}/{len(todo)}',flush=True)
    fits=pd.DataFrame(fitrows);pp=pd.DataFrame(predrows,columns=['dataset','subject','cvid','task_id','pred']).merge(people,on=['dataset','subject'],how='left',validate='many_to_one')
    cr=[];oof=[]
    for cv in range(1,11):
        for sf in range(1,6):
            pred=pp[(pp.cvid==cv)&(pp['fold']!=sf)].groupby('task_id').pred.mean();obs=hab[(hab.crossValidation_id==cv)&(hab['fold']==sf)].groupby('task_id').choice.agg(['sum','size','mean','std']).rename(columns={'sum':'y','size':'n','mean':'rate'})
            j=obs.join(pred,how='inner');j['se2']=(j['std']**2)/j.n;te=j.reset_index();mse,naive,irr,comp,ll=metrics(te)
            cr.append(dict(model='RFDT-P individual-average',cvid=cv,subject_fold=sf,mse=mse,naive=naive,irreducible=irr,completeness=comp,logloss=ll))
            for r in te.itertuples(index=False):oof.append(dict(model='RFDT-P individual-average',cvid=cv,subject_fold=sf,task_id=r.task_id,obs_rate=r.rate,pred=r.pred,n=int(r.n)))
    fits.to_csv(out/'hab22_final_individual_fit_parameters.csv',index=False);pd.DataFrame(cr).to_csv(out/'hab22_final_individual_cell_metrics.csv',index=False);pd.DataFrame(oof).to_csv(out/'hab22_final_individual_oof.csv',index=False)

# ---------------- targeted low-rho exact-design synthetic recovery ----------------
def recovery_truths(oldrec, nreps=None):
    """Prespecified targeted recovery around the newly admitted low-rho region.

    The purpose is local identifiability validation, not another model search.
    Six structural settings are crossed with rho in {0.15, 0.22, 0.30}.
    Two settings deliberately probe the high-beta neighborhood seen in the
    HAB22 training fits; the remaining settings span the previously validated
    regular region.  No human held-out outcomes enter this construction.
    """
    structures = [
        # Near the HAB22 pooled training-fit neighborhood (diagnostic truth).
        (0.45, 3.00, 0.61, 2.50, 'human_near'),
        # A second high-beta but structurally different point.
        (0.70, 2.50, 0.35, 1.50, 'high_beta_alt'),
        # Four representative regular truths inherited from the Phase-3b region.
        (0.288469, 1.107912, 0.529180, 3.551497, 'regular_A'),
        (0.677836, 0.735013, 0.666415, 3.019205, 'regular_B'),
        (0.522545, 1.211539, 0.704664, 1.552705, 'regular_C'),
        (0.183855, 1.307025, 0.857185, 1.818711, 'regular_D'),
    ]
    rhos = [0.15, 0.22, 0.30]
    rows=[]
    rep=0
    for a,b,k,l,label in structures:
        for rho in rhos:
            rows.append(dict(rep=rep, structure=label, true_alpha=a, true_beta=b,
                             true_kappa=k, true_lambda=l, true_rho=rho))
            rep += 1
    out=pd.DataFrame(rows)
    if nreps is not None:
        nreps=int(nreps)
        if nreps < 1 or nreps > len(out):
            raise ValueError(f'--recovery-reps must be between 1 and {len(out)} for this targeted design')
        out=out.iloc[:nreps].copy()
    return out

def fit_recovery_one(args):
    work,row,nstarts=args;work=Path(work);design=pd.read_pickle(work/'design.pkl');XA=np.load(work/'XA.npy');XB=np.load(work/'XB.npy');W=np.load(work/'W.npy');counts=pd.read_pickle(work/'task_counts.pkl')
    a,b,k,l,r=[float(row[x]) for x in ['true_alpha','true_beta','true_kappa','true_lambda','true_rho']]
    xtrue=np.array([a,b,k,l,r]);p,_,_=predict_rfdt(XA,XB,W,xtrue,'power');rng=np.random.default_rng(810000+int(row['rep']));n=counts.n.to_numpy(int);y=rng.binomial(n,p)
    res,_=fit_rfdt(XA,XB,W,y.astype(float),n.astype(float),model='power',n_halton=nstarts,seed=20260912+int(row['rep']))
    ah,bh,kh,lh,rh=unpack_params(res.x,'power')
    return dict(rep=int(row['rep']),structure=str(row.get('structure','')),true_alpha=a,hat_alpha=ah,true_beta=b,hat_beta=bh,true_kappa=k,hat_kappa=kh,true_lambda=l,hat_lambda=lh,true_rho=r,hat_rho=rh,nll=float(res.fun),success=bool(res.success))

def run_recovery(work,oldrec,jobs,nreps,nstarts,out):
    truths=recovery_truths(oldrec,nreps); rows=[]
    truths.to_csv(out/'hab22_final_lowrho_recovery_truths.csv',index=False)
    print(f'Targeted low-rho exact-design recovery: {len(truths)} synthetic datasets',flush=True)
    todo=[(str(work),r._asdict(),nstarts) for r in truths.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs=[ex.submit(fit_recovery_one,x) for x in todo]
        for i,f in enumerate(as_completed(futs),1):
            rows.append(f.result()); print(f'  recovery {i}/{len(todo)}',flush=True)
    raw=pd.DataFrame(rows).sort_values('rep')
    raw['rho_abs_error']=np.abs(raw.hat_rho-raw.true_rho)
    raw['rho_at_lower_bound']=raw.hat_rho <= POWER_BOUNDS['rho'][0] + 1e-5
    raw['rho_at_upper_bound']=raw.hat_rho >= POWER_BOUNDS['rho'][1] - 1e-5
    raw.to_csv(out/'hab22_final_lowrho_recovery_raw.csv',index=False)

    sr=[]
    for par in ['alpha','beta','kappa','lambda','rho']:
        t=raw['true_'+par].to_numpy(); h=raw['hat_'+par].to_numpy(); e=h-t
        corr=float(np.corrcoef(t,h)[0,1]) if len(raw)>1 and np.std(t)>0 and np.std(h)>0 else np.nan
        sr.append(dict(parameter=par,n=len(raw),bias=float(e.mean()),
                       rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),
                       corr_true_hat=corr))
    summ=pd.DataFrame(sr)
    summ.to_csv(out/'hab22_final_lowrho_recovery_summary.csv',index=False)

    by=[]
    for rho,g in raw.groupby('true_rho'):
        for par in ['alpha','beta','kappa','lambda','rho']:
            e=g['hat_'+par].to_numpy()-g['true_'+par].to_numpy()
            by.append(dict(true_rho=float(rho),parameter=par,n=len(g),
                           bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e*e))),
                           mae=float(np.mean(np.abs(e)))))
    pd.DataFrame(by).to_csv(out/'hab22_final_lowrho_recovery_by_rho.csv',index=False)

    diag={
        'n_synthetic_datasets':int(len(raw)),
        'rho_truths':sorted(raw.true_rho.unique().tolist()),
        'rho_hat_min':float(raw.hat_rho.min()),
        'rho_hat_max':float(raw.hat_rho.max()),
        'n_rho_at_lower_bound':int(raw.rho_at_lower_bound.sum()),
        'n_rho_at_upper_bound':int(raw.rho_at_upper_bound.sum()),
        'max_abs_rho_error':float(raw.rho_abs_error.max()),
        'mean_abs_rho_error':float(raw.rho_abs_error.mean()),
    }
    with open(out/'hab22_final_lowrho_recovery_diagnostics.json','w') as f: json.dump(diag,f,indent=2)
    return summ

def summarize(cells):
    s=cells.groupby('model',as_index=False).agg(mean_mse=('mse','mean'),sem_mse=('mse',lambda x:x.std(ddof=1)/np.sqrt(x.notna().sum())),mean_completeness=('completeness','mean'),mean_logloss=('logloss','mean'))
    s=s.sort_values('mean_mse').reset_index(drop=True);s.insert(0,'rank_mse',np.arange(1,len(s)+1));return s

def main():
    ap=argparse.ArgumentParser(description='FINAL frozen HAB22 RFDT-P rerun after training-only rho-boundary audit, plus targeted low-rho exact-design recovery.')
    ap.add_argument('--plonsky',required=True,type=Path);ap.add_argument('--recovery',required=True,type=Path);ap.add_argument('--out',type=Path,default=Path('hab22_rfdt_final_frozen_output'))
    ap.add_argument('--jobs',type=int,default=max(1,min(8,(os.cpu_count() or 2)-1)));ap.add_argument('--starts',type=int,default=24)
    ap.add_argument('--recovery-reps',type=int,default=18);ap.add_argument('--recovery-starts',type=int,default=24)
    ap.add_argument('--skip-recovery',action='store_true');ap.add_argument('--skip-empirical',action='store_true')
    ap.add_argument('--individual',action='store_true');ap.add_argument('--individual-starts',type=int,default=6)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);work=a.out/'_work';work.mkdir(exist_ok=True)
    hab,gb,design,XA,XB,W,behavioral_cols,oldrec,checks=prepare(a.plonsky,a.recovery,work)
    hab.to_pickle(work/'hab.pkl');design.to_pickle(work/'design.pkl');np.save(work/'XA.npy',XA);np.save(work/'XB.npy',XB);np.save(work/'W.npy',W)
    hab.groupby('task_id',as_index=False).choice.size().rename(columns={'size':'n'}).sort_values('task_id').reset_index(drop=True).to_pickle(work/'task_counts.pkl')
    with open(a.out/'reproduction_checks.json','w') as f:json.dump(checks,f,indent=2)
    spec={
          'status':'FINAL HAB22 development-set specification after the prespecified signed-power extension and a training-only rho-boundary audit; HAB22 is not an untouched external confirmation of RFDT-P',
          'utility':'u_rho(x)=sign(x)|x|^rho',
          'rho_bounds':[float(POWER_BOUNDS['rho'][0]),float(POWER_BOUNDS['rho'][1])],
          'rho_domain_rationale':'lower numerical bound opened from 0.25 to 0.05 solely because the prior HAB22 training-only boundary audit showed interior optima just below 0.25; no held-out outcomes were consulted for this change',
          'utility_scale':'for every rho, divide comparative evidence by dataset-wide median task RMS(D_rho), using stimuli only; median=1',
          'sampling':'independent/product hypothetical sampling',
          'readout':'P(B)=logistic(lambda*m_star)',
          'structural_regime':'0 <= kappa < 1',
          'numerical_kappa_upper':float(POWER_BOUNDS['kappa'][1]),
          'bounds':POWER_BOUNDS,
          'task_folds':'stored 10 crossValidation_id folds',
          'participant_folds':'exact Plonsky set.seed(123) 5-fold recreation within context',
          'training':'task fold != test AND participant fold != test',
          'test':'task fold == test AND participant fold == test',
          'training_objective':'binomial NLL on training task counts',
          'power_models':['RFDT-P','RFDT-P rho=1','RFDT-P kappa=0','RFDT-P alpha=0','RFDT-P alpha=1'],
          'comparators':'same 60 stored behavioral models + BEAST + stored BEAST-GB',
          'no_test_tuning':True,
          'targeted_lowrho_recovery':{
              'rho_truths':[0.15,0.22,0.30],
              'n_structural_settings':6,
              'purpose':'verify local recovery in the newly admitted low-rho region and around the HAB22 fitted neighborhood; synthetic outcomes only'
          },
          'individual_average_sensitivity':bool(a.individual),
          'interpretation_rule':'This closes HAB22 model development. No further static RFDT functional changes will be selected from HAB22. Choices13k is the first external test of this frozen RFDT-P specification.'}
    with open(a.out/'analysis_spec.json','w') as f:json.dump(spec,f,indent=2)

    if not a.skip_recovery:
        rs=run_recovery(work,oldrec,a.jobs,a.recovery_reps,a.recovery_starts,a.out);print('\nRecovery summary:\n'+rs.to_string(index=False),flush=True)
    if a.skip_empirical:
        print(f'Saved recovery to {a.out.resolve()}');return

    bench_cells,bench_preds=eval_stored(hab,gb,behavioral_cols)
    labels={'power':'RFDT-P','rho1':'RFDT-P rho=1','kappa0':'RFDT-P kappa=0','alpha0':'RFDT-P alpha=0','alpha1':'RFDT-P alpha=1'}
    todo=[(str(work),cv,sf,m,a.starts) for cv in range(1,11) for sf in range(1,6) for m in labels]
    fitrows=[];cellrows=[];predrows=[];print(f'FINAL pooled RFDT-P benchmark: {len(todo)} fits, jobs={a.jobs}, starts={a.starts+2}',flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs=[ex.submit(fit_one_pooled,x) for x in todo]
        for i,f in enumerate(as_completed(futs),1):
            cv,sf,m,res,starts,aa,bb,kk,ll,rr,scale,te=f.result();label=labels[m];mse,naive,irr,comp,logloss=metrics(te)
            cellrows.append(dict(model=label,cvid=cv,subject_fold=sf,mse=mse,naive=naive,irreducible=irr,completeness=comp,logloss=logloss))
            fitrows.append(dict(model=label,cvid=cv,subject_fold=sf,alpha=aa,beta=bb,kappa=kk,lambda_=ll,rho=rr,train_nll=float(res.fun),success=bool(res.success),nit=int(res.nit),weak_beta_flag=bb<.5,
                                alpha_at_bound=(m=='power' and (abs(aa)<1e-6 or abs(aa-1)<1e-6)),kappa_near_one=(m!='kappa0' and kk>.99),rho_at_bound=(m!='rho1' and (abs(rr-POWER_BOUNDS['rho'][0])<1e-6 or abs(rr-POWER_BOUNDS['rho'][1])<1e-6))))
            for r in te.itertuples(index=False):predrows.append(dict(model=label,cvid=cv,subject_fold=sf,task_id=r.task_id,context=r.dataset,obs_rate=r.rate,pred=r.pred,n=int(r.n)))
            if i%25==0:print(f'  completed {i}/{len(todo)}',flush=True)
    rc=pd.DataFrame(cellrows);rp=pd.DataFrame(predrows);fits=pd.DataFrame(fitrows);cells=pd.concat([bench_cells,rc],ignore_index=True);preds=pd.concat([bench_preds,rp.drop(columns=['context'])],ignore_index=True)
    fits.to_csv(a.out/'hab22_final_fit_parameters.csv',index=False);cells.to_csv(a.out/'hab22_final_cell_metrics.csv',index=False);preds.to_csv(a.out/'hab22_final_oof_predictions.csv',index=False);summarize(cells).to_csv(a.out/'hab22_final_model_ranking.csv',index=False)

    # Context diagnostics are descriptive only and never used for tuning.
    cp=rp.copy();cp['sqerr']=(cp.pred-cp.obs_rate)**2
    ctx=cp.groupby(['model','context'],as_index=False).agg(mse=('sqerr','mean'),n_rows=('sqerr','size'));ctx.to_csv(a.out/'hab22_final_context_mse.csv',index=False)
    # Nested held-out differences against rho=1 on identical task/cell predictions.
    base=rp[rp.model=='RFDT-P rho=1'][['cvid','subject_fold','task_id','pred','obs_rate']].rename(columns={'pred':'pred_rho1'})
    powr=rp[rp.model=='RFDT-P'][['cvid','subject_fold','task_id','pred','context']].rename(columns={'pred':'pred_power'})
    z=powr.merge(base,on=['cvid','subject_fold','task_id'],how='inner');z['se_power']=(z.pred_power-z.obs_rate)**2;z['se_rho1']=(z.pred_rho1-z.obs_rate)**2;z['delta_se_power_minus_rho1']=z.se_power-z.se_rho1
    z.to_csv(a.out/'hab22_final_vs_rho1_taskcell_differences.csv',index=False)

    if a.individual:
        run_individual(work,hab,a.jobs,a.individual_starts,a.out);ind=pd.read_csv(a.out/'hab22_final_individual_cell_metrics.csv');summarize(pd.concat([cells,ind],ignore_index=True)).to_csv(a.out/'hab22_final_model_ranking_with_individual.csv',index=False)

    manifest={
        'hab22_closed_after_this_run':True,
        'final_rho_bounds':[float(POWER_BOUNDS['rho'][0]),float(POWER_BOUNDS['rho'][1])],
        'final_kappa_bounds':[float(POWER_BOUNDS['kappa'][0]),float(POWER_BOUNDS['kappa'][1])],
        'next_static_dataset':'Choices13k',
        'no_further_HAB22_model_changes':True
    }
    with open(a.out/'HAB22_STATIC_MODEL_FREEZE.json','w') as f:json.dump(manifest,f,indent=2)

    print('\nTop models:\n'+summarize(cells).head(15).to_string(index=False),flush=True)
    print('\nRFDT-P parameter diagnostics:\n'+fits.groupby('model')[['alpha','beta','kappa','lambda_','rho','weak_beta_flag','alpha_at_bound','kappa_near_one','rho_at_bound']].mean(numeric_only=True).to_string(),flush=True)
    print(f'\nSaved to {a.out.resolve()}')
if __name__=='__main__':main()
