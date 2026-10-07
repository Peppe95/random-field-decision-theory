"""Corrective production pipeline. Historical analysis files are never modified."""
from __future__ import annotations
import concurrent.futures,gzip,io,json,math,multiprocessing,os,shutil,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from numba import set_num_threads,get_num_threads
from scipy.special import logsumexp
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
from . import core
from .io import (ROOT,MODELS,TARGETS,SAMPLES,atomic_npz,banks,dump_json,evidence_arrays,
                 load_json,load_sample,read_bank,read_run,sha,sources,versions)
from .simulation import simulate_global,seeds_for_cells
from .ddm import make_ddm_table


def count_path(run,level,batch,model,g):
    return Path(run)/'atlases'/level/batch/model/('g%04d.npz'%g)


def _global_identity(row,model):
    return tuple(float(row[x]) for x in ['alpha','kappa','N_eff']) if model!='DDM' else (0.,)


def _ensure_ddm(run,level,config,design,cutoffs):
    dest=run/'atlases'/level/'DDM.npz'
    if dest.exists():return
    _,h,_=read_bank(run,level,'DDM')
    print('DDM analytic table: %d candidates x 63 problems'%len(h),flush=True)
    q,meta=make_ddm_table(h,design,cutoffs,config['ddm']['absolute_series_tolerance'])
    qt,_=make_ddm_table(h,design,cutoffs,config['ddm']['check_tolerance'])
    discrepancy=float(np.abs(q-qt).max())
    if discrepancy>config['ddm']['domain_max_allowed_discrepancy']:
        raise ArithmeticError('DDM tolerance convergence failed on this bank.')
    atomic_npz(dest,joint=q)
    dump_json(dest.with_suffix('.json'),{'source_hash':sources(),'file_sha256':sha(dest),
               'tolerance_discrepancy_max':discrepancy,'chunks':meta,'invariants':core.check_joint(q)})


def simulate_atlas(run,level='base',batch='train_a',threads=8,selected=None,paths=None,phase=None):
    run,config,design,folds,cuts=read_run(run)
    _ensure_ddm(run,level,config,design,cuts)
    if paths is None:paths=int(config['simulation']['train_paths'])
    if phase is None:phase=int(config['simulation']['train_phases'][batch])
    set_num_threads(max(1,int(threads)))
    start_all=time.time()
    for model in ('RFDT','RFDT_K0'):
        g,h,u=read_bank(run,level,model)
        idx=list(range(len(g))) if selected is None else sorted(selected.get(model,[]))
        print('%s / %s / %s: %d globals, %d nuisance candidates, %d paths/cell, %d threads'%(
            level,batch,model,len(idx),len(h),paths,get_num_threads()),flush=True)
        D=[];pr=None
        for rho in h.rho:
            d,p,sc=evidence_arrays(design,float(rho));D.append(d);pr=p
        D=np.array(D);ha=h[['beta','bound','tau_s','t0']].to_numpy(float)
        base=core.rfdt_base_measure(h,cuts)
        model_start=time.time();done=0
        for gi in idx:
            out=count_path(run,level,batch,model,gi)
            row=g.iloc[gi]
            if out.exists():
                with np.load(out) as z:
                    if int(z['paths'])!=paths or int(z['phase'])!=phase:raise ValueError('Stored path/batch mismatch '+str(out))
                    if z['counts'].shape!=(len(h),63,14):raise ValueError('Stored count shape mismatch.')
                done+=1;continue
            t=time.time();prefix=0
            counts=np.zeros((len(h),63,14),np.uint32);unresolved=np.zeros((len(h),63),np.uint32)
            beyond=np.zeros_like(unresolved);maxt=np.zeros((len(h),63),float)
            # The predetermined larger bank reuses simulation prefixes, not
            # changed outcomes or fitted weights. This isolates bank resolution.
            if level=='expanded' and batch in ('train_a','train_b'):
                bg,bh,bu=read_bank(run,'base',model)
                old=count_path(run,'base',batch,model,gi)
                if gi<len(bg) and old.exists() and _global_identity(bg.iloc[gi],model)==_global_identity(row,model):
                    if not np.allclose(h.iloc[:len(bh)][bh.columns].to_numpy(float),bh.to_numpy(float),atol=1e-14,rtol=1e-14):
                        raise ValueError('Expanded candidate bank changed its prefix.')
                    with np.load(old) as z:
                        if int(z['paths'])==paths and int(z['phase'])==phase:
                            prefix=len(bh);counts[:prefix]=z['counts'];unresolved[:prefix]=z['unresolved']
                            beyond[:prefix]=z['beyond_90'];maxt[:prefix]=z['max_internal_time']
            seed=seeds_for_cells(gi,len(h),63,model,phase)[prefix:]
            cc,uu,bb,mm=simulate_global(float(row.alpha),float(row.kappa),int(row.N_eff),
                             ha[prefix:],D[prefix:],pr,cuts,paths,float(config['simulation']['horizon']),seed)
            counts[prefix:]=cc;unresolved[prefix:]=uu;beyond[prefix:]=bb;maxt[prefix:]=mm
            if int(unresolved.sum())>0:
                fail=out.with_suffix('.unresolved.npz')
                atomic_npz(fail,counts=counts,unresolved=unresolved,beyond_90=beyond,max_internal_time=maxt)
                raise RuntimeError('Unresolved trajectories at the extended computational horizon; saved '+str(fail)+'. No probabilities or choices were imputed. Return this diagnostic.')
            if not np.all(counts.sum(axis=-1)==paths):raise ArithmeticError('Lost simulated trials.')
            q=core.joint_from_counts(counts,base,config['smoothing']['total_pseudocount_mass'])
            core.for_target(q,'window')
            atomic_npz(out,counts=counts,unresolved=unresolved,beyond_90=beyond,max_internal_time=maxt,
                       paths=np.array(paths),phase=np.array(phase),global_index=np.array(gi))
            done+=1
            dump_json(out.with_suffix('.json'),{'sha256':sha(out),'seconds':time.time()-t,'prefix_reused':prefix,
                     'global':row.to_dict(),'invariants':core.check_joint(q),
                     'minimum_cell_events':int(counts.min()),'paths_beyond_90':int(beyond.sum()),
                     'maximum_observed_internal_time':float(maxt.max())})
            rate=(time.time()-model_start)/max(1,done)
            print('  %s global %d/%d (index %d): %.1fs; %.1f min elapsed; approx %.1f min remaining'%(
                    model,done,len(idx),gi,time.time()-t,(time.time()-model_start)/60,rate*(len(idx)-done)/60),flush=True)
        print('  completed',model,flush=True)
    marker=run/'atlases'/level/batch/'ATLAS_COMPLETE.json'
    dump_json(marker,{'level':level,'batch':batch,'paths':paths,'phase':phase,
                     'selected_globals':{k:sorted(v) for k,v in selected.items()} if selected else 'all',
                     'seconds_this_invocation':time.time()-start_all,'source_hashes':sources()})
    print('Atlas stage complete:',marker,flush=True)


def joint_for(run,level,batches,model,g,h,cutoffs,mass):
    if model=='DDM':
        with np.load(Path(run)/'atlases'/level/'DDM.npz') as z:return z['joint']
    if isinstance(batches,str):batches=[batches]
    counts=None
    for b in batches:
        with np.load(count_path(run,level,b,model,g)) as z:
            c=z['counts'].astype(np.uint64)
            if np.any(z['unresolved']):raise ValueError('Unresolved counts cannot enter a likelihood.')
            counts=c if counts is None else counts+c
    return core.joint_from_counts(counts,core.rfdt_base_measure(h,cutoffs),mass)


def fit_root(run,level,batch,sample,target,model):
    return Path(run)/'fits'/level/batch/target/sample/model


def _fit_global_job(job):
    run,level,batch,sample,target,model,gi=job
    with threadpool_limits(limits=1):
        run,config,design,folds,cuts=read_run(run)
        gd,h,X=read_bank(run,level,model)
        data=load_sample(run,sample,design,folds,cuts)
        events=data['events'][target]
        q=core.for_target(joint_for(run,level,batch,model,gi,h,cuts,config['smoothing']['total_pseudocount_mass']),target)
        dest=fit_root(run,level,batch,sample,target,model)/('g%04d'%gi)
        dest.mkdir(parents=True,exist_ok=True)
        for fold in range(7):
            objfile=dest/('fold%d.npz'%fold);diagfile=dest/('fold%d.json'%fold)
            if objfile.exists() and diagfile.exists():
                diag=load_json(diagfile)
                if not diag['kkt_pass']:raise RuntimeError('Stored optimization failed diagnostic: '+str(diagfile))
                continue
            L=core.training_likelihood(q,events,folds,fold)
            theta,post,ln,diag=core.fit_finite_bank(L,X,config['eb'])
            train_n=(events[:,folds!=fold]>=0).sum(axis=1)
            prior=np.exp(core.log_prior(X,theta))
            # Full float64 conditional nuisance weights, including participants
            # with zero training rows. No rounded parameter summary substitutes.
            atomic_npz(objfile,theta=theta,posterior=post,prior=prior,
                       participant_log_marginal=ln,train_counts=train_n,
                       score=np.array(ln.sum()),global_index=np.array(gi),fold=np.array(fold))
            diag.update({'sample':sample,'target':target,'model':model,'global_index':gi,'fold':fold,
                         'object_sha256':sha(objfile),'candidate_invariants':core.check_joint(q)})
            dump_json(diagfile,diag)
            if not diag['kkt_pass']:
                raise RuntimeError('Finite-bank optimizer did not satisfy KKT tolerance. '+str(diagfile)+
                                   ' preserved; no held-out score should be interpreted.')
        return gi


def _collect_globals(run,level,batch,sample,target,model):
    g,h,X=read_bank(run,level,model)
    scores=np.empty((7,len(g)),float)
    for gi in range(len(g)):
        for f in range(7):
            p=fit_root(run,level,batch,sample,target,model)/('g%04d'%gi)/('fold%d.npz'%f)
            with np.load(p) as z:scores[f,gi]=float(z['score'])
    logprior=np.log(g.global_prior_mass.to_numpy(float))
    logw=scores+logprior[None,:]
    logw-=logsumexp(logw,axis=1)[:,None]
    w=np.exp(logw);w/=w.sum(axis=1,keepdims=True)
    p=fit_root(run,level,batch,sample,target,model)/'global_weights.npz'
    atomic_npz(p,weights=w,logweights=logw,scores=scores,prior=g.global_prior_mass.to_numpy(float))
    rows=[]
    for f in range(7):
        for gi,row in g.iterrows():
            rows.append({'fold':f,'global_index':gi,'weight':float(w[f,gi]),'logweight':float(logw[f,gi]),
                         'train_log_marginal':float(scores[f,gi]),**row.to_dict()})
    pd.DataFrame(rows).to_csv(p.with_suffix('.csv'),index=False)
    return w


def predict_with_weights(run,level,fit_batch,sample,target,model,count_batches,outfile,mass=None):
    run,config,design,folds,cuts=read_run(run)
    gd,h,X=read_bank(run,level,model)
    data=load_sample(run,sample,design,folds,cuts);events=data['events'][target]
    n,P=events.shape
    if mass is None:mass=config['smoothing']['total_pseudocount_mass']
    root=fit_root(run,level,fit_batch,sample,target,model)
    with np.load(root/'global_weights.npz') as z:w=z['weights']
    pred=np.zeros((n,P,14),float)
    active=np.flatnonzero(np.any(w>0,axis=0))
    for gi in active:
        q=core.for_target(joint_for(run,level,count_batches,model,gi,h,cuts,mass),target)
        for fold in range(7):
            if w[fold,gi]==0:continue
            with np.load(root/('g%04d'%gi)/('fold%d.npz'%fold)) as z:post=z['posterior']
            for j in np.flatnonzero(folds==fold):
                idx=np.flatnonzero(events[:,j]>=0)
                if len(idx):pred[idx,j]+=w[fold,gi]*(post[idx] @ q[:,j,:])
    ii,jj=np.where(events>=0)
    p=pred[ii,jj];checks=core.check_joint(p)
    pc,pe,lj,lc,lr=core.split_scores(p,data['choice'][ii,jj],data['bin'][ii,jj])
    frame=pd.DataFrame({'participant_id':np.array(data['participant_ids'])[ii],
                        'problem_id':design.problem_id.to_numpy()[jj],'fold':folds[jj],
                        'choice_risky':data['choice'][ii,jj],'rt':data['rt'][ii,jj],
                        'click_count':data['click_count'][ii,jj], 'rt_bin':data['bin'][ii,jj],
                        'sample':sample,'target':target,'model':model,'stage':'corrected_refit',
                        'p_risky':pc[:,1],'p_observed_event':pe,'log_score':lj,
                        'log_choice':lc,'log_rt_given_choice':lr})
    for c in range(2):
        for b in range(7):frame['q_c%d_b%d'%(c,b)]=p[:,c*7+b]
    cq=p.reshape(-1,2,7)[np.arange(len(ii)),data['choice'][ii,jj]]/pc[np.arange(len(ii)),data['choice'][ii,jj]][:,None]
    for b in range(7):frame['p_bin%d_given_observed_choice'%b]=cq[:,b]
    outfile=Path(outfile);outfile.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(outfile,index=False,compression='gzip')
    dump_json(outfile.with_name(outfile.name+'.checks.json'),{**checks,'rows':len(frame),
            'n_canonical_participants':n,'n_contributing_participants':frame.participant_id.nunique(),
            'n_globals_numerically_nonzero':len(active),'pseudo_mass':mass,
            'weight_source':str(root),'count_batches':count_batches,
            'window_convention':config['window_interpretation'] if target=='window' else 'unconditional coarsened RT',
            'decomposition_error':float(np.abs(lj-lc-lr).max()),'file_sha256':sha(outfile)})
    return frame


def calibration(pred,outprefix):
    p=pred
    q=p[['q_c%d_b%d'%(c,b) for c in range(2) for b in range(7)]].to_numpy().reshape(-1,2,7)
    rows=[]
    for b in range(7):
        rows.append({'bin':b,'n_observed':int((p.rt_bin==b).sum()),'n_total':len(p),
                     'observed_fraction':float((p.rt_bin==b).mean()),
                     'mean_joint_marginal_bin_probability':float(q[:,:,b].sum(axis=1).mean()),
                     'mean_bin_probability_given_observed_choice':float(p['p_bin%d_given_observed_choice'%b].mean())})
    pd.DataFrame(rows).to_csv(str(outprefix)+'_rt.csv',index=False)
    # Calibration groups depend on predictions, not on outcome-driven bin edges.
    group=np.minimum((p.p_risky.to_numpy()*10).astype(int),9)
    pp=p.assign(probability_bin=group)
    pp.groupby('probability_bin').agg(n=('p_risky','size'),mean_predicted=('p_risky','mean'),
                                    observed_risky=('choice_risky','mean')).reset_index().to_csv(str(outprefix)+'_choice.csv',index=False)


def compare_predictions(predA,predB,canonical_pids,reps,seed):
    keys=['participant_id','problem_id']
    A=predA.sort_values(keys).reset_index(drop=True);B=predB.sort_values(keys).reset_index(drop=True)
    for col in keys+['fold','choice_risky','rt_bin','click_count','target','sample']:
        if not np.array_equal(A[col].to_numpy(),B[col].to_numpy()):raise ValueError('Paired observations differ in '+col)
    ds=A.log_score.to_numpy()-B.log_score.to_numpy()
    temp=pd.DataFrame({'participant_id':A.participant_id,'delta_sum':ds})
    part=temp.groupby('participant_id').agg(delta_sum=('delta_sum','sum'),n=('delta_sum','size')).reindex(canonical_pids).fillna(0).reset_index()
    delta,lo,hi=core.paired_cluster_interval(part.delta_sum,part.n,reps,seed)
    nz=part.n>0
    summary={'sample':A['sample'].iloc[0],'target':A['target'].iloc[0],
             'model_A':A['model'].iloc[0],'model_B':B['model'].iloc[0],
             'n_canonical_participants':len(part),'n_contributing_participants':int(nz.sum()),'n_trials':len(A),
             'mean_log_score_A':float(A.log_score.mean()),'mean_log_score_B':float(B.log_score.mean()),
             'A_minus_B':delta,'ci_low':lo,'ci_high':hi,
             'participant_A_win_fraction':float((part.loc[nz,'delta_sum']>0).mean()),
             'choice_log_loss_A':float(-A.log_choice.mean()),'choice_log_loss_B':float(-B.log_choice.mean()),
             'RT_log_score_A':float(A.log_rt_given_choice.mean()),'RT_log_score_B':float(B.log_rt_given_choice.mean()),
             'choice_component':float((A.log_choice-B.log_choice).mean()),
             'RT_component':float((A.log_rt_given_choice-B.log_rt_given_choice).mean()),
             'brier_A':float(np.mean((A.p_risky-A.choice_risky)**2)),
             'brier_B':float(np.mean((B.p_risky-B.choice_risky)**2))}
    fold=[]
    for f in range(7):
        m=A.fold.to_numpy()==f
        fold.append({'fold':f,'n':int(m.sum()),'difference_sum':float(ds[m].sum()),
                     'A_minus_B':float(ds[m].mean()) if m.any() else np.nan})
    tail=[]
    for label,m in [('short_tail',A.rt_bin.to_numpy()==0),('window',A.rt_bin.between(1,5).to_numpy()),('long_tail',A.rt_bin.to_numpy()==6)]:
        tail.append({'region':label,'n':int(m.sum()),'fraction':float(m.mean()),
                     'delta_sum':float(ds[m].sum()),'contribution_per_all_trials':float(ds[m].sum()/len(ds)),
                     'conditional_region_mean_difference':float(ds[m].mean()) if m.any() else np.nan})
    if abs(summary['choice_component']+summary['RT_component']-delta)>2e-12:raise ArithmeticError('Aggregate decomposition error.')
    return summary,part,pd.DataFrame(fold),pd.DataFrame(tail)


def summarize_directory(run,predroot,outdir,config,design,folds,cuts,compute_ci=True):
    summaries=[];outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    for target in TARGETS:
        for sample in SAMPLES:
            data=load_sample(run,sample,design,folds,cuts)
            preds={m:pd.read_csv(Path(predroot)/target/sample/('predictions_'+m+'.csv.gz')) for m in MODELS}
            for ma,mb in [('RFDT','DDM'),('RFDT','RFDT_K0')]:
                s,p,f,t=compare_predictions(preds[ma],preds[mb],data['participant_ids'],
                                            config['bootstrap_reps'] if compute_ci else 200,config['seeds']['bootstrap'])
                summaries.append(s)
                prefix=outdir/(target+'_'+sample+'_'+ma+'_vs_'+mb)
                p.to_csv(str(prefix)+'_participants.csv',index=False)
                f.to_csv(str(prefix)+'_folds.csv',index=False)
                t.to_csv(str(prefix)+'_tails.csv',index=False)
            for model,pred in preds.items():calibration(pred,outdir/(target+'_'+sample+'_'+model))
    table=pd.DataFrame(summaries);table.to_csv(outdir/'comparisons.csv',index=False)
    return table


def fit_stage(run,level='base',batch='train_a',jobs=4):
    run,config,design,folds,cuts=read_run(run)
    if not (run/'atlases'/level/batch/'ATLAS_COMPLETE.json').exists():raise RuntimeError('Complete this atlas batch first.')
    start=time.time();predroot=run/'predictions'/level/batch
    for target in TARGETS:
        for sample in SAMPLES:
            for model in MODELS:
                gd,h,X=read_bank(run,level,model)
                # Analytic DDM likelihoods do not change between simulator batches.
                base_ddm=fit_root(run,level,'train_a',sample,target,model)
                dest=fit_root(run,level,batch,sample,target,model)
                if model=='DDM' and batch=='train_b' and (base_ddm/'global_weights.npz').exists():
                    if not dest.exists():shutil.copytree(base_ddm,dest)
                    print('Reusing identical analytic DDM fit for',target,sample,flush=True)
                else:
                    print('Fitting %s / %s / %s / %s / %s (%d global candidates)'%(
                        level,batch,target,sample,model,len(gd)),flush=True)
                    todo=[(str(run),level,batch,sample,target,model,int(gi)) for gi in range(len(gd))]
                    if jobs<=1:
                        for j,job in enumerate(todo):
                            _fit_global_job(job)
                            print('  fitted global %d/%d'%(j+1,len(todo)),flush=True)
                    else:
                        # Spawn avoids inheriting an OpenMP/BLAS thread state on macOS.
                        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs,
                                mp_context=multiprocessing.get_context('spawn')) as ex:
                            futures=[ex.submit(_fit_global_job,j) for j in todo]
                            for k,fu in enumerate(concurrent.futures.as_completed(futures)):
                                gi=fu.result()
                                print('  fitted global %d/%d (index %d)'%(k+1,len(todo),gi),flush=True)
                    _collect_globals(run,level,batch,sample,target,model)
                out=predroot/target/sample/('predictions_'+model+'.csv.gz')
                if not out.exists():
                    with threadpool_limits(limits=1):
                        predict_with_weights(run,level,batch,sample,target,model,batch,out)
    table=summarize_directory(run,predroot,run/'summaries'/level/batch,config,design,folds,cuts)
    dump_json(run/'summaries'/level/batch/'FIT_COMPLETE.json',{'seconds_this_invocation':time.time()-start,
                'status':'Corrected fit complete; simulation/grid precision NOT yet certified.',
                'source_hashes':sources(),'versions':versions()})
    print('Corrected fits complete. Numerical precision checks are still required.',flush=True)
    print(table[['target','sample','model_A','model_B','A_minus_B','ci_low','ci_high']].to_string(index=False),flush=True)


def precision_stage(run,level='base',fit_batch='train_a',threads=8,paths=None):
    """Independent evaluation of ALL numerically nonzero fitted global mixtures.

    Each batch is simulated independently of training and the other batches.
    Nuisance weights are kept fixed, NOT re-estimated from evaluation paths.
    For the pooled estimate, raw counts are added BEFORE the single smoothing
    operation. Delete-one-batch jackknife measures shared simulation error.
    """
    run,config,design,folds,cuts=read_run(run)
    if not (run/'summaries'/level/fit_batch/'FIT_COMPLETE.json').exists():raise RuntimeError('Fit the designated reference batch first.')
    if paths is None:paths=config['simulation']['evaluation_paths']
    allowed=config['simulation']['allowed_evaluation_paths']
    if paths not in allowed:raise ValueError('Evaluation path count must follow frozen schedule '+str(allowed))
    level_ix=allowed.index(paths);nb=config['simulation']['evaluation_batches']
    selected={m:set() for m in ('RFDT','RFDT_K0')}
    for target in TARGETS:
        for sample in SAMPLES:
            for model in selected:
                p=fit_root(run,level,fit_batch,sample,target,model)/'global_weights.npz'
                with np.load(p) as z:
                    selected[model].update(int(x) for x in np.flatnonzero(np.any(z['weights']>0,axis=0)))
    root=run/'precision'/level/('M%d'%paths)
    root.mkdir(parents=True,exist_ok=True)
    dump_json(root/'EVALUATION_PLAN.json',{'reference_fit':fit_batch,'global_selection':'all nonzero float64 global weights; no nuisance truncation',
            'selected_globals':{m:sorted(v) for m,v in selected.items()},'paths_per_batch':paths,'batches':nb,
            'phase_ids':[config['simulation']['evaluation_phase_start']+level_ix*nb+b for b in range(nb)],
            'selection_does_not_use_heldout_outcomes':True})
    names=[]
    for b in range(nb):
        batch='evaluation_M%d_b%d'%(paths,b);names.append(batch)
        phase=config['simulation']['evaluation_phase_start']+level_ix*nb+b
        simulate_atlas(run,level,batch,threads,selected,paths,phase)
    # Re-evaluate batch-specific, pooled and leave-one-batch-out probability tables.
    specs=[('batch_%d'%b,[n],.5) for b,n in enumerate(names)]
    specs+=[('pooled',names,.5)]
    specs += [('leave_%d'%b,[n for k,n in enumerate(names) if k!=b],.5) for b in range(nb)]
    specs += [('smoothing_%g'%a,names,a) for a in config['smoothing']['sensitivity_masses'] if a!=.5]
    means={}
    for label,countsets,mass in specs:
        predroot=root/label/'predictions'
        for target in TARGETS:
            for sample in SAMPLES:
                for model in MODELS:
                    outfile=predroot/target/sample/('predictions_'+model+'.csv.gz')
                    if not outfile.exists():
                        with threadpool_limits(limits=1):
                            predict_with_weights(run,level,fit_batch,sample,target,model,countsets,outfile,mass)
        table=summarize_directory(run,predroot,root/label/'summary',config,design,folds,cuts,
                                   compute_ci=(label=='pooled'))
        means[label]=table
    ref=means['pooled'];rows=[]
    keycols=['target','sample','model_A','model_B']
    for _,r in ref.iterrows():
        def get(label):
            z=means[label]
            mask=np.ones(len(z),bool)
            for k in keycols:mask&=z[k].to_numpy()==r[k]
            if mask.sum()!=1:raise ArithmeticError('Precision comparison keys differ.')
            return float(z.loc[mask,'A_minus_B'].iloc[0])
        leave=np.array([get('leave_%d'%b) for b in range(nb)])
        se=float(np.sqrt((nb-1)/nb*np.sum((leave-leave.mean())**2)))
        hw=float(student_t.ppf(.975,nb-1)*se)
        half=float((r.ci_high-r.ci_low)/2)
        shifts=[abs(get('smoothing_%g'%a)-float(r.A_minus_B))
                for a in config['smoothing']['sensitivity_masses'] if a!=.5]
        maxshift=max(shifts,default=0.)
        ok=(hw<=config['precision']['mc_95_halfwidth_nats_per_trial'] and
            hw<=config['precision']['mc_fraction_of_participant_CI_halfwidth']*half and
            maxshift<=config['precision']['smoothing_max_absolute_shift'])
        rows.append({**{k:r[k] for k in keycols},'pooled_A_minus_B':float(r.A_minus_B),
                     'paths_per_candidate_problem_pooled':int(paths*nb),
                     'jackknife_MC_SE':se,'MC_t95_halfwidth':hw,
                     'participant_bootstrap_halfwidth':half,
                     'mean_batch_score_minus_pooled_score':float(np.mean([get('batch_%d'%b) for b in range(nb)])-r.A_minus_B),
                     'max_fixed_weight_smoothing_shift':maxshift,'evaluation_precision_pass':bool(ok)})
    tab=pd.DataFrame(rows);tab.to_csv(root/'precision_report.csv',index=False)
    dump_json(root/'PRECISION_STATUS.json',{'all_pass':bool(tab.evaluation_precision_pass.all()),
                  'scope':'Independent fixed-weight probability evaluation only; does not test training-atlas noise or bank resolution.',
                  'batches':nb,'paths':paths,
                  'next_prespecified_paths':allowed[level_ix+1] if level_ix+1<len(allowed) else None})
    print(tab.to_string(index=False),flush=True)
    return tab


def convergence_report(run):
    """Keep stochastic training, candidate integration, and evaluation checks separate."""
    run,config,design,folds,cuts=read_run(run)
    base=run/'summaries/base/train_a/comparisons.csv'
    if not base.exists():raise RuntimeError('No completed base training-A summary.')
    A=pd.read_csv(base);rows=[];keys=['sample','target','model_A','model_B']
    for label,path in [('independent_training_replica',run/'summaries/base/train_b/comparisons.csv'),
                       ('predetermined_bank_expansion',run/'summaries/expanded/train_a/comparisons.csv')]:
        if not path.exists():
            rows.append({'check':label,'status':'NOT_RUN','pass':False});continue
        B=pd.read_csv(path)
        m=A.merge(B,on=keys,suffixes=('_reference','_check'),validate='one_to_one')
        for r in m.to_dict('records'):
            diff=float(r['A_minus_B_check']-r['A_minus_B_reference'])
            half=float((r['ci_high_reference']-r['ci_low_reference'])/2)
            limit=min(config['precision']['training_replica_or_grid_max_absolute_shift'],
                      config['precision']['training_replica_or_grid_fraction_of_participant_CI_halfwidth']*half)
            rows.append({'check':label,**{k:r[k] for k in keys},'status':'MEASURED',
                         'reference_A_minus_B':float(r['A_minus_B_reference']),
                         'check_A_minus_B':float(r['A_minus_B_check']),
                         'change':diff,'absolute_change_limit':limit,'pass':bool(abs(diff)<=limit)})
    evals=sorted((run/'precision/base').glob('M*/precision_report.csv')) if(run/'precision/base').exists() else []
    evaluation_pass=False
    if evals:
        latest=max(evals,key=lambda p:int(p.parent.name[1:]))
        ev=pd.read_csv(latest);evaluation_pass=bool(ev.evaluation_precision_pass.all())
    else:latest=None
    table=pd.DataFrame(rows);table.to_csv(run/'convergence_report.csv',index=False)
    passed=bool(evaluation_pass and all(bool(r['pass']) for r in rows))
    dump_json(run/'CORRECTION_STATUS.json',{'numerical_checks_pass':passed,
       'fixed_weight_evaluation_report':str(latest) if latest else None,
       'full_production_empirical_results_have_been_run':True,
       'note':'A passed tolerance is a practical resolution statement, not a proof of continuous-parameter convergence or model validity. '
              'If any check fails, preserve all outputs and return the diagnostics; do not select the more favorable result.'})
    print(table.to_string(index=False),flush=True)
    print('Numerical checks satisfied:',passed,flush=True)
    return passed


def package_results(run,out,include_weights=False):
    run,config,design,folds,cuts=read_run(run,require_validation=False)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(str(out)+' already exists. Choose a different output filename; historical packages are not overwritten.')
    files=[];excluded=[]
    for p in run.rglob('*'):
        if not p.is_file() or p.suffix=='.tmp':continue
        rel=str(p.relative_to(run))
        if rel.startswith('atlases/'):
            excluded.append(p);continue
        if rel.startswith('fits/') and p.suffix=='.npz' and p.name.startswith('fold') and not include_weights:
            excluded.append(p);continue
        # Keep only pooled high-precision predictions in the compact handoff;
        # batch/leave-out numeric arrays remain local, summaries are included.
        if rel.startswith('precision/') and '/predictions/' in rel and '/pooled/' not in rel:
            excluded.append(p);continue
        files.append(p)
    manifest=[{'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    local=[{'path':str(p.relative_to(run)),'bytes':p.stat().st_size,
            'reason':'raw atlas, full weights, or independent-batch prediction retained locally'} for p in excluded]
    readme='''# Corrected Guo results\n\nRead config.json, PROTOCOL_LOCK.json, validation reports and sample_flow.csv first.\n
These are POST-RESULT CORRECTIVE analyses, not a new preregistered confirmation.\n
The numerical accuracy checks are separate from participant-bootstrap sampling intervals.\n
Absent convergence_report.csv / CORRECTION_STATUS.json means the follow-up numerical checks are not complete.\n
Full conditional nuisance posterior weights and raw simulation counts are retained in the local run directory.\n
LOCAL_ARTIFACT_INVENTORY.csv lists them. Do not delete that directory.\n'''
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:z.write(p,str(p.relative_to(run)))
        z.writestr('PACKAGE_README.md',readme)
        z.writestr('PACKAGE_MANIFEST.csv',pd.DataFrame(manifest).to_csv(index=False))
        z.writestr('LOCAL_ARTIFACT_INVENTORY.csv',pd.DataFrame(local).to_csv(index=False))
        # Include inherited full source as provenance without importing it.
        for p in (ROOT/'legacy').iterdir():
            if p.is_file():z.write(p,'legacy_source/'+p.name)
    print('Wrote:',out,'(%.1f MB)'%(out.stat().st_size/1e6),flush=True)
    return out
