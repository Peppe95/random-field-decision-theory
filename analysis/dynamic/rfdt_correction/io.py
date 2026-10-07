"""Input validation, frozen banks and atomic file operations. No model fitting here."""
from __future__ import annotations
import hashlib,io,json,math,os,platform,shutil,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parent.parent
MODELS=('RFDT','RFDT_K0','DDM')
TARGETS=('single_click','window')
SAMPLES={'primary':'guo_exact_risk_main_LITERAL_paper_catch_rule.csv',
         'sensitivity':'guo_exact_risk_main_ARCHIVE_countmatch_1092.csv'}
EXPECTED_HASHES={'preprocessing':'a77cd6f5c446f32ede9c4b59402170114433150ded357a80d94e32645bfc5694',
                 'recovery':'4f4f09c24aa3d2b90c965c92af0f6b4bd4cf597a9ccd9debe699311f7ce399aa'}
EXPECTED_FLOW={'primary':(1049,66087,897,41022,39446),
               'sensitivity':(1092,68796,934,42522,40876)}

DEFAULT_CONFIG={
 'package_version':'1.0.0',
 'status':'Post-result corrective protocol; not preregistered confirmation',
 'models':list(MODELS),'targets':list(TARGETS),'samples':list(SAMPLES),
 'base':{'rfdt_global_continuous':96,'rfdt_nuisance':192,'ddm_nuisance':4096},
 'expanded':{'rfdt_global_continuous':128,'rfdt_nuisance':256,'ddm_nuisance':8192},
 'rfdt_supports':{'alpha':[.08,.92],'kappa':[.08,.90],'N_eff':[80,120,180],
                   'rho':[.05,1.50],'beta':[.20,1.60],'bound':[.12,.36],
                   'tau_s':[.75,1.25],'t0':[.05,.35]},
 'ddm_supports':{'rho':[.05,3.0],'vcoeff':[-100.,100.],'vmax':[.01,50.],
                 'a':[.10,8.],'z':[.02,.98],'t0':[.01,.29]},
 'seeds':{'rfdt_global':202609133,'rfdt_nuisance':202609134,'ddm_nuisance':202609132,
           'bootstrap':2026091801},
 'simulation':{'train_paths':2000,'horizon':5760.,'legacy_horizon':90.,
                'train_phases':{'train_a':0,'train_b':1},
                'evaluation_paths':4000,'evaluation_batches':4,
                'allowed_evaluation_paths':[4000,8000,16000],
                'evaluation_phase_start':16,'unresolved_allowed':0},
 'smoothing':{'total_pseudocount_mass':.5,'sensitivity_masses':[.05,.5,5.],
              'base':'uniform over physically possible cells of the 14-event partition',
              'ddm_smoothing':False},
 'eb':{'mu_bounds':[0.,1.],'sd_min':.055,'sd_max':.48,'starts':4,'maxiter':600,
        'ftol':1e-13,'gtol':1e-7,'kkt_tolerance':1e-4,
        'global_weights':'softmax(profile training marginal LL + discrete prior mass)',
        'global_prediction_truncation':False},
 'ddm':{'absolute_series_tolerance':1e-12,'check_tolerance':1e-14,
         'domain_max_allowed_discrepancy':1e-9},
 'bootstrap_reps':10000,
 'precision':{'mc_95_halfwidth_nats_per_trial':.001,
              'mc_fraction_of_participant_CI_halfwidth':.10,
              'training_replica_or_grid_max_absolute_shift':.005,
              'training_replica_or_grid_fraction_of_participant_CI_halfwidth':.25,
              'smoothing_max_absolute_shift':.001,
              'rule':'Report every predeclared comparison; never select a model by these checks.'},
 'window_interpretation':'mixture of candidate-level conditional-on-window distributions; refit on selected-window rows',
 'legacy_analysis':'optional and not part of this runner; old files are not edited or recycled as 14-cell tables'
}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temp,path)


def load_json(path):return json.loads(Path(path).read_text())


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)


def zip_member(path,suffix):
    with zipfile.ZipFile(path) as z:
        hits=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
        if len(hits)!=1:raise ValueError('Expected exactly one %s in %s; found %d'%(suffix,path,len(hits)))
        return z.read(hits[0])


def zip_csv(path,suffix):return pd.read_csv(io.BytesIO(zip_member(path,suffix)))


def sources():
    fs=[ROOT/'run_rfdt_guo_correction.py']+list((ROOT/'rfdt_correction').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(fs)}


def versions():
    import scipy,numba,threadpoolctl,mpmath
    return {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
            'pandas':pd.__version__,'scipy':scipy.__version__,'numba':numba.__version__,
            'mpmath':mpmath.__version__,'threadpoolctl':threadpoolctl.__version__}


def sobol(n,d,seed):
    return qmc.Sobol(d=d,scramble=True,seed=seed).random_base2(int(math.ceil(math.log2(max(n,2)))))[:n]


def lerp(u,r):return r[0]+(r[1]-r[0])*np.asarray(u)

def loglerp(u,r):return np.exp(lerp(u,np.log(r)))


def banks(config,level):
    sizes=config[level];r=config['rfdt_supports'];d=config['ddm_supports'];s=config['seeds']
    u=sobol(sizes['rfdt_global_continuous'],2,s['rfdt_global'])
    rows=[]
    for i,x in enumerate(u):
        for N in r['N_eff']:
            alpha=float(lerp(x[0],r['alpha']))
            rows.append({'alpha':alpha,'kappa':float(lerp(x[1],r['kappa'])),
                         'N_eff':N,'N_adaptive':int(np.rint(alpha*N)),'sobol_row':i})
    rg=pd.DataFrame(rows);rg['global_prior_mass']=1/len(rg)
    # Exactly identical kappa=0 processes are collapsed, retaining their total
    # projected quadrature prior mass (rather than silently changing alpha's prior).
    r0=rg.copy();r0['kappa']=0.
    r0=r0.groupby(['N_eff','N_adaptive'],sort=False,as_index=False).agg(
        {'alpha':'first','kappa':'first','sobol_row':'first','global_prior_mass':'sum'})
    hsize=sizes['rfdt_nuisance']
    ru=np.vstack([np.full((1,5),.5),sobol(hsize-1,5,s['rfdt_nuisance'])])
    rh=pd.DataFrame({n:lerp(ru[:,k],r[n]) for k,n in enumerate(['rho','beta','bound','tau_s','t0'])})
    dsize=sizes['ddm_nuisance']
    du=np.vstack([np.full((1,6),.5),sobol(dsize-1,6,s['ddm_nuisance'])])
    coeff=2*du[:,1]-1
    dh=pd.DataFrame({'rho':lerp(du[:,0],d['rho']),
                     'vcoeff':np.sign(coeff)*np.expm1(np.abs(coeff)*np.log1p(100.)),
                     'vmax':loglerp(du[:,2],d['vmax']),'a':loglerp(du[:,3],d['a']),
                     'z':lerp(du[:,4],d['z']),'t0':lerp(du[:,5],d['t0'])})
    dg=pd.DataFrame({'dummy':[0.],'global_prior_mass':[1.]})
    return {'RFDT':(rg,rh,ru),'RFDT_K0':(r0,rh.copy(),ru.copy()),'DDM':(dg,dh,du)}


def evidence_arrays(design,rho):
    p=design.probability.to_numpy(float);R=design.risky_reward.to_numpy(float);A=design.sure_reward.to_numpy(float)
    # Work from RAW rewards, not from normalized differences.
    ua=np.sign(A)*np.abs(A)**rho;ur=np.sign(R)*np.abs(R)**rho
    raw=np.column_stack([-ua,ur-ua]);pr=np.column_stack([1-p,p])
    scale=float(np.median(np.sqrt((pr*raw*raw).sum(axis=1))))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid utility normalization.')
    return raw/scale,pr,scale


def load_design(recovery):
    d=zip_csv(recovery,'guo_phase3bd_design.csv');d.problem_id=d.problem_id.astype(str)
    if len(d)!=63 or d.problem_id.duplicated().any():raise ValueError('Expected unique 63-problem design.')
    rmap={v:i for i,v in enumerate(sorted(d.risky_reward.unique()))}
    pmap={v:i for i,v in enumerate(sorted(d.probability.unique()))}
    if len(rmap)!=7 or len(pmap)!=9:raise ValueError('Not the 7 x 9 Guo grid.')
    folds=np.array([(rmap[R]+pmap[p])%7 for R,p in zip(d.risky_reward,d.probability)],int)
    if not np.array_equal(np.bincount(folds),np.full(7,9)):raise ValueError('Unbalanced modular folds.')
    old=zip_csv(recovery,'rt_bin_edges.csv');cuts=[]
    for pid in d.problem_id:
        g=old[old.problem_id==pid].sort_values('bin')
        if len(g)!=5:raise ValueError('Wrong legacy bin count for '+pid)
        inner=g.upper.to_numpy(float)[:4]
        if not np.all(np.diff(inner)>0) or not(.3<inner[0]<inner[-1]<10):raise ValueError('Interior cutoffs outside window.')
        cuts.append(np.r_[.3,inner,10.])
    return d,folds,np.array(cuts,float),old


def observations(df,design,folds,cutoffs,check_sample=None):
    required={'participant_id','problem_id','choice_risky','rt','change_mind'}
    if not required.issubset(df):raise ValueError('Missing observation columns.')
    d=df.copy();d.participant_id=d.participant_id.astype(str);d.problem_id=d.problem_id.astype(str)
    if d.duplicated(['participant_id','problem_id']).any():raise ValueError('Duplicate participant/problem pair.')
    pids=sorted(d.participant_id.unique());n=len(pids);P=len(design)
    if len(d)!=n*P:raise ValueError('Canonical table is not complete participant x problem grid.')
    index=pd.MultiIndex.from_product([pids,design.problem_id],names=['participant_id','problem_id'])
    d=d.set_index(['participant_id','problem_id']).reindex(index).reset_index()
    if d.choice_risky.isna().any():raise ValueError('Missing canonical row after reindex.')
    c=d.choice_risky.to_numpy(int).reshape(n,P)
    rt=d.rt.to_numpy(float).reshape(n,P);click=d.change_mind.to_numpy(float).reshape(n,P)
    if np.any((c!=0)&(c!=1)):raise ValueError('Choice must be 0=safe or 1=risky.')
    single=click==1
    if np.any(single & (~np.isfinite(rt)|(rt<0))):raise ValueError('Invalid single-click timing value; no automatic exclusion allowed.')
    b=np.full((n,P),-1,np.int8)
    for j in range(P):
        x=rt[:,j]
        z=np.searchsorted(cutoffs[j],x,side='right')
        z[x==10.]=5
        b[:,j]=z
    window=single & (rt>=.3)&(rt<=10)
    full_events=np.where(single,c*7+b,-1)
    window_events=np.where(window,c*7+b,-1)
    flow={'canonical_participants':n,'canonical_trials':len(d),
          'single_click_participants':int(single.any(axis=1).sum()),
          'single_click_trials':int(single.sum()),'window_trials':int(window.sum()),
          'short_tail_trials':int((single&(rt<.3)).sum()),
          'long_tail_trials':int((single&(rt>10)).sum()),
          'multiple_or_other_click_trials':int((~single).sum())}
    if check_sample:
        expected=EXPECTED_FLOW[check_sample]
        actual=tuple(flow[k] for k in ('canonical_participants','canonical_trials','single_click_participants','single_click_trials','window_trials'))
        if actual!=expected:raise ValueError('Sample-flow discrepancy: %s expected %s got %s'%(check_sample,expected,actual))
    return {'frame':d,'participant_ids':pids,'choice':c,'rt':rt,'click_count':click,'bin':b,
            'events':{'single_click':full_events,'window':window_events},'folds':folds,'flow':flow}


def read_bank(run,level,model):
    p=Path(run)/'banks'/level
    return pd.read_csv(p/(model+'_globals.csv')),pd.read_csv(p/(model+'_nuisance.csv')),np.load(p/(model+'_units.npy'))


def read_run(run,require_validation=True):
    run=Path(run)
    lock=load_json(run/'PROTOCOL_LOCK.json')
    if lock['source_hashes']!=sources():raise RuntimeError('Code has changed since protocol freeze. Use a new run directory; never mix versions.')
    if sha(run/'config.json')!=lock['config_sha256']:raise RuntimeError('Frozen config changed.')
    if require_validation and not(run/'validation/VALIDATION_PASSED.json').exists():raise RuntimeError('Run validate first.')
    config=load_json(run/'config.json')
    design=pd.read_csv(run/'design.csv');folds=np.load(run/'folds.npy');cuts=np.load(run/'cutoffs.npy')
    return run,config,design,folds,cuts


def load_sample(run,sample,design,folds,cuts):
    d=pd.read_csv(Path(run)/'inputs'/(sample+'.csv.gz'))
    return observations(d,design,folds,cuts,check_sample=None if load_json(Path(run)/'config.json').get('smoke',False) else sample)


def prepare(preprocessing,recovery,out,smoke=False):
    for name,p in [('preprocessing',preprocessing),('recovery',recovery)]:
        p=Path(p)
        if not p.is_file():raise FileNotFoundError('Missing %s ZIP: %s'%(name,p))
        if sha(p)!=EXPECTED_HASHES[name]:raise ValueError('Unexpected %s archive hash. Check which file you selected; do not bypass silently.'%name)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(str(out)+' already exists; use a new directory or continue with its next command.')
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+out.name+'_prepare_',dir=out.parent))
    config=json.loads(json.dumps(DEFAULT_CONFIG))
    if smoke:
        config['base']={'rfdt_global_continuous':2,'rfdt_nuisance':8,'ddm_nuisance':16}
        config['expanded']={'rfdt_global_continuous':3,'rfdt_nuisance':10,'ddm_nuisance':24}
        config['simulation']['train_paths']=30;config['simulation']['evaluation_paths']=50
        config['simulation']['allowed_evaluation_paths']=[50,100,200]
        config['eb']['maxiter']=150;config['eb']['starts']=2;config['bootstrap_reps']=200
    config['smoke']=bool(smoke)
    try:
        (temp/'inputs').mkdir();(temp/'source_snapshot').mkdir();(temp/'protocol').mkdir()
        dump_json(temp/'config.json',config)
        for rel in sources():
            dest=temp/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        shutil.copy2(ROOT/'protocol/NEXT_DYNAMIC_ANALYSIS_SPEC.md',temp/'protocol/NEXT_DYNAMIC_ANALYSIS_SPEC.md')
        shutil.copy2(ROOT/'protocol/IMPLEMENTATION_DECISIONS.md',temp/'protocol/IMPLEMENTATION_DECISIONS.md')
        dump_json(temp/'PROTOCOL_LOCK.json',{'status':config['status'],'source_hashes':sources(),
                  'config_sha256':sha(temp/'config.json'),'versions':versions(),
                  'input_sha256':{n:sha(p) for n,p in [('preprocessing',preprocessing),('recovery',recovery)]},
                  'corrected_human_fits_run':False})
        design,folds,cuts,old=load_design(recovery)
        design.to_csv(temp/'design.csv',index=False);np.save(temp/'folds.npy',folds);np.save(temp/'cutoffs.npy',cuts)
        old.to_csv(temp/'original_rt_bin_edges.csv',index=False)
        rows=[]
        for j,pid in enumerate(design.problem_id):
            for b in range(7):
                rows.append({'problem_id':pid,'bin':b,'lower':0. if b==0 else cuts[j,b-1],
                             'upper':np.inf if b==6 else cuts[j,b],
                             'label':['short_tail','window_1','window_2','window_3','window_4','window_5','long_tail'][b]})
        pd.DataFrame(rows).to_csv(temp/'observation_bins.csv',index=False)
        flow=[]
        for sample,basename in SAMPLES.items():
            df=zip_csv(preprocessing,basename)
            data=observations(df,design,folds,cuts,sample)
            df.to_csv(temp/'inputs'/(sample+'.csv.gz'),index=False,compression='gzip')
            flow.append({'sample':sample,**data['flow']})
        pd.DataFrame(flow).to_csv(temp/'sample_flow.csv',index=False)
        for level in ('base','expanded'):
            p=temp/'banks'/level;p.mkdir(parents=True)
            for model,(g,h,u) in banks(config,level).items():
                h['utility_scale']=[evidence_arrays(design,float(r))[2] for r in h.rho]
                g.to_csv(p/(model+'_globals.csv'),index=False);h.to_csv(p/(model+'_nuisance.csv'),index=False)
                np.save(p/(model+'_units.npy'),u)
        os.replace(temp,out)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    print('Prepared and locked:',out,flush=True)
    print(pd.DataFrame(flow).to_string(index=False),flush=True)
    return out
