"""Numerical and synthetic software checks; no human outcomes used for fitting."""
from __future__ import annotations
import ast,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import mpmath as mp
from .io import (ROOT,DEFAULT_CONFIG,MODELS,atomic_npz,banks,dump_json,evidence_arrays,
                 load_json,read_bank,read_run,sha,sources,versions)
from . import core,ddm


def check_engine_source():
    spec=load_json(ROOT/'legacy/ENGINE_PROVENANCE.json')
    src=ROOT/'legacy/run_rfdt_phase2b_recovery_v3.py'
    if sha(src)!=spec['source_sha256']:raise AssertionError('Inherited engine source changed.')
    new=ROOT/'rfdt_correction/engine.py'
    lines=new.read_text().splitlines(keepends=True);tree=ast.parse(new.read_text());found={}
    import hashlib
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in spec['extracted_function_sha256']:
            first=min([d.lineno for d in node.decorator_list]+[node.lineno])
            found[node.name]=hashlib.sha256(''.join(lines[first-1:node.end_lineno]).encode()).hexdigest()
    if found!=spec['extracted_function_sha256']:raise AssertionError('An inherited RFDT function was altered.')
    return {'exact_function_hash_match':True,'functions':found,'source_sha256':spec['source_sha256']}


def mp_lower(v,a,z,t):
    """Independent 90-digit evaluation; no double-precision clipping/normalization."""
    with mp.workdps(90):
        v,a,z,t=[mp.mpf(str(x)) for x in (v,a,z,t)]
        vv=v*a;tt=t/(a*a)
        if vv==0:hit=1-z
        else:hit=(mp.exp(-2*vv*z)-mp.exp(-2*vv))/(1-mp.exp(-2*vv))
        if t<=0:return mp.mpf(0),hit
        if tt<mp.mpf('.2'):
            nu=abs(vv)
            def Phi(x):return mp.erfc(-x/mp.sqrt(2))/2
            def term(k):
                y=z+2*k;d=abs(y)
                return mp.sign(y)*mp.exp(-vv*z)*(mp.exp(-nu*d)*Phi((nu*tt-d)/mp.sqrt(tt))+
                                                         mp.exp(nu*d)*Phi((-nu*tt-d)/mp.sqrt(tt)))
            terms=[term(0)]
            for k in range(1,80):
                terms+=[term(k),term(-k)]
                if k>8 and max(abs(terms[-1]),abs(terms[-2]))<mp.mpf('1e-85'):break
            cdf=mp.fsum(terms);return cdf,hit-cdf
        terms=[]
        for n in range(1,1000):
            lam=(vv*vv+(n*mp.pi)**2)/2
            terms.append(mp.pi*n*mp.sin(n*mp.pi*z)*mp.exp(-vv*z-lam*tt)/lam)
            if n>25 and abs(terms[-1])<mp.mpf('1e-85'):break
        sf=mp.fsum(terms);return hit-sf,sf


def high_precision_check():
    cases=[]
    # Domain corners, near-zero drift, and representation-switch times.
    for v,a,z in [(0.,1.,.5),(50.,8.,.98),(-50.,8.,.98),(50.,8.,.02),(-50.,8.,.02),
                  (50.,.1,.98),(-50.,.1,.02),(1e-8,8.,.2),(-1e-8,.1,.8),(3.,2.,.5),(-3.,2.,.5)]:
        for factor in (1e-5,.01,.119999,.120001,.3,3.):
            T=factor*a*a
            F,S,h,m=ddm.lower_cdf_sf(v,a,z,T,1e-13)
            fm,sm=mp_lower(v,a,z,T)
            cases.append({'v':v,'a':a,'z':z,'time':T,'cdf':float(F),'sf':float(S),
                          'reference_cdf':float(fm),'reference_sf':float(sm),
                          'max_abs_error':max(abs(float(F)-float(fm)),abs(float(S)-float(sm)))})
    table=pd.DataFrame(cases)
    if table.max_abs_error.max()>1e-10:raise AssertionError('90-digit DDM reference check failed.')
    return table


def ddm_domain_check(run,config,design,cuts):
    _,h,u=read_bank(run,'expanded','DDM')
    q,m=ddm.make_ddm_table(h,design,cuts,config['ddm']['absolute_series_tolerance'])
    qr,mr=ddm.make_ddm_table(h,design,cuts,config['ddm']['check_tolerance'])
    maxerr=float(np.max(abs(q-qr)))
    window_error=float(np.max(abs(core.for_target(q,'window')-core.for_target(qr,'window'))))
    reflect_err=0.;complement_err=0.;minimum=float(q.min())
    for start in range(0,len(h),64):
        b=h.iloc[start:start+64];ev=[]
        for rho in b.rho:
            d,p,sc=evidence_arrays(design,float(rho));ev.append((d*p).sum(axis=1))
        v=b.vmax.to_numpy()[:,None]*np.tanh(.5*b.vcoeff.to_numpy()[:,None]*np.array(ev))
        qq,_=ddm.joint_bins(-v,b.a.to_numpy(),1-b.z.to_numpy(),b.t0.to_numpy(),cuts,1e-12)
        correct=np.concatenate([q[start:start+len(b),: ,7:],q[start:start+len(b),:,:7]],axis=-1)
        reflect_err=max(reflect_err,float(abs(qq-correct).max()))
        p1=ddm.p_upper(v,b.a.to_numpy()[:,None],b.z.to_numpy()[:,None])
        p0=ddm.p_upper(-v,b.a.to_numpy()[:,None],1-b.z.to_numpy()[:,None])
        complement_err=max(complement_err,float(abs(p0+p1-1).max()))
    if max(maxerr,window_error,reflect_err,complement_err)>config['ddm']['domain_max_allowed_discrepancy']:
        raise AssertionError('Entire-bank DDM invariance/convergence check failed.')
    core.check_joint(q);core.for_target(q,'window')
    result={'n_candidates':len(h),'n_problems':len(design),'candidate_problem_combinations':len(h)*len(design),
             'max_tolerance_discrepancy':maxerr,'max_conditional_window_tolerance_discrepancy':window_error,'max_label_reflection_error':reflect_err,
             'max_hitting_complement_error':complement_err,'minimum_probability':minimum,
             'max_documented_roundoff_adjustment':max(x['raw_mass_error_max'] for x in m),
             'joint_checks':core.check_joint(q),'all_pass':True}
    # Cache validated analytic tables; no need to evaluate them again in atlas.
    atomic_npz(Path(run)/'atlases/expanded/DDM.npz',joint=q)
    dump_json(Path(run)/'atlases/expanded/DDM.json',result)
    _,hb,_=read_bank(run,'base','DDM')
    if not np.allclose(h.iloc[:len(hb)].to_numpy(float),hb.to_numpy(float),atol=1e-13,rtol=1e-13):raise AssertionError('DDM bank prefix mismatch.')
    atomic_npz(Path(run)/'atlases/base/DDM.npz',joint=q[:len(hb)])
    dump_json(Path(run)/'atlases/base/DDM.json',result)
    return result


def _make_synthetic_run(parent,config,design,folds,cuts):
    # Tiny artificial data run to exercise serialization, fits, predictions,
    # both observation conventions, zero-training participants and packaging.
    from .io import sources
    root=Path(parent)/'synthetic_smoke'
    if root.exists():shutil.rmtree(root)
    root.mkdir();(root/'inputs').mkdir();(root/'validation').mkdir()
    cfg=json.loads(json.dumps(config));cfg['smoke']=True
    cfg['base']={'rfdt_global_continuous':2,'rfdt_nuisance':10,'ddm_nuisance':24}
    cfg['expanded']={'rfdt_global_continuous':3,'rfdt_nuisance':12,'ddm_nuisance':32}
    cfg['simulation']['train_paths']=24
    cfg['simulation']['evaluation_paths']=32;cfg['simulation']['allowed_evaluation_paths']=[32,64,128]
    cfg['simulation']['evaluation_batches']=3
    cfg['eb']['starts']=2;cfg['eb']['maxiter']=200;cfg['bootstrap_reps']=200
    dump_json(root/'config.json',cfg)
    dump_json(root/'PROTOCOL_LOCK.json',{'source_hashes':sources(),'config_sha256':sha(root/'config.json'),
                                      'status':'SYNTHETIC SOFTWARE TEST ONLY'})
    dump_json(root/'validation/VALIDATION_PASSED.json',{'synthetic_only':True})
    design.to_csv(root/'design.csv',index=False);np.save(root/'folds.npy',folds);np.save(root/'cutoffs.npy',cuts)
    for level in ('base','expanded'):
        p=root/'banks'/level;p.mkdir(parents=True)
        for model,(g,h,u) in banks(cfg,level).items():
            h['utility_scale']=[evidence_arrays(design,float(r))[2] for r in h.rho]
            g.to_csv(p/(model+'_globals.csv'),index=False);h.to_csv(p/(model+'_nuisance.csv'),index=False);np.save(p/(model+'_units.npy'),u)
    rng=np.random.default_rng(91410)
    for sample,ns in [('primary',8),('sensitivity',9)]:
        rows=[]
        for i in range(ns):
            for j,pid in enumerate(design.problem_id):
                single=(rng.random()<.8)
                if i==0:single=False
                if i==1:single=(folds[j]==0)  # deliberately zero training when fold 0 held out
                b=int(rng.choice([1,2,3,4,5,6],p=[.1,.2,.2,.2,.2,.1]))
                rt=(cuts[j,b-1]+cuts[j,b])/2 if b<6 else 11.
                rows.append({'participant_id':'synthetic_%03d'%i,'problem_id':pid,
                             'choice_risky':int(rng.random()<.5),'rt':rt,'change_mind':1 if single else 2})
        pd.DataFrame(rows).to_csv(root/'inputs'/(sample+'.csv.gz'),index=False,compression='gzip')
    return root


def validate(run,synthetic=True):
    run,config,design,folds,cuts=read_run(run,require_validation=False)
    dest=run/'validation';dest.mkdir(exist_ok=True)
    start=time.time()
    print('Checking unchanged simulator source and numerical unit tests ...',flush=True)
    dump_json(dest/'engine_source_check.json',check_engine_source())
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],env=env,capture_output=True,text=True)
    (dest/'unit_tests.txt').write_text(r.stdout+'\n'+r.stderr)
    if r.returncode!=0:raise RuntimeError('Unit tests failed. Read '+str(dest/'unit_tests.txt'))
    print('Checking 90-digit reference values ...',flush=True)
    ref=high_precision_check();ref.to_csv(dest/'ddm_90digit_reference.csv',index=False)
    print('Checking entire base/expanded nonlinear DDM candidate domain ...',flush=True)
    domain=ddm_domain_check(run,config,design,cuts);dump_json(dest/'ddm_full_domain.json',domain)
    synthetic_result='not_run'
    if synthetic:
        print('Running tiny end-to-end test on artificial observations only ...',flush=True)
        root=_make_synthetic_run(dest,config,design,folds,cuts)
        from .pipeline import simulate_atlas,fit_stage
        # Same public commands, same saved object and prediction formats.
        simulate_atlas(root,'base','train_a',threads=2)
        fit_stage(root,'base','train_a',jobs=1)
        synthetic_result='completed; artificial observations, not a human comparison'
    result={'all_pass':True,'source_hashes':sources(),'versions':versions(),'seconds':time.time()-start,
             'high_precision_max_error':float(ref.max_abs_error.max()),'ddm_domain':domain,
             'synthetic_pipeline_test':synthetic_result,
             'NOT_CLAIMED':'Full production fitting, simulation precision, or corrected human conclusions.'}
    dump_json(dest/'VALIDATION_PASSED.json',result)
    print('VALIDATION PASSED. Ready for the new atlas, not for a scientific conclusion.',flush=True)
    return result
