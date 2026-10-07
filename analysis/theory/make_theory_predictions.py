#!/usr/bin/env python3
"""Reproduce RFDT analytical illustrations and independent numerical checks.
No human observations or fitted parameters are read. All examples are illustrative.
Run: python make_theory_predictions.py
Requires numpy, scipy, pandas and matplotlib. Each plot has its own single axes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'theory_checks'; OUT.mkdir(exist_ok=True)

def qa(m,d,p,beta,kappa):
    z=kappa*m+beta*d
    Q=float(p@np.tanh(z))
    A=float(np.tanh((logsumexp(np.log(p)+z)-logsumexp(np.log(p)-z))/2))
    return Q,A

def equilibrium(d,p,alpha,beta,kappa):
    def g(m):
        Q,A=qa(m,d,p,beta,kappa)
        return m-(1-alpha)*Q-alpha*A
    return brentq(g,-1,1,xtol=5e-15)

def meanfield(d,p,alpha,beta,kappa,r,tmax=30.0):
    K=len(d); sig=np.array([-1.,1.])[:,None]
    rho=np.exp(np.log(p)[None,:]+beta*sig*d[None,:])
    rho/=rho.sum(axis=1,keepdims=True)
    x0=np.stack([np.tile((1-alpha)*p/2,(2,1)), np.tile(alpha*p/2,(2,1))])
    def rhs(t,state):
        x=state[:-1].reshape(2,2,K); m=float((x*sig[None,:,:]).sum())
        pi=expit(2*sig*(kappa*m+beta*d[None,:]))
        dx=pi[None,:,:]*x.sum(axis=1,keepdims=True)-x
        dx[1]+=r*(rho*x[1].sum(axis=1,keepdims=True)-x[1])
        return np.r_[dx.ravel(),m]
    sol=solve_ivp(rhs,(0,tmax),np.r_[x0.ravel(),0.],method='DOP853',rtol=3e-11,atol=2e-13,dense_output=True,max_step=.10)
    if not sol.success: raise RuntimeError(sol.message)
    def values(t):
        st=sol.sol(t); x=st[:-1].reshape((2,2,K)+np.shape(t))
        m=(x*sig[None,:,:,None]).sum(axis=(0,1,2)) if np.ndim(t) else float((x*sig[None,:,:]).sum())
        return m,st[-1]
    return sol,values

def save(fig,name):
    fig.savefig(ROOT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(ROOT/(name+'.png'),bbox_inches='tight',dpi=240)
    plt.close(fig)

# Temporal illustration: A=1 for sure; B=4 with probability .2, otherwise 0.
d=np.array([-1.,3.]); p=np.array([.8,.2]); alpha=.85; beta=1.; kappa=.5; r=1.
q,a=qa(0,d,p,beta,kappa); v=(1-alpha)*q+alpha*a; ms=equilibrium(d,p,alpha,beta,kappa)
sol,ev=meanfield(d,p,alpha,beta,kappa,r,100.)
t=np.linspace(0,32,3201); m,z=ev(t)
opt=minimize_scalar(lambda s: float(sol.sol(s)[-1]),bounds=(.01,34),method='bounded',options={'xatol':1e-12})
tmin=float(opt.x); zmin=float(opt.fun)
asml=.35; alrg=1.40
assert zmin<-asml and zmin>-alrg and z[-1]>alrg
Ts=brentq(lambda s:float(sol.sol(s)[-1])+asml,.00001,tmin)
Tl=brentq(lambda s:float(sol.sol(s)[-1])-alrg,tmin,34)
alpha_crit=-q/(a-q)
# One axes, exact numerical integral, no hand-drawn path.
fig,ax=plt.subplots(figsize=(7.1,3.65))
ax.plot(t,z,lw=2,label=r'Unabsorbed mean-field support $Z(s)$')
ax.axhline(0,ls=':',lw=.75)
for bound,ls in [(asml,':'),(alrg,'--')]:
    ax.axhline(bound,ls=ls,lw=.85)
    ax.axhline(-bound,ls=ls,lw=.85)
ax.plot([Ts],[-asml],marker='o',ms=7,ls='None')
ax.plot([Tl],[alrg],marker='s',ms=7,ls='None')
ax.annotate('Lower threshold: choose A',xy=(Ts,-asml),xytext=(4.,-1.73),arrowprops={'arrowstyle':'->','lw':.8},fontsize=10)
ax.annotate('Higher threshold: choose B',xy=(Tl,alrg),xytext=(14.5,2.32),arrowprops={'arrowstyle':'->','lw':.8},fontsize=10)
ax.set(xlim=(0,32),ylim=(-1.95,3.0),xlabel=r'Internal time $s$',ylabel=r'Accumulated support $Z(s)$')
ax.text(31.3,alrg+.03,r'$+a_{\rm high}$',ha='right',va='bottom',fontsize=10)
ax.text(31.3,-alrg-.03,r'$-a_{\rm high}$',ha='right',va='top',fontsize=10)
ax.text(31.3,asml+.02,r'$+a_{\rm low}$',ha='right',va='bottom',fontsize=9)
ax.text(31.3,-asml-.02,r'$-a_{\rm low}$',ha='right',va='top',fontsize=9)
ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=10)
fig.tight_layout();save(fig,'Figure_theory_temporal')
pd.DataFrame({'internal_time':t,'m':m,'Z':z}).to_csv(OUT/'temporal_example.csv',index=False)
# Shared-common-event dilution. Common utility is zero; scale held fixed.
a2=.4; B=float(p@np.sinh(d)); C=float(p@np.cosh(d));
qc=(-a2*B/((1-a2)*q)-1)/(C-1)
qs=np.linspace(.002,1,400); rows=[]
fig,ax=plt.subplots(figsize=(6.6,3.5))
for k,ls in [(0.,'-'),(.5,'--'),(.85,':')]:
    pp=[]
    for eta in qs:
        dm=np.r_[d,0.]; pm=np.r_[eta*p,1-eta]
        # Eliminate a zero-probability state at eta=1.
        good=pm>0; mm=equilibrium(dm[good],pm[good],a2,1.,k)
        prob=expit(2*mm); pp.append(prob); rows.append((eta,k,mm,prob))
    ax.plot(qs,pp,ls=ls,lw=1.8,label=fr'$\kappa={k:g}$')
ax.axhline(.5,ls=':',lw=.7);ax.axvline(qc,ls='--',lw=.8)
ax.annotate(fr'$q_*={qc:.3f}$',xy=(qc,.5),xytext=(.48,.56),arrowprops={'arrowstyle':'->','lw':.8},fontsize=10)
ax.set(xlim=(0,1),ylim=(.32,.61),xlabel=r'Probability $q$ that the original choice is implemented',ylabel=r'Equilibrium readout $P(B)$')
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=10,loc='lower left');fig.tight_layout();save(fig,'Figure_theory_common_event')
pd.DataFrame(rows,columns=['q','kappa','m_star','p_B']).to_csv(OUT/'common_event_example.csv',index=False)
# Same equilibrium for all strictly positive r; different deterministic trajectories.
fig,ax=plt.subplots(figsize=(6.6,3.5)); rates=[.25,1.,4.]
for rate,ls in zip(rates,['-','--',':']):
    sr,er=meanfield(d,p,alpha,beta,kappa,rate,120.)
    tr=np.linspace(0,60,1501);mr,zr=er(tr)
    ax.plot(tr,mr,ls=ls,lw=1.8,label=fr'$r={rate:g}$')
ax.axhline(ms,ls='-.',lw=.8,label=r'Common equilibrium $m^*$')
ax.axhline(0,ls=':',lw=.7);ax.set(xlabel=r'Internal time $s$',ylabel=r'Mean-field state $m(s)$',xlim=(0,60))
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=10);fig.tight_layout();save(fig,'Figure_theory_refresh')
# Checks independent of human fits.
rng=np.random.default_rng(294015)
checks={'source':'Own deductions from fixed RFDT equations; no empirical fit',
        'temporal_example':{'D':d.tolist(),'p':p.tolist(),'alpha':alpha,'beta':beta,'kappa':kappa,'r':r,'Q0':q,'A0':a,'V':v,'m_star':ms,'alpha_critical':alpha_crit,'minimum_Z':zmin,'time_minimum':tmin,'low_bound':asml,'high_bound':alrg,'first_hit_low':Ts,'first_hit_high':Tl},
        'common_event':{'alpha':a2,'beta':1.,'S':B,'C':C,'q_critical':qc}}
# Common-event formula equals direct calculation; stationary sign independent kappa.
err=0.; sign_ok=True
for _ in range(160):
    dd=rng.uniform(-3,3,5);pp=rng.dirichlet(np.ones(5));aa=rng.uniform(.05,.95);bb=rng.uniform(.1,1.5);eta=rng.uniform(.01,.99)
    QQ,AA=qa(0,dd,pp,bb,.4);SS=float(pp@np.sinh(bb*dd));CC=float(pp@np.cosh(bb*dd))
    DD=np.r_[dd,0];PP=np.r_[eta*pp,1-eta]; qd,ad=qa(0,DD,PP,bb,.4)
    vf=(1-aa)*eta*QQ+aa*eta*SS/(1+eta*(CC-1))
    err=max(err,abs(vf-((1-aa)*qd+aa*ad)))
    for kk in [0,.2,.8,.99]:
        mm=equilibrium(DD,PP,aa,bb,kk)
        sign_ok &= np.sign(mm)==np.sign(vf)
checks['common_event_max_formula_error']=err; checks['equilibrium_sign_checks_pass']=bool(sign_ok)
# Mixed cycle in old orientation: first option is favored when value >0.
lot=[(np.array([0.,1.]),np.array([.8,.2])),(np.array([-1.,3.]),np.array([.8,.2])),(np.array([-3.,2.]),np.array([.3,.7]))]
cyc=[]
for ia,ib in [(0,1),(1,2),(2,0)]:
    xa,pa=lot[ia];xb,pb=lot[ib];dd=(xa[:,None]-xb[None,:]).ravel();pp=(pa[:,None]*pb[None,:]).ravel()
    QQ,AA=qa(0,dd,pp,1,0);cyc.append(.6*QQ+.4*AA)
checks['cycle_V_first_minus_second']=cyc
# CE expansion with fixed bounded utility; finite support assumption explicit.
U=np.array([0.,1.,4.]);pp=np.array([.65,.25,.10]);mu=float(pp@U);mu3=float(pp@(U-mu)**3)
ce_rows=[]
for aa in [.2,.4,.8]:
 for bb in [.02,.04,.08]:
    ce=brentq(lambda c:((1-aa)*qa(0,U-c,pp,bb,0)[0]+aa*qa(0,U-c,pp,bb,0)[1]),0,4,xtol=1e-14)
    approx=mu+(3*aa-2)*mu3*bb*bb/6
    ce_rows.append({'alpha':aa,'beta':bb,'exact_CE':ce,'cubic_expansion_CE':approx,'remainder_over_beta4':(ce-approx)/bb**4})
pd.DataFrame(ce_rows).to_csv(OUT/'certainty_equivalent_checks.csv',index=False)
# Direct fourfold example, same alpha beta kappa lambda for all comparisons.
ff=[]
for sign in [1,-1]:
 for prob in [.1,.9]:
    dd=sign*(np.array([0.,10.])-10*prob);pp=np.array([1-prob,prob]); mm=equilibrium(dd,pp,.8,.2,.5)
    ff.append({'payoff_sign':sign,'nonzero_probability':prob,'risky_m':mm,'risky_probability':float(expit(2*mm))})
checks['illustrative_fourfold']=ff
# ODE invariants and derivative identity.
st=sol.sol(t);x=st[:-1].reshape(2,2,2,-1)
checks['max_occupancy_mass_error']=float(np.max(np.abs(x.sum(axis=(0,1,2))-1)))
checks['max_persistent_marginal_error']=float(np.max(np.abs(x[0].sum(axis=0)-(1-alpha)*p[:,None])))
checks['minimum_occupancy']=float(x.min())
checks['terminal_equilibrium_error_at_s100']=float(abs(ev(np.array([100.]))[0][0]-ms))

# Verify the occupancy free-energy derivative by two independent calculations.
entropy_errors=[]; dissipation=[]; convexity_gaps=[]
for rep in range(40):
    K=4; dd=rng.uniform(-2,2,K); pp=rng.dirichlet(np.ones(K)); aa=rng.uniform(.1,.9)
    bb=rng.uniform(.2,1.8); kk=rng.uniform(0,.98); rr=rng.uniform(.1,3)
    sig=np.array([-1.,1.])[:,None]; w=np.array([1-aa,aa])
    xp=np.zeros((2,K))
    frac=rng.uniform(.05,.95,K)
    xp[0]=(1-aa)*pp*frac; xp[1]=(1-aa)*pp*(1-frac)
    xa=aa*rng.dirichlet(np.ones(2*K)).reshape(2,K)
    xx=np.stack([xp,xa]); mm=float((xx*sig[None,:,:]).sum())
    pi=expit(2*sig*(kk*mm+bb*dd))
    rho=np.exp(np.log(pp)[None,:]+bb*sig*dd);rho/=rho.sum(axis=1,keepdims=True)
    dx=pi[None,:,:]*xx.sum(axis=1,keepdims=True)-xx
    dx[1]+=rr*(rho*xx[1].sum(axis=1,keepdims=True)-xx[1])
    gradient=np.log(xx/(w[:,None,None]*pp[None,None,:]/2))+1-bb*sig[None,:,:]*dd-kk*mm*sig[None,:,:]
    deriv=float(np.sum(gradient*dx)); edge_deriv=0.
    for g in [0,1]:
        for j in range(K):
            ab=xx[g,0,j]*pi[1,j];ba=xx[g,1,j]*pi[0,j]
            edge_deriv-=(ab-ba)*np.log(ab/ba)
    for si in [0,1]:
        for j in range(K):
            for k in range(j+1,K):
                ab=rr*xx[1,si,j]*rho[si,k];ba=rr*xx[1,si,k]*rho[si,j]
                edge_deriv-=(ab-ba)*np.log(ab/ba)
    entropy_errors.append(abs(deriv-edge_deriv));dissipation.append(deriv)
    vec=rng.normal(size=xx.shape); denom=float(np.sum(vec*vec/xx))
    quadratic=denom-kk*float(np.sum(vec*sig[None,:,:]))**2
    convexity_gaps.append(quadratic-(1-kk)*denom)
checks['max_free_energy_flux_identity_error']=max(entropy_errors)
checks['max_free_energy_derivative_random_states']=max(dissipation)
checks['minimum_hessian_bound_slack']=min(convexity_gaps)
checks['all_numeric_checks_pass']=bool(err<1e-12 and sign_ok and all(vv>0 for vv in cyc) and checks['max_occupancy_mass_error']<1e-9 and max(entropy_errors)<1e-10 and max(dissipation)<0 and min(convexity_gaps)>-1e-10)
assert checks['all_numeric_checks_pass']
(OUT/'analytical_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
