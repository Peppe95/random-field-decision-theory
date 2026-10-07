#!/usr/bin/env python3
"""Local runner for the bounded Guo dynamic correction. See README_RUN_LOCAL.md."""
from __future__ import annotations
import os
# Prevent each fit worker from starting its own full BLAS thread pool. Numba
# simulator parallelism is set separately by --threads in the atlas command.
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import traceback
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Verify original archives, freeze protocol, prepare banks and observation bookkeeping.')
    p.add_argument('--preprocessing',required=True);p.add_argument('--recovery',required=True)
    p.add_argument('--out',default='rfdt_guo_corrected')
    p=sub.add_parser('validate',help='Numerical tests + full DDM-domain check + artificial-data end-to-end test.')
    p.add_argument('--run',default='rfdt_guo_corrected')
    for cmd in ('atlas','fit'):
        p=sub.add_parser(cmd)
        p.add_argument('--run',default='rfdt_guo_corrected')
        p.add_argument('--level',choices=['base','expanded'],default='base')
        p.add_argument('--batch',choices=['train_a','train_b'],default='train_a')
        if cmd=='atlas':p.add_argument('--threads',type=int,default=8)
        else:p.add_argument('--jobs',type=int,default=4)
    p=sub.add_parser('precision',help='Independent fixed-weight evaluation with explicit simulation-error estimates.')
    p.add_argument('--run',default='rfdt_guo_corrected');p.add_argument('--level',choices=['base','expanded'],default='base')
    p.add_argument('--fit-batch',choices=['train_a','train_b'],default='train_a')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--paths',type=int,default=None)
    p=sub.add_parser('convergence',help='Combine evaluation, refitting-replicate and bank-resolution diagnostics.')
    p.add_argument('--run',default='rfdt_guo_corrected')
    p=sub.add_parser('package',help='Create a compact results ZIP. Atlases and full weights stay local by default.')
    p.add_argument('--run',default='rfdt_guo_corrected');p.add_argument('--out',required=True)
    p.add_argument('--include-weights',action='store_true')
    args=ap.parse_args()
    try:
        if args.command=='prepare':
            from rfdt_correction.io import prepare
            prepare(args.preprocessing,args.recovery,args.out)
        elif args.command=='validate':
            from rfdt_correction.validation import validate
            validate(args.run)
        elif args.command=='atlas':
            from rfdt_correction.pipeline import simulate_atlas
            simulate_atlas(args.run,args.level,args.batch,args.threads)
        elif args.command=='fit':
            from rfdt_correction.pipeline import fit_stage
            fit_stage(args.run,args.level,args.batch,args.jobs)
        elif args.command=='precision':
            from rfdt_correction.pipeline import precision_stage
            precision_stage(args.run,args.level,args.fit_batch,args.threads,args.paths)
        elif args.command=='convergence':
            from rfdt_correction.pipeline import convergence_report
            convergence_report(args.run)
        elif args.command=='package':
            from rfdt_correction.pipeline import package_results
            package_results(args.run,args.out,args.include_weights)
    except Exception as exc:
        print('\nSTOPPED: '+str(exc),flush=True)
        print('Completed chunks and diagnostic files have been retained. Do not use --overwrite or edit the frozen run.',flush=True)
        traceback.print_exc()
        raise SystemExit(1)

if __name__=='__main__':main()
