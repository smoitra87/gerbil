import os, sys

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Create parallel run script")
    parser.add_argument("--start_expid", type=int,  help="Starting experiment id")
    parser.add_argument("--end_expid", type=int,  help="ending experiment id")
    parser.add_argument("--nparallel", type=int, default=1, \
         help="Number of parallel runs")
    parser.add_argument("--run_script_name", type=str,  help="Name of run script",\
            default="run_in_parallel.sh")
    parser.add_argument("--best_model_file", type=str,  help="Best Model file")
    parser.add_argument("--infer_method", type=str,  \
            help="gaussian_exact/mf", default='mf')
    parser.add_argument("--blosum90", action='store_true', help="Calculate blosum90 scores")
    parser.add_argument("--valid_only", action='store_true', help="Calculate blosum90 scores")
    parser.add_argument("--ncols", type=int, nargs='+', help="Num cols")
    parser.add_argument("--multmode", type=str, help="Multicol mode",default='rand')
    args = parser.parse_args()

    job_q = [[] for _ in range(args.nparallel)]
    for idx in range(args.start_expid, args.end_expid+1):
        job_q[idx % args.nparallel].append('exp%d'%(idx))
    
    with open(args.best_model_file) as fin:
        best_models = [line.strip() for line in fin]

    best_dict = {}
    for idx in range(args.start_expid, args.end_expid+1):
        best_dict['exp%d'%(idx)] = next(model for model in best_models if 'exp%d'%(idx) in model)

    cmd = 'python impute.py  --model_file experiments/{0} '
    cmd += '--train_file  experiments/{1}/trainers/{2} '
    cmd += '--mf-steps 2 --outf experiments/likelihoods/{1}/{3} '
    cmd += '--infer-method {}'.format(args.infer_method)

    if args.blosum90:
        cmd += ' --blosum90'

    if args.valid_only:
        cmd += ' --valid_only'

    if args.infer_method == 'multicol':
        cmd += ' --ncols {4}'
        cmd += ' --multmode {5}'

    with open(args.run_script_name,'w') as fout:
        print >>fout, "#!/bin/sh"
        for qidx in range(args.nparallel):
            print >>fout
            jobs = []
            for expid in job_q[qidx]:
                trainer = 'train_CD_rbm1.pbtxt' if 'rbm1' in best_dict[expid] else 'train_CD_joint.pbtxt'
                if args.infer_method == 'multicol':
                    for ncol in args.ncols : 
                        if args.blosum90:
                            pll_f = '{0}_{1}_{2}{3}multbl90.pkl'.format(expid, os.path.split(best_dict[expid])[1], args.multmode, ncol)
                        else:
                            pll_f = '{0}_{1}_{2}{3}multimp.pkl'.format(expid, os.path.split(best_dict[expid])[1], args.multmode, ncol)
                        cmd2 = cmd.format(best_dict[expid], expid, trainer, pll_f, ncol, args.multmode)
                        jobs.append(cmd2)
                else:
                    if args.blosum90:
                        pll_f = '{0}_{1}_blosum90.pkl'.format(expid, os.path.split(best_dict[expid])[1])
                    else:
                        pll_f = '{0}_{1}_pll.pkl'.format(expid, os.path.split(best_dict[expid])[1])

                    cmd2 = cmd.format(best_dict[expid], expid, trainer, pll_f)
                    jobs.append(cmd2)

            print >>fout, "( " + " ; sleep 2 ; ".join(jobs) + " ) & ";
   
    os.system('chmod +x {}'.format(args.run_script_name))
