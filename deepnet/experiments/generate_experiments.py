import os, sys
import shutil
from itertools import product
import copy 
import commands

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Create Boltzmann Machine job")
    parser.add_argument("--start_job_id", type=int, \
        help="Starting Job Id e.g. 1000")
    parser.add_argument("--model", type=str, \
        default='rbm', help="Model type rbm/dbm")
    parser.add_argument("--data_dir", type=str, \
        help="Dataset directory e.g. PF00240")
    parser.add_argument("--base_epsilon", type=float, nargs = '*', \
        default=[0.1], help="Learning rate e.g. 0.1")
    parser.add_argument("--l2_decay", type=float, nargs = '*', \
        default=[0.01], help="l2 decay weight e.g. 0.01")
    parser.add_argument("--hidden1_width", type=int, nargs = '*', \
        default=[1000], help="Width of Hidden Layer 1 e.g. 1000")
    parser.add_argument("--hidden2_width", type=int, nargs = '*', \
        default=[1000], help="Width of Hidden Layer 2 e.g. 1000")
    parser.add_argument("--steps", type=int, nargs = '*', \
        default=[100000], help="Number of epochs to run e.g. 100000")
    parser.add_argument("--batchsize", type=int, nargs = '*', \
        default=[500], help="Number of sequences in batch e.g. 500")
    args = parser.parse_args()   

    if not args.start_job_id:
        raise ValueError('Starting Job Id requiresd, quitting...')

    if not args.data_dir:
        raise ValueError('data_dir {} not found'.format(args.data_dir))

    params = copy.deepcopy(args.__dict__)
    for k in params.keys():
        params["--"+k] = params[k]
        del params[k]

    
    params['--data_dir'] = ['datasets/{}'.format(args.data_dir)]
    cmd = "grep dimensions ../datasets/{}/data.pbtxt ".format(args.data_dir) +\
                "| tr -d ' '| cut -d: -f2 | head -n1"
    dimensions = commands.getstatusoutput(cmd)[1].strip()
    params["--input_width"] = [dimensions]

    def append_to_list(k, param_list, params):
        param_list.append([k+" "+str(v) for v in params[k]])

    # Order params such that ones towards the end are sampled more
    param_list = []
    param_list.append(['--initial_momentum 0.5 --final_momentum 0.9'])
    param_list.append(['--model_dir experiments/{0}/dbm_models'+\
            ' --rep_dir experiments/{0}/dbm_reps'])
    param_list.append(["--model {}".format(args.model)])
    append_to_list("--l2_decay", param_list, params)
    append_to_list("--base_epsilon", param_list, params)
    append_to_list("--hidden1_width", param_list, params)
    append_to_list("--steps", param_list, params)
    append_to_list("--batchsize", param_list, params)
    append_to_list("--input_width", param_list, params)
    append_to_list("--data_dir", param_list, params)

    if args.model == 'dbm':
        append_to_list("--hidden2_width", param_list, params)

    tuplist = list(product(*param_list))

    for idx, tup in list(enumerate(tuplist)):
        expid ='exp'+str(idx + args.start_job_id) 
        tup_args = " ".join(tup).format(expid)
        
        if os.path.exists(expid):
            sys.stderr.write("{} already exists".format(expid))
            sys.exit(1)

        model_name = args.model

        shutil.copytree('{}_template'.format(model_name), expid)
        cmd ='cd {0} && python setup_protos.py {1}'.format(expid, tup_args) 
        print(cmd)

        os.system(cmd) 
        with open(os.path.join(expid, 'README'),'w') as fout:
            fout.write(tup_args)
