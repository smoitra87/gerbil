import os,sys

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("Creates csv of results")
    parser.add_argument("--expid", type=int, nargs=1, help="Experiment id")
    args = parser.parse_args()

    expid = args.expid[0]

    cmd = "cp -r experiments/exp{0} results/".format(expid)
    print cmd ; os.system(cmd)

    cmd = "cp experiments/likelihoods/exp{}/*.pkl results/likelihoods/".format(expid)
    print cmd ; os.system(cmd)

    cmd = "cd results && python collect_likelihoods.py --globstr 'likelihoods/exp{}*pll.pkl'".format(expid) +\
        " --outf likelihoods_exp{}.csv --select_best valid-imperr".format(expid)
    print cmd ; os.system(cmd)

    cmd = "cd results && python collect_results2.py --mode csv --outf models_scan.csv"
    print cmd ; os.system(cmd)
    
    cmd = "cd results && python join_on_cols.py --file1 models_scan.csv " +\
        "--file2 likelihoods_exp{0}.csv  --cols expid model_name --outf imperr_exp{0}.csv".format(expid)
    print cmd ; os.system(cmd)
