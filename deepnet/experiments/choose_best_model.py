import os, sys
import glob
import pickle

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--impute_dir", type=str, help="Imputation error dir")
    parser.add_argument("--model_dir", type=str, help="Location models")
    parser.add_argument("--output_dir", type=str, help="Location Output")
    parser.add_argument("--globstr", type=str, help="Location Output")
    parser.add_argument("--print_only", action='store_true', help="Don't copy best model")
    args = parser.parse_args()

    if args.globstr:
        model_pll_files = glob.glob(args.globstr)
    elif args.impute_dir:
        model_pll_files = glob.glob(os.path.join(args.impute_dir,'*.pkl'))
    else:
        raise ValueError('Need input directory location')

    min_imperr = 1
    min_f = None

    for f in model_pll_files:
        with open(f) as fin:
            pll = pickle.load(fin)
        
        imperr_f =  pll['imperr']['valid'].mean() 
        
        if imperr_f < min_imperr:
            min_imperr, min_f = imperr_f, f
        
    # HACKY
    model_name = "_".join(os.path.basename(min_f).split('_')[1:3])
    model_prefix = "_".join(os.path.basename(min_f).split('_')[1:2])
    
    from_f = os.path.join(args.model_dir,model_name)

    if args.print_only:
        print from_f
    else:
        to_f = os.path.join(args.output_dir, model_prefix+'_imperr_BEST')
        print("cp {0} {1}".format(from_f, to_f))
        os.system("cp {0} {1}".format(from_f, to_f))
    




