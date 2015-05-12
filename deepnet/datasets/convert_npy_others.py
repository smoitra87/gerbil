import argparse
import os, sys
import random
import numpy as np

random.seed(42)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Converts Fasta files into ascii")
    parser.add_argument("--inf", type=str, help="input npy file")
    parser.add_argument("--format", type=str, help="(fasta, msa, ascii)")

    args = parser.parse_args()

    AALIST= [aa for aa in "AVLIPFWMGSTCYNQDEKRH-"]
    aa_dict = dict((aa,idx) for (idx,aa) in enumerate(AALIST))
    aa_reverse_dict = dict((v,k) for k,v in aa_dict.items())

    base, _ = os.path.splitext(args.inf)

    if args.format != 'msa' : 
        print("Only msa output format supported")
        sys.exit(1)
    aln = np.load(args.inf)
    
    with open(base +  '.msa','w') as fout:
        for seq in aln:
            seq = "".join([aa_reverse_dict[aa]  for aa in seq])
            print >>fout, seq
