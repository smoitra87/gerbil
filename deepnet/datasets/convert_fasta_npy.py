import argparse
import os, sys
import random
import numpy as np
from Bio import SeqIO

random.seed(42)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Converts Fasta files into ascii")
    parser.add_argument("inf", type=str, help="input fasta file")
    parser.add_argument("--noshuffle", action="store_true", help="Don't shuffle the data")
    parser.add_argument("--split", action='store_true', help="Split into train, valid and test")
    parser.add_argument("--train_frac", type=float, default=0.7, help="Training data fraction")
    parser.add_argument("--valid_frac", type=float, default=0.15, help="Validation data fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Testing data fraction")
    parser.add_argument("--binarize", action='store_true', help="binarize output")
    args = parser.parse_args()

    AALIST= [aa for aa in "AVLIPFWMGSTCYNQDEKRH-"]
    aa_dict = dict((aa,idx) for (idx,aa) in enumerate(AALIST))
    aa_dict["X"] = len(AALIST) - 1

    base, _ = os.path.splitext(args.inf)

    aa_lines = []

    for seqr in SeqIO.parse(args.inf,"fasta"):
        line = str(seqr.seq)
        aa_line = [aa_dict[aa]*2/len(AALIST) if args.binarize else aa_dict[aa] for aa in line]
        aa_lines.append(aa_line)

    if not args.noshuffle : 
        random.shuffle(aa_lines)

    if args.split:
        train_frac = args.train_frac
        valid_frac = args.valid_frac
        nseq = len(aa_lines)
        train = aa_lines[:int(nseq * train_frac)]
        valid = aa_lines[int(nseq * train_frac):int(nseq * train_frac) + int(nseq * valid_frac)]
        test = aa_lines[int(nseq * train_frac) + int(nseq * valid_frac):]

        for f in ['train','valid','test']:
            seqs = eval(f)
            seqs = np.asarray(seqs, dtype="int").squeeze()
            fout_str = base+ "_" +  f + '.npy'
            np.save(fout_str, seqs)
    else :
        with open(base + '.npy','w') as fout :        
            for aa_line in aa_lines:
                print >>fout, ",".join(str(aa) for aa in aa_line)

	X = util.convert_mat_from_msa(args.inputf)
	if args.binarize:
		X = np.array(X >= 10, dtype=int)
	np.save(args.outputf, X)
