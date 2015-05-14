import os, sys

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--fastafile", type=str, help="Fasta File")
    parser.add_argument("--familyname", type=str, help="Name of MSA family")
    args = parser.parse_args()

    if not args.fastafile:
        raise ValueError('Fasta file not defined')

    if not args.familyname:
        raise ValueError('family name not provided')

    if os.path.exists(args.familyname):
        raise IOError('Directory {} already exists'.format(args.familyname))

    cmd = "cp -r template {}".format(args.familyname)
    print cmd ; os.system(cmd) 

    cmd = "cp {} {}".format(args.fastafile, args.familyname)
    print cmd ; os.system(cmd) 

    fasta_base = os.path.basename(args.fastafile)
    cmd = "python convert_fasta_npy.py --train_frac 0.6 --valid_frac 0.2" +\
        " --test_frac 0.2 {} --split".format(os.path.join(args.familyname, fasta_base))
    print cmd ; os.system(cmd) 

    fasta_base2 = os.path.splitext(fasta_base)[0]
    for t in ('train', 'test', 'valid'):
        cmd = "python convert_npy_others.py " +\
        "--inf {}_{}.npy --format msa".format(\
            os.path.join(args.familyname, fasta_base2), t)
    print cmd ; os.system(cmd) 

    cmd = 'python setup_data_pbtxt.py --pfamid {}'.format(args.familyname)
    print cmd ; os.system(cmd) 

    cmd = 'python setup_data_paths.py'
    print cmd ; os.system(cmd) 


