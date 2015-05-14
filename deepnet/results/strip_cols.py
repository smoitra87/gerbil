import os,sys
import csv

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Join two csv files")
    parser.add_argument("--file1",  type=str)
    parser.add_argument("--outf", type=str)
    parser.add_argument("--keep_cols", type=str, nargs="+", help="Names of columns to keep")
    args = parser.parse_args()

    with open(args.file1, 'rb') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       f1 = list(reader)

    f1_header, f1_rows = f1[0], f1[1:]

    from collections import OrderedDict
    colmap = OrderedDict((t for t in zip(f1_header,zip(*f1_rows)) if t[0] in args.keep_cols))

    header,rows = zip(*colmap.items())
    print >>sys.stdout, ",".join(header)
    for row in zip(*rows):
        print >>sys.stdout, ",".join(row)

