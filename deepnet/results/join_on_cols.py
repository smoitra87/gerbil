import os,sys
import csv

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = "Join two csv files")
    parser.add_argument("--file1",  type=str)
    parser.add_argument("--file2", type=str)
    parser.add_argument("--cols", nargs='+', type=str)
    parser.add_argument("--outf", type=str)
    args = parser.parse_args()

    with open(args.file1, 'rb') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       f1 = list(reader)

    with open(args.file2, 'rb') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       f2 = list(reader)

    f1_header, f1_rows = f1[0], f1[1:]
    f2_header, f2_rows = f2[0], f2[1:]

    
    f1_cols, f2_cols = [],[]
    for col in args.cols:
        f1_cols.append(next(col_idx for col_idx, col2 in enumerate(f1_header) if col2 == col ))
        f2_cols.append(next(col_idx for col_idx, col2 in enumerate(f2_header) if col2 == col ))

    f1_colmap,f2_colmap = {},{}
    for row in f2_rows : 
        f2_colmap[tuple(row[col] for col in f2_cols)] = [e for (idx,e) in enumerate(row) if \
                idx not in f2_cols ]

    for row in f1_rows : 
        f1_colmap[tuple(row[col] for col in f1_cols)] = [e for (idx,e) in enumerate(row) if \
                idx not in f1_cols ]
    
    common_cols = set(f1_colmap.keys()).intersection(f2_colmap.keys())

    joined_header = args.cols + [col for col in f1_header if col not in args.cols] + \
            [col for col in f2_header if col not in args.cols]
    joined_rows = []
    for coltup in common_cols:
      joined_rows.append(list(coltup) + f1_colmap[coltup] + f2_colmap[coltup] ) 

    with open(args.outf, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(joined_header)
        for record in joined_rows:
            writer.writerow(record)
