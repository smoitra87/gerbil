from deepnet import util
import numpy as np
import os

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--pfamid")
    args = parser.parse_args()

    pfamid = args.pfamid

    data_pbtxt_file = os.path.join(pfamid,'data.pbtxt')
    data_pbtxt = util.ReadData(data_pbtxt_file)
    data_pbtxt.name = pfamid
    data_pbtxt.prefix = os.path.join(os.path.split(data_pbtxt.prefix)[0],pfamid)
    for data in data_pbtxt.data:
       fname = os.path.basename(data.file_pattern)
       for t in ('train','valid','test'):
           if t in data.name:
               X = np.load(os.path.join(pfamid,pfamid+"_"+t+".npy"))
               data.size = X.shape[0]
               data.dimensions[0] = X.shape[1]
               data.file_pattern = os.path.abspath(os.path.join(pfamid, pfamid+"_"+t+".npy"))

    util.WritePbtxt(data_pbtxt_file, data_pbtxt)
