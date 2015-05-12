from deepnet import util
import os

def SetupDataPbtxt(data_pbtxt_file, data_path):
  data_pbtxt = util.ReadData(data_pbtxt_file)
  for data in data_pbtxt.data:
     fname = os.path.basename(data.file_pattern)
     data.file_pattern = os.path.join(data_path, fname)
  util.WritePbtxt(data_pbtxt_file, data_pbtxt)

if __name__ == '__main__':
    from commands import getstatusoutput
    for data_pbtxt_file in getstatusoutput("find . -name 'data.pbtxt'")[1].split():
        SetupDataPbtxt(data_pbtxt_file, \
                os.path.dirname(os.path.abspath(data_pbtxt_file)))
