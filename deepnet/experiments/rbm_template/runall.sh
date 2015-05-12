#!/bin/bash
#Experiment id for the current job

expid=$(basename $(dirname "$(readlink -f "$0")"))
# Location of deepnet. EDIT this for your setup.
if [ `hostname` == 'langmead.pc.cs.cmu.edu' ] ; then
    deepnet=/storage/data1/dbm/deepnet/deepnet
else
    deepnet=/home/ubuntu/deepnet/deepnet
fi

trainer=${deepnet}/trainer.py

echo "Training first layer RBM."
python ${trainer} models/rbm1.pbtxt \
trainers/train_CD_rbm1.pbtxt eval.pbtxt || exit 1
