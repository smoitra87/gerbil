#!/bin/bash
#Experiment id for the current job

expid=$(basename $(dirname "$(readlink -f "$0")"))
# Location of deepnet. EDIT this for your setup.
deepnet=${GERBILPATH}/deepnet

trainer=${deepnet}/trainer.py

echo "Training first layer RBM."
python ${trainer} models/rbm.pbtxt \
trainers/train_CD_rbm.pbtxt eval.pbtxt || exit 1
