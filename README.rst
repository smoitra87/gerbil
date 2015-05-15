(GERBIL) GEnerative Restricted Boltzmann machines of protein famILies
=====================================================================

.. image:: http://upload.wikimedia.org/wikipedia/commons/6/6b/Gerbil.JPG
   :height: 200px
   :width: 200 px
   :alt: A real life gerbil
   :align: center

GERBIL is a tool for generative modeling of protein sequence families. 
It is built upon the `deepnet`_ package, created by Nitish Srivastava.

The models are described in `thesis.pdf`_.

Currently the following generative models are supported:

* Restricted Boltzmann Machines
* Deep Boltzmann Machines

**NOTE** : Currently under development. Usage should be considered experimental.

See `INSTALL.txt`_ for installation instructions

.. _INSTALL.txt: https://github.com/smoitra87/gerbil/blob/master/INSTALL.txt
.. _deepnet: https://github.com/nitishsrivastava/deepnet
.. _thesis.pdf: https://www.cs.cmu.edu/~subhodee/thesis/thesis.pdf

Alternatively, you can run on AWS using the Gerbil AMI (see below)

Running on AWS
--------------
We provide a precreated AMI with all libraries and paths set. 

* Create a ``g2.2xlarge`` instance and use the Gerbil specific AMI ``ami-d18ab9e1``.

* Pull the latest gerbil version ::

    $ cd gerbil/ && git pull origin master

Getting Started
---------------

* Export the following variables::
    
    $ export GERBILPATH=/path/to/gerbil
    $ export PYTHONPATH=${GERBILPATH}:${PYTHONPATH}

* First create the datasets. In this example we will consider the PF00240 family ::

    $ cd deepnet/datasets
    $ wget http://gremlin.bakerlab.org/fasta_2013/PF00240.fas -P /tmp/
    $ python create_dataset.py --fastafile /tmp/PF00240.fas --familyname PF00240

Restricted Boltzmann Machine (RBM)
----------------------------------

* To see full range of options available ::
    
    $ python generate_experiments.py -h

* Create a new experiment for learning the parameters of the RBM  ::
    
    $ cd deepnet/experiments
    $ python generate_experiments.py --start_job_id 1 \
        --model rbm --data_dir PF00240

* Learn the params of the model ::
    
    $ cd exp1 && ./runall.sh    

* Calculate imputation error. Job file written out to ``./run_in_parallel.sh``  ::

    $ cd deepnet/
    $ python impute_parallel_run.py --start_expid 1 --end_expid 1 --model_prefix rbm
    $ ./run_in_parallel.sh 

* Create a csv table with results ::

    $ python create_results_csv.py --expid 1
    $ less results/imperr_exp1.csv

Deep Boltzmann Machine (DBM)
----------------------------
A DBM needs a RBM to warmstart from. Continuing from ``exp1`` we will create 
another experiment ``exp2`` to train a DBM.


* Create a new experiment ::
    
    $ cd deepnet/experiments
    $ python generate_experiments.py --start_job_id 2 \
        --model dbm --data_dir PF00240

* Copy over the best RBM model in ``exp1`` onto ``exp2`` :: 
    
    $ python choose_best_model.py --impute_dir likelihoods/exp1/ \
        --model_dir exp1/dbm_models/ --output_dir exp2/dbm_models

* Learn the params of the model ::
    
    $ cd exp2 && ./runall.sh    

* Calculate imputation error. Job file written out to ``./run_in_parallel.sh``  ::

    $ cd deepnet/
    $ python impute_parallel_run.py --start_expid 2 --end_expid 2 --model_prefix dbm
    ./run_in_parallel.sh 

* Create a csv table with results ::

    $ python create_results_csv.py --expid 2
    $ less results/imperr_exp2.csv

Extracting params from protocol buffers to .MAT files
-----------------------------------------------------

* Get the best model from an experiment ::
    
    $ cd deepnet/experiments
    $ python choose_best_model.py --impute_dir likelihoods/exp1/ \
        --model_dir exp1/dbm_models/ --print_only
    exp1/dbm_models/rbm1_1430616201
    

* Extract the params from the best model ::
    
    $ cd deepnet/
    $ python write_model_to_mat.py \
        experiments/exp1/dbm_models/rbm1_1430616201 /path/to/my/folder/rbm1_1430616201.mat


