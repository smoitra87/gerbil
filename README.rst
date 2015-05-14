(GERBIL) GEnerative Restricted Boltzmann machines of protein famILies
=====================================================================


Currently the following generative models are supported:

* Restricted Boltzmann Machines
* Deep Boltzmann Machines

**NOTE** : Currently under development. Usage should be considered experimental.

See `INSTALL.txt`_ for installation instructions


.. _INSTALL.txt: https://github.com/smoitra87/gerbil/blob/master/INSTALL.txt


Getting Started
---------------

* Export the following variables::
    
    export GERBILPATH=/path/to/gerbil
    export PYTHONPATH=${GERBILPATH}:${PYTHONPATH}

* First create the datasets. In this example we will consider the PF00240 family ::

    cd deepnet/datasets
    wget http://gremlin.bakerlab.org/fasta_2013/PF00240.fas -P /tmp/
    python create_dataset.py --fastafile /tmp/PF00240.fas --familyname PF00240

Restricted Boltzmann Machine (RBM)
----------------------------------

* To see full range of options available ::
    
    python generate_experiments.py -h

* Create a new experiment for learning the parameters of the RBM  ::
    
    cd deepnet/experiments
    python generate_experiments.py --start_job_id 1 \
        --model rbm --data_dir PF00240

* Learn the params of the model ::
    
    cd exp1 && ./runall.sh    

* Calculate imputation error. Job file written out to ``./run_in_parallel.sh``  ::

    cd deepnet/
    python impute_parallel_run.py --start_expid 1 --end_expid 1 --model_prefix rbm
    ./run_in_parallel.sh 

* Create a csv table with results ::

    python create_results_csv.py --expid 1
    less results/imperr_exp1.csv

Deep Boltzmann Machine (DBM)
----------------------------
A DBM needs a RBM to warmstart from. Continuing from ``exp1`` we will create 
another experiment ``exp2`` to train a DBM.


* Create a new experiment ::
    
    cd deepnet/experiments
    python generate_experiments.py --start_job_id 2 \
        --model dbm --data_dir PF00240

* Copy over the best RBM model in ``exp1`` onto ``exp2`` :: 
    
    python choose_best_model.py --impute_dir likelihoods/exp1/ \
        --model_dir exp1/dbm_models/ --output_dir exp2/dbm_models

* Learn the params of the model ::
    
    cd exp2 && ./runall.sh    

* Calculate imputation error. Job file written out to ``./run_in_parallel.sh``  ::

    cd deepnet/
    python impute_parallel_run.py --start_expid 2 --end_expid 2 --model_prefix rbm
    ./run_in_parallel.sh 

* Create a csv table with results ::

    python create_results_csv.py --expid 2
    less results/imperr_exp2.csv

Running on AWS
--------------
We provide a precreated AMI with all libraries and paths set. 

* Create a ``g2.2xlarge`` instance and use the Gerbil specific AMI ``todo``.

* Pull the latest gerbil version ::

    cd gerbil/ && git pull origin master

Documentation
-------------
Models described in `thesis.pdf`_.

.. _thesis.pdf: https://www.cs.cmu.edu/thesis/thesis.pdf
