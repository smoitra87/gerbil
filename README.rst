(GERBIL) GEnerative Restricted Boltzmann machines of protein famILies
=====================================================================


Currently the following generative models are supported:
* Restricted Boltzmann Machines
* Deep Boltzmann Machines

**NOTE** : Currently under development. Usage should be considered experimental.

Restricted Boltzmann Machine
----------------------------
These are the instructions for learning a Restricted Boltzmann Machine (RBM)

* Export the following variables::
    
    export GERBILPATH=/path/to/gerbil
    export PYTHONPATH=${GERBILPATH}:${PYTHONPATH}

* First create the datasets. In this example we will consider the PF00240 family ::

    cd deepnet/datasets
    wget http://gremlin.bakerlab.org/fasta_2013/PF00240.fas -P /tmp/
    python create_dataset.py --fastafile /tmp/PF00240.fas --familyname PF00240

* Then, create a new experiment for learning the parameters of the RBM  ::
    
    cd deepnet/experiments
    python generate_experiments.py --start_job_id 1 \
        --model rbm --data_dir PF00240

* Learn the params of the modeld ::
    
    cd exp1 && ./runall.sh    

* To see full range of options available ::
    
    python generate_experiments.py -h

* Calculate imputation error. Job file written out to ``./run_in_paralle.sh``  ::

    cd deepnet/
    python impute_parallel_run.py --start_expid 1 --end_expid 1 --model_prefix rbm
    ./run_in_parallel.sh 

* Create a csv table with results ::

    python create_results_csv.py --expid 1
    less results/imperr_exp1.csv

Deep Boltzmann Machine
----------------------
Coming Soon..

Running on AWS
--------------


Documentation
-------------
Models described in `thesis.pdf`_.

.. _thesis.pdf: https://www.cs.cmu.edu/thesis/thesis.pdf
