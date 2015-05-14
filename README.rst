(GERBIL) GEnerative Restricted Boltzmann machines of protein famILies
=====================================================================


Generative Models of Protein Sequence Families:
-----------------------------------------------
* Restricted Boltzmann Machines
* Deep Boltzmann Machines

**NOTE** : Currently under development. Usage should be considered experimental.

Documentation
-------------
Models described in `thesis.pdf`_.

.. _thesis.pdf: https://www.cs.cmu.edu/thesis/thesis.pdf

Restricted Boltzmann Machine
----------------------------
These are the instructions for learning a Restricted Boltzmann Machine (RBM)

* Export the following variables::
    
    export GERBILPATH=/path/to/gerbil
    export PYTHONPATH=${GERBILPATH}/deepnet:${PYTHONPATH}

* First create the datasets. In this example we will consider the PF00240 family ::

    cd deepnet/datasets
    wget http://gremlin.bakerlab.org/fasta_2013/PF00240.fas -P tmp/
    python create_dataset.py --fastafile /tmp/PF00240.fas --familyname PF00240

* Then, create a new experiment for learning the parameters of the RBM  ::
    
    cd deepnet/experiments
    python generate_experiments.py --

* To see full range of options available ::
    
    python generate_experiments.py -h

* Learn the params of the model

* Calculate imputation error on test set

* 


Deep Boltzmann Machine
----------------------
Coming Soon..

