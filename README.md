# Learning to play the Sudoku and to configure Renault vans

* See Sudoku [Readme.md](Sudoku/README.md) file for learning how to play the Sudoku.
* See Renault [Readme.md](renault/README.md) file for learning car configuration preferences.

## Archive contents

In the top directory, we provide the following main files:

* `BuildPyToulbar2.sh` is a batch script that will pull and compile
  Toulbar2 Python interface on a Linux machine with the proper
  dependencies installed (see below).

* `CFN.py`: is a Python stub to interact with PyToulbar2, the Python API of
  [Toulbar2](https://github.com/toulbar2/toulbar2).

* `PEMRF.py`: is a Python implementation of PE_MRF, the ADMM-based
  Graphical Model regularized maximum log-likelihood parameters and
  structure estimation algorithm for pairwise Graphical Models
  originally described in [this
  paper](https://stanford.edu/~boyd/papers/pdf/pairwise_exp_struct.pdf)
  with L1 (Lasso), L1/L2 (Group Lasso) and L2 (Ridge) regularizations.

* `Sudoku-train-and-test.py`: trains a Sudoku solver and tests its
  performance on a set of 1000 test samples coming either from RRN
  fixed number of hints test sets and SAT-Net test set (in the
  `Sudoku/test-sets` directory, look for the `rrn-test-??.csv` and
  `satnet-test.csv` files). These data-sets are described in the
  original [RRN paper](https://arxiv.org/abs/1711.08028) and the
  original [SAT-Net paper](https://arxiv.org/abs/1905.12149). The
  values of lambda used have been precomputed using the
  `Sudoku-validate.py` below and are available in the `Sudoku/lambdas`
  directory. See Sudoku [Readme.md](Sudoku/README.md) file for more
  details.

* `Sudoku-validate.py`: validation loop to identify a suitable value
  of lambda using a given number of samples. The 1024 first values of
  the RRN validation set are used. See Sudoku
  [Readme.md](Sudoku/README.md) file for more details.

* `renault.py`: training, validation and test script for learning user
   preferences using Renault's data. See Renault
   [Readme.md](renault/README.md) file for more details.

## Additional requirements:

You must compile the PyToulbar2 interface. Look for toulbar2
requirements on https://github.com/toulbar2/toulbar2, install
them. Under Linux, executing the `BuildPyToulbar2.sh` script should
finish the job.

