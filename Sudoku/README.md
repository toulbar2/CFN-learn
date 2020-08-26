# Learning to play the Sudoku

This is a quick description of how to learn to play the Sudoku using
the combination of a Convex optimization learning algorithm and a
discrete Branch and Bound based optimizer (ToulBar2) as described in our
[CP'2020
paper](https://miat.inrae.fr/schiex/Export/Pushing_Data_in_your_CP_model.pdf). A
video of the paper presentation is also available
[here](https://www.youtube.com/watch?v=IpUr6KIEjMs).

## Sudoku directory contents

In this git repository, we provide the following files:

* `train-sufficient-statistics` directory: files `A_1000` to
  `A_180000` are pickled matrices containing the sufficient
  statistics for PEMRF assuming that exact solutions are
  available. These can be computed in linear time from the
  corresponding learning set (computes from the RRN learning set,
  using samples as ordered in the original RRN learning set). The
  files `B_1000` to `B_180000` are pickled matrices containing the
  sufficient statistics for PEMRF assuming that solutions are
  available as images. These can be computed in linear time from the
  corresponding learning set (RRN learning set, using samples as
  ordered in the original RRN learning set) and the LeNet output
  scores for every used MNIST digits (the scores have been computed on the
  original MNIST dataset using the PyTorch based LeNet
  implementation available [here](https://github.com/pytorch/examples/blob/master/mnist/main.py).

* `LeNet-outputs/MNIST_test_marginal`: a python pickled array of
  LeNet scores for all MNIST digits. This is used to add unary cost
  functions for validation (hints as images, mode 1 and 2) and to
  evaluate the likelihood that the predicted image is correct
  (solutions as images, mode 2). Computed using the same PyTorch based
  LeNet implementation as above.

* `validation-set/rrn-validation.csv`: the validation set of the
  Recurrent Relation net paper. We only use the first 1024 samples for
  validation.

* `test-sets` directory: `satnet-test.csv` contains the SAT-Net test
  set (1000 samples). Used for comparison with SAT-Net. The
  `rrn-test-??.csv` files contain the subset of the RRN test set with
  exactly "??"  hints (?? varying from 17 to 34, with 1000 samples
  each).

## Training and Testing

To train a Sudoku solver and test it on a given `<number>` of training
samples on a given `<test set>` just execute (top directory):

`python3 Sudoku-train-and-test.py <mode> <number> <test set>`

The `<mode>` describes the assumptions that are made on the nature of
the training and validation set:

* `mode 0`: assumes that hints and solutions in the training and
  validation sets can be directly used.

* `mode 1`: assumes that hints are available as images and needs to be
  LeNet decoded as in the SAT-net paper.

* mode 2: assumes that hints and solutions are available as images and
  are decoded using LeNet. This makes training and validation far more
  challenging.

This script uses precomputed values of lambda that have been obtained
using the validation loop (see below) and are stored in the `lambdas`
directory. The test produces a `test-<mode>-<number of samples>-<test
set>` file containing the following statistics:

* `training_size`: the number of training samples used for training

* `correct_grid_ratio`: the fraction of grids that have been perfectly predicted

* `correct_cell_ratio`: the ratio of cells/digits that have been correctly predicted

* `ADMM_time`: the cpu time used by ADMM for CFN learning 

* `total_toulbar2_time`: the total cpu time used by toulbar2 for
  testing all samples (needs to be divied by the number of samples: 1000)

* `funcnumber`: the number of learned cost functions

* `exact`: True if the number of cost functions learned is 810 and if
  they all contain a soft difference cost function. Scopes need to be
  further checked.

Example: `python3 Sudoku-train-and-test.py 0 13000 satnet-test`

will train the Sudoku solver on the 13000 first RRN training samples
assuming that hints and solutions can be directly used (mode 0), using
a precompted value of lambda estimated using the same assumptions. It
will then test its performance on the `satnet-test` set.

Example: `python3 Sudoku-train-and-test.py 0 13000 rrn-test-17`

will train again and test the same Sudoku solver on the hardest (17
hints) test set of the RRN paper.

## Validation

Validation is achieved using the `Sudoku-validate.py` script. The
script takes as input a mode (0,1, 2) and a training sample size (that
can vary from 1,000 to 180,000 by step of 1000 up to 20,000 and then
by steps of 10,000). Optionally, it can take a maximum number of
backtracks (default 20000 here). It produces a set of measures for
various tested values of lambda (and validation sample sizes) stored
in a file named `probe-<mode>-<size>.csv` where `<mode>` and `<size>`
are the mode and training sample size given. Each line in this file
contains the following statistics:

* `lambda`: the value of the regularization parameter lambda tried

* `num_val`: the number of validation samples this value has been tested on

* `badg`: the number of bad predicted grids

* `badc`: the ratio of bad predicted digits/cells

* `ADMMtime`: the cpu time used by ADMM for CFN learning for this value of lambda

* `tb2time`: the total cpu time used by toulbar2 for validation of all samples

* `funcnumber`: the number of learned cost functions

* `exact`: True is all functions learned are soft difference cost
  functions. Scopes need to be further checked manually. Not used for
  the selection of lambda.

The validated value of lambda (with maximum performance on the
validation set) is written in a file named
`lambda-<mode>-<size>`. Note that since toulbar2 cpu-time is used as a
tie-breaking criteria for the selection of lambda, this code provides
not fully reproducible results. In our (limited) tests, this has only
very minor effects on the final accuracy of the learned solver.

Example: `python3 Sudoku-validate.py 0 15000`

will explore possible values for lambda assuming that hints and
solutions can be directly used (mode 0), using the first 15000 samples
of the RRN training set. The search depends on a few parameters that
needs to be fixed in the script:

* `lbnd` and `rbnd` give the left and right bound of the interval of
  values of lambda that will be explored a priori (the validation
  procedure can extend this interval). 
  
* `n_lambdas_half` controls the interval division factor of the
  search. At each iteration, `2*n_lambdas_half+1` evenly spaced
  values of lambda will be tried.

* `num_val` is the initial number of samples that are tested. This
  doubles at each iteration and must be a power of two.

The default values provided should work fine for mode 0 and 1. Mode 2
may require a more intense sampling of lambda_values (and possibly
increased btlimit too).
