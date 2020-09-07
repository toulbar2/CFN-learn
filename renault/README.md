
# Learning car configuration preferences

## Prerequisites

The python3-sklearn (scikit python3) must be installed. 

## Data files
* **1-year sales history of satisfiable car configurations**: config_medium_filtered.txt (8253) / config_big_filtered.txt (8338) (car configurations incompatible with the constraints have been removed and repeated configurations have been duplicated)
* **Domains**: domain_medium.pkl / domain_big.pkl (pickle ordered domain dictionary data structures)
* **Constraints**: medium_domainsorted.xml / big_domainsorted.xml (domain values have been sorted in increasing order compared to the original files)
* **Indices of the 10-fold cross-validation for each sale history**: index_csv_medium.txt / index_csv_big.txt (duplicated configurations are kept in the same fold)

## Lambda parameters
The lambda parameter selected using the StARS algorithm for the different folds of the cross-validation are given in the corresponding files:
* lambda_medium_l1.txt
* lambda_medium_l1_l2.txt
* lambda_big_l1.txt
* lambda_big_l1_l2.txt

## Main script ../renault.py
The parameters are:
* penalty norm [0-2] (L1/L2/L1_L2 norms)
* lambda value [0-infty] (unused if last parameter set to 1)
* validation fold [0-9]
* instance name [medium|big] 
* minimum arity of learned cost functions [1|2]
* maximum arity of learned cost functions [1|2]
* combining learned preferences with known constraints [0|1]
* comparison with predictions from an oracle method knowing the test set [0|1]
* find lambda value automatically using the StARS (Stability Approach to Regularization Selection) algorithm [0|1]

## Experiments

Count the number of solutions on medium car configuration system [[Favier et al, CP2009]](http://miat.inrae.fr/degivry/Favier09a.pdf):

```
toulbar2 renault/medium_domainsorted.xml -ub=1 -a -O=-3 -B=1 -hbfs: -nopre
```

*partial output:*

```
148 unassigned variables, 426 values in all current domains (med. size:2, max size:20) and 173 non-unary cost functions (med. arity:2, med. degree:1)
Initial lower and upper bounds: [0, 1] 100.000%
Tree decomposition width  : 10
Tree decomposition height : 27
Number of clusters        : 112
Tree decomposition time: 0.007 seconds.
Number of solutions    : =  278744
Number of #goods       :    4126
Number of used #goods  :    3475
Size of sep            :    7
Time                   :    0.093 seconds
... in 2169 backtracks and 4338 nodes
```

Learn the user preferences using L1 norm with preselected lambda found by the StARS algorithm [[Liu et al, NIPS 2010]](http://papers.nips.cc/paper/3966-stability-approach-to-regularization-selection-stars-for-high-dimensional-graphical-models) and combine them to mandatory constraints in order to simulate an on-line car configuration prediction tool, using the last test fold of a 10-fold cross validation protocol:

```
python3 renault.py 0 35.112 9 medium 1 2 1 1 1
```

*partial output:*

```
Number of training samples: 7431
Number of test samples: 821
Primal and dual met after 314 iterations
The CFN has 324 cost functions
UB before preprocessing: 974.410922
UB after preprocessing: 37.393479
UB: -18.66815
UB: -18.66815
UB: -20.327802
...
```

Example for running the code on the medium dataset with an L1 penalty for 10-fold cross validation and 10 repetitions:

```
/bin/bash
filename=./renault/lambda_medium_l1.txt
declare -a myArray
myArray=(`cat "$filename"`)
for ((fold=0; fold<10 ; i++))
do
    for ((j=0; j<10 ; j++))
    do
	    python3 renault.py 0 ${myArray[$fold]} ${fold} medium 1 2 1 1 ${j}
    done
done
```

![medium.png](http://genoweb.toulouse.inra.fr/~degivry/evalgm/medium.png)
