#!/usr/bin/env python3
# Learn to play the Sudoku: validation phase as described in:
#  Pushing data into CP models using Graphical Model Learning and Solving.
#  Céline Brouard, Simon de Givry, Thomas Schiex
#  Proc. CP'2020, Virtually at Louvain, September 2020.
import lzma
import numpy as np
import math,time,sys,os
from numpy import linalg as la
from PEMRF import *
import pandas as pd
from colorama import Fore, Style
import pickle

# comparator for lambda statistics: we prefer the value of lambda that
# give better solutions at the grid level, and then cell level. If
# there are ex-aeco, prefer less regularization or shorter resolution
# time (comment to select)
def probe_better(a,b):
    if (a[2] != b[2]): return (a[2] < b[2]) # number of bad grids
    if (a[3] != b[3]): return (a[3] < b[3]) # ratio of bad cells
#    if (a[0] != b[0]): return (a[0] < b[0]) # regularization strength
    if (a[5] != b[5]): return (a[5] < b[5]) # shorter cpu-time
    return False

# transforms a grid of digits (hint or solution) into a grid of unary
# cost functions equal to LeNet output on corresponding MNIST
# digits. The MNIST digit chosen for a cell of a given grid is always
# the is determined by hashing the digit (c), the position of the
# digit in the grid and the index of the sample in the sample set. The
# handwritten digit used in a hint or solution will be the same in a
# pair (same digit, position and sample number).
def MNIST_transform(sample, sample_idx):
    lw = [[] for i in sample]
    for idx,c in enumerate(sample):
        if (c != '0'):
            img_idx = hash(c+str(idx)+str(sample_idx)) % exp_logits_len[int(c)]
            lwi = list(map(lambda x: max(0,-math.log(x)), exp_logits[int(c)][img_idx][1:]))
            minlwi = min(lwi)
            lw[idx] = list(map(lambda x: (x - minlwi),lwi))
    return lw

# decodes a grid (hint or solution) assuming it needs to be Lenet decoded
def MNIST_decode(sample, sample_idx):
    dec = []
    for idx,c in enumerate(sample):
        if (c != '0'):
            img_idx = hash(c+str(idx)+str(sample_idx)) % exp_logits_len[int(c)]
            dec.append(exp_logits[int(c)][img_idx][1:].argmax()+1)
        else:
            dec.append(0)
    return dec

# computes the best solution of the CFN knowing the exact solution and
# the hints. In this case, we can use the exact solution to compute a
# tight upper bound (mode 0)
def find_best_sol0(CFNdata, ltruth, hints):
    tb2_time = 0
    # the CFN is totally assigned with the known solution
    cfn = toCFN(*CFNdata, assign = ltruth, btLimit = btlimit)
    ctime = time.process_time()            
    # and the cost of the solution provides an upper bound    
    ub = cfn.Solve()[1]
    tb2_time += time.process_time()-ctime
    del cfn
    # we now assign only the hints and ask for a maximum of 2
    # assignmentq.
    cfn = toCFN(*CFNdata, assign = hints, twoSol = True, btLimit = btlimit)
    # we shift the ub by the encoding resolution to be able to recover
    # the solution
    cfn.UpdateUB(ub+1e-6)
    ctime = time.process_time()            
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime
    del cfn
    return sol, tb2_time

# computes the best solution of the CFN knowing the best solution and
# a LeNet decoding of the hints. In this case, we can use the
# exact solution to compute a tight upper bound (mode 1).
def find_best_sol1(CFNdata, ltruth, fuzz_hints):
    tb2_time = 0
    # we assign all variables knowing the exact solutions and add the
    # unary cost functions that represent the confidence scores of LeNet
    cfn = toCFN(*CFNdata, assign = ltruth, weight = fuzz_hints, btLimit = btlimit)
    ctime = time.process_time()
    ub = cfn.Solve()[1]
    tb2_time += time.process_time()-ctime    
    del cfn
    
    cfn = toCFN(*CFNdata, weight = fuzz_hints, btLimit = btlimit)
    # we shift the ub by the encoding resolution to be able to recover
    # the solution
    cfn.UpdateUB(ub+1e-6)
    ctime = time.process_time()
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime
    del cfn
    return sol, tb2_time

# computes the best solution of the CFN knowing the LeNet decoding of
# the hints and the solution. In this case, we cannot use the exact
# solution to compute a tight upper bound (mode 2).
def find_best_sol2(CFNdata, fuzz_hints):
    tb2_time = 0
    # we add the unary cost functions encoding the LeNet confidence
    # scores for the observed hints
    cfn = toCFN(*CFNdata, weight = fuzz_hints, btLimit = btlimit)
    # first, we try to find a solution with maximum zero cost
    cfn.UpdateUB(1e-6)
    ctime = time.process_time()
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime
    del cfn
    # if no solution hs been found, we relax the upper bound and possibly the btlimit
    if (not sol):
        print("No easy solution found!")
        cfn = toCFN(*CFNdata, weight = fuzz_hints, btLimit = btlimit)
        ctime = time.process_time()            
        sol = cfn.Solve()
        tb2_time += time.process_time()-ctime
        del cfn
    return sol, tb2_time

def pgrid(mode,lt,lh,lp=None,ld=None):
    print()
    print("   S O L U T I O N            ",end='')
    if (ld): print("  D E C O D E D             ",end='')
    if (lp): print("P R E D I C T E D",end='')
    print('\n')
    for i in range(9):
        for j in range(3):
            print(" ".join([Fore.WHITE+str(a+1) if a==b else Style.RESET_ALL+str(a+1) for a,b in zip(lt[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')
        print(end='    ')
        if (ld and mode == 2):
            for j in range(3):
                print(" ".join([Fore.GREEN+str(a) if (a-1)==b else Fore.RED+str(a) for a,b in zip(ld[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3])]),end='   ')
            print(end='    ')
        if (ld and mode == 1):
            for j in range(3):
                print(" ".join([Style.RESET_ALL+"-" if b<0 else Fore.GREEN+str(a) if (a-1)==b else Fore.RED+str(a) for a,b in zip(ld[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')
            print(end='    ')      
        if (lp):
            if (mode > 0):
                for j in range(3):
                    print(" ".join([Fore.GREEN+str(min(9,a+1)) if a==b else Fore.RED+str(min(9,a+1)) for a,b in zip(lp[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3])]),end='   ')
            else:
                for j in range(3):
                    print(" ".join([Fore.WHITE+str(b+1) if c==b else Fore.GREEN+str(min(9,a+1)) if a==b else Fore.RED+str(min(9,a+1)) for a,b,c in zip(lp[i*9+j*3:i*9+j*3+3],lt[i*9+j*3:i*9+j*3+3],lh[i*9+j*3:i*9+j*3+3])]),end='   ')                    
        print()
        if (i%3 == 2): print()
    print(Style.RESET_ALL)


################################################################
# Main section
################################################################
Norms = ["l1","l2","l1_l2"]

if (len(sys.argv) not in [3,4]):
    print("Bad number of arguments!")
    print("mode [0-2] training sample size [1-180000] {btlimit:20000}")
    print("mode 0: hints and solutions as digital exact information.")
    print("mode 1: hints as images, solutions as digital exact information.")
    print("mode 2: hints and solutions as images.")
    print("training sample size: a multiple of 1000, between 1000 and 180000")  
    exit()

mode = int(sys.argv[1])
norm_type = Norms[0]
num_sample = int(sys.argv[2])
num_validations = 1024
validation_set = os.path.join("Sudoku",os.path.join("validation-set","rrn-validation.csv.xz"))

# Parameters of the hunt for lambda: We explore a grid of 2n+1 values
# of lambda with a given initial sample size for testing. We restrict
# the interval around the most promising value of lambda and sart
# again with more test samples and continue until the max. number of
# samples has been reached.

# search effort limit
btlimit = 20000
if (len(sys.argv) == 4):
    btlimit = int(sys.argv[3])
# numbers of lambda explored on each side
n_lambdas_half = 4
n_lambdas = 1 + (n_lambdas_half * 2)
# left and right interval bounds for lambda
lbnd = 0.1
rbnd = 10
# number of initial samples tested (must be a power of 2)
num_val = 32

# sufficient statistics for PEMRF have been precomputed either from
# exact solutions provided as digits (matrices A*) or from LeNet
# predictions applied to randomly chosen digit images from MNIST
# (matrices B*). We use B* only for mode 2 (solutions are images)
A_matrix_fn = os.path.join("train-sufficient-statistics",("A_" if (mode < 2) else "B_")+str(num_sample)+".xz")
    
# precomputed confidence scores output by LeNet for every possible
# prediction on every possible MNIST digit.
exp_logits = pickle.load(lzma.open(os.path.join("Sudoku","LeNet-outputs/MNIST_test_marginal.xz"), "rb" ))
exp_logits_len = list(map(lambda x: len(x), exp_logits))

# read the validation set and separate it into hints and associated solutions
valid_CSV = pd.read_csv(validation_set,sep=",",nrows=num_validations,header=None).values
valid_hints = valid_CSV[:][:,0]
valid_sols = valid_CSV[:][:,1]

# number of variables and domain size of the Sudoku problem
num_nodes = 81
num_values = 9

# prepare dimensions for PEMRF (pure discrete distributions mode)
m = np.array([1, num_nodes])    # number of nodes for each type of distribution (plus initial value 1)
dim = np.array([1, num_values]) # dimension for each type of distribution (plus initial value 1)
d = np.sum(np.multiply(m, dim))-1

# load the precomputed sufficient statistics matrix
A = pickle.load(lzma.open(os.path.join("Sudoku",A_matrix_fn),"rb"))

# for each lambda tested, we collect:
# lambda, nbr tests, # bad grids, ratio of bad cells, ADMM tb2 cpu-times, #functions, exact model (right number and contents - scopes not tested here)
probes_dict = {} 

while (num_val <= num_validations):
    lratio = (math.log10(rbnd)-math.log10(lbnd))/(n_lambdas-1)
    lamb = lbnd
    probes = []
    
    for idx in range(n_lambdas):
        print(Fore.CYAN + "Lambda is",lamb)

        # prepare matrices for PEMRF
        Z_init = np.ones([d+1,d+1])*0.2
        U_init = np.zeros([d+1, d+1])
        ctime = time.process_time()
        # learn the CFN
        CFNdata = ADMM(A, Z_init, U_init, lamb, num_sample, m, dim, norm_type, 4)
        ADMM_time = time.process_time()-ctime
        
        func_count, exact = CFNcount(*CFNdata)
        print("The CFN has",func_count,"binary functions")
        print("The CFN has only (soft) differences: ",exact,Style.RESET_ALL)
        
        ndiff = 0
        bad = 0
        total_tb2_time = 0
        # if PEMRF learns no function, there is no point in increasing lambda beyond this
        # we just fill the probes with known results
        if (func_count == 0):
            probes.append((lamb, num_val, num_val, 1.0, ADMM_time, 0, 0, 0, num_val, 0, 0))
            for i in range(idx+1, n_lambdas):
                probes.append((lamb, num_val, num_val, 1.0, 0, 0, 0, 0, num_val, 0, 0))
            break

        # time for validation. We use an increasingly large subset of the validation set
        for s,hint in enumerate(valid_hints[:num_val]):
            # s is the sample number, hint is the hint and ltruth is the solution
            ltruth = [int(v)-1 for v in valid_sols[s].strip()]
            lhint = [int(v)-1 for v in hint]
            sol = []
            tb2_time = 0
            
            if (mode == 0):
                # mode 0: ltruth and hints are available as digits
                sol, tb2_time = find_best_sol0(CFNdata, ltruth, lhint)
            elif (mode == 1):
                # mode 1: we encode hints using LeNet scores
                sol, tb2_time = find_best_sol1(CFNdata, ltruth, MNIST_transform(hint,s))
            elif (mode == 2):
                # mode 2: we necode both hints and solution using LeNet score
                sol, tb2_time = find_best_sol2(CFNdata, MNIST_transform(hint,s))

            total_tb2_time += tb2_time
            
            if (sol):
                pgrid(mode,ltruth,lhint,list(sol[0]), MNIST_decode(valid_sols[s].strip(), s) if mode > 0 else None)
                if (mode < 2):
                    # exact solution known, we can count the number of
                    # wrong cells
                    diff = sum(a != b for a, b in zip(list(sol[0]), ltruth))
                else:
                    # the solution is only available as an image. We
                    # compute the LeNet scores of the predicted digit
                    # in the soft_max output of LeNet on the
                    # handwritten digit used.
                    lread = MNIST_transform(valid_sols[s],s)
                    diff = sum([lread[i][sol[0][i]] for i in range(num_nodes)])
                ndiff += diff
                if (diff > 0):
                    print(Fore.RED,"Best solution has score:",diff," Sample",s+1,"/",num_val,Style.RESET_ALL)
                    bad += 1
                else:
                    print(Fore.GREEN,"Zero score solution found. Sample",s+1,"/",num_val,Style.RESET_ALL)
            # no solution found in the backtrack budget, this is bad, all cell predictions are bad too
            else:
                pgrid(mode,ltruth,lhint,None, MNIST_decode(valid_sols[s].strip(), s) if mode > 0 else None)
                print(Fore.RED,"No solution found. Sample",s+1,"/",num_val,Style.RESET_ALL)
                bad += 1
                ndiff += 81*(20 if mode == 2 else 1)
                
        probe = (lamb, num_val, bad, ndiff/(num_val*81), ADMM_time, total_tb2_time, func_count, int(exact))
        probes.append(probe)
        probes_dict[lamb] = probe

        print(Fore.CYAN +"======================================")
        print("=====> Lambda                        :", lamb)
        print("=====> Ratio of incorrect grids      :", bad,"/", num_val)
        print("=====> Ratio of wrongly guessed cells:", probe[3])
        print("=====> ADMM cpu-time                 :", probe[4])
        print("=====> Toulbar2 average cpu-time     :", probe[5]/num_val)
        print("=====> Number of functions in model  :", probe[6])
        print("=====> Expected #/nature of functions:", probe[7])
        print("======================================", Style.RESET_ALL)

        lamb *= pow(10,lratio)

    # select the currently best Lambda (Bayesian optimization would be worth a try)
    probe = probes[0];
    for p in probes:
        if (probe_better(p, probe)):
            probe = p

    print(Fore.CYAN + "==================================================================")    
    print("=====> Best lambda:", probe[0])
    lbnd = probe[0] / pow(10,lratio * n_lambdas_half / 4)
    rbnd = probe[0] * pow(10,lratio * n_lambdas_half / 4)
    num_val = num_val * 2

    print("=====> Left bound : ",lbnd)
    print("=====> Right bound: ",rbnd)
    print("==================================================================",Style.RESET_ALL)

filename = "probe-"+str(mode)+"-"+sys.argv[2]

with open(filename, 'w') as file:
    file.write("lambda, num_val, badg, badc, ADMMtime, tb2time, funcnumber, exact\n")   
    lambdas = list(probes_dict.keys())
    lambdas.sort()
    for lamb in lambdas:
        s = ", ".join([str(i) for i in probes_dict[lamb]])
        file.write(str(s))
        file.write("\n")

fileout = "lambda-"+str(mode)+"-"+sys.argv[2]
lambdas = pd.read_csv(filename)
sorted = lambdas.sort_values(by=[" num_val"," badg"," badc"," tb2time"],ascending=[False,True,True,True])
lamb = sorted.values[0][0]
print("Best lambda is",lamb)
with open(fileout,'w') as f:
    f.write(str(lamb))

