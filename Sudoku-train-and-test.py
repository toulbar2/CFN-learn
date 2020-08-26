#!/usr/bin/env python3
import lzma
import numpy as np
import math,time,sys,os
from numpy import linalg as la
from PEMRF import *
import pandas as pd
from colorama import Fore, Style
import pickle

# computes a "best" solution of the CFN from exact hints. We try a
# heuristic ub and if no solution is found relax it.
def find_best_sol0(CFNdata,  hints):
    tb2_time = 0
    # we assign the hints and ask for a less than zero cost solution
    cfn = toCFN(*CFNdata, assign = hints,  btLimit = btlimit)
    cfn.UpdateUB(1e-6)
    ctime = time.process_time()            
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime
    del cfn
    if (not sol):
        cfn = toCFN(*CFNdata, assign = hint, btLimit = btlimit)
        ctime = time.process_time()            
        sol = cfn.Solve()
        tb2_time += time.process_time()-ctime
        del cfn
    return sol, tb2_time

# computes a best solution of the CFN knowing only the hints as images
def find_best_sol12(CFNdata, fuzz_hints):
    tb2_time = 0
    # we add the unary cost functions that represent the confidence
    # scores of LeNet and use the heurisric ub
    cfn = toCFN(*CFNdata, weight = fuzz_hints,  btLimit = btlimit)
    cfn.UpdateUB(1e-6)
    ctime = time.process_time()                
    sol = cfn.Solve()
    tb2_time += time.process_time()-ctime    
    del cfn
    # else we relax our bound
    if (not sol):
        cfn = toCFN(*CFNdata, weight = fuzz_hints, btLimit = btlimit)
        ctime = time.process_time()            
        sol = cfn.Solve()
        tb2_time += time.process_time()-ctime
        del cfn
    return sol, tb2_time

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

# print the true solution with hints, its LeNet decoded variant (if given, mode
# 1/2) and the predicted solution (if given/found)
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

    
# Main section
Norms = ["l1","l2","l1_l2"]

if (len(sys.argv) not in [4,5]):
    print("Bad number of arguments!")
    print("mode [0-2] training sample size [1000-180000] test set [satnet-test/rrn-test-??] {btlimit}")
    exit()

mode = int(sys.argv[1])
norm_type = Norms[0]
training_size= int(sys.argv[2])
num_sample = 1000
A_matrix_fn = os.path.join("train-sufficient-statistics",("A_"+str(training_size) if mode < 2 else "B_"+str(training_size))+".xz")
test_set = os.path.join("Sudoku",os.path.join("test-sets",sys.argv[3]+".csv.xz"))
btlimit = 50000
if (len(sys.argv) == 5):
    btlimit = int(sys.argv[4])

test_CSV = pd.read_csv(test_set,sep=",",nrows=num_sample,header=None).values
test_hints = test_CSV[:][:,0]
test_sols = test_CSV[:][:,1]

# for noisy hints
exp_logits = pickle.load(lzma.open(os.path.join("Sudoku","LeNet-outputs/MNIST_test_marginal.xz"), "rb"))
exp_logits_len = list(map(lambda x: len(x), exp_logits))

num_nodes = 81
num_values = 9

m = np.array([1, num_nodes])    # number of nodes for each type of distribution (plus initial value 1)
dim = np.array([1, num_values]) # dimension for each type of distribution (plus initial value 1)
d = np.sum(np.multiply(m, dim))-1

# Load the precomputed A matrix
A = pickle.load(lzma.open(os.path.join("Sudoku",A_matrix_fn),"rb"))

with open(os.path.join("Sudoku","lambdas/lambda-"+sys.argv[1]+"-"+sys.argv[2]),'r') as f:
    lamb = float(f.read())
print(Fore.CYAN + "Lambda is",lamb)

Z_init = np.ones([d+1,d+1])*0.2
U_init = np.zeros([d+1, d+1])
ctime = time.process_time()
CFNdata = ADMM(A, Z_init, U_init, lamb, training_size, m, dim, norm_type, 4)
ADMM_time = time.process_time()-ctime

func_count,exact = CFNcount(*CFNdata)
print("The CFN has",func_count,"binary functions")
print("The CFN has only (soft) differences: ",exact,Style.RESET_ALL)
        
ndiff = 0
bad = 0
total_tb2_time = 0

for s,hint in enumerate(test_hints):
    ltruth = [int(v)-1 for v in test_sols[s].strip()]
    lhint = [int(v)-1 for v in hint]
    sol = []
    tb2_time = 0
    if (mode == 0):
        # mode 0: ltruth and hints are available as digits
        sol, tb2_time = find_best_sol0(CFNdata, lhint)
    else:
        sol, tb2_time = find_best_sol12(CFNdata, MNIST_transform(hint,s))
    total_tb2_time += tb2_time
    
    if (sol):
        pgrid(mode,ltruth,lhint,list(sol[0]), MNIST_decode(test_sols[s].strip(), s) if mode > 0 else None)
        if (mode < 2):
            # exact solution known, we can count the number of
            # wrong cells
            diff = sum(a != b for a, b in zip(list(sol[0]), ltruth))
        else:
            # the solution is only available as an image. We
            # compute the LeNet scores of the predicted digit
            # in the soft_max output of LeNet on the
            # handwritten digit used.
            lread = MNIST_transform(test_sols[s].strip(),s)
            diff = sum([lread[i][sol[0][i]] for i in range(num_nodes)])
        ndiff += diff
        if (diff > 0):
            print(Fore.RED,"Best solution has score:",diff," Sample",s+1,"/",num_sample,Style.RESET_ALL)
            bad += 1
        else:
            print(Fore.GREEN,"Zero score solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        # no solution found in the backtrack budget, this is bad, all cell predictions are bad too
    else:
        pgrid(mode,ltruth,lhint,None, MNIST_decode(test_sols[s].strip(), s) if mode > 0 else None)
        print(Fore.RED,"No solution found. Sample",s+1,"/",num_sample,Style.RESET_ALL)
        bad += 1
        ndiff += 81*(20 if mode == 2 else 1)

probe = (lamb, num_sample, bad, ndiff/(num_sample*81), ADMM_time, total_tb2_time, func_count, exact)

print(Fore.CYAN +"======================================")
print("=====> Lambda:", lamb)
print("=====> Number of incorrect solutions :", bad,"/", num_sample)
print("=====> Ratio of wrongly guessed cells:", probe[3])
print("=====> ADMM cpu-time                 :", probe[4])
print("=====> Toulbar2 average cpu-time     :", probe[5]/num_sample)
print("=====> Number of functions in model  :", probe[6])
print("=====> Exact model                   :", probe[7])
print("======================================", Style.RESET_ALL)

filename = "test-"+sys.argv[1]+"-"+sys.argv[2]+"-"+sys.argv[3] 

with open(filename, 'w') as file:
    file.write("training_size,correct_grid_ratio,correct_cell_ratio,ADMM_time,total_toulbar2_time, funcnumber, exact\n")   
    file.write(str(training_size)+", "+str((num_sample-bad)/num_sample)+", "+str(1.0-probe[3])+", "+str(probe[4])+", "+str(probe[5])+", "+str(probe[6])+", "+str(int(probe[7])))
    file.write("\n")

