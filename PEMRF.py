#!/usr/bin/python

import numpy as np
from numpy import linalg as la
import sys
from collections import OrderedDict
from decimal import Decimal
import gzip
import CFN
import simplejson as json
from json import encoder

def get(dim,i):
    if (1+i >= len(dim)):
        return int(dim[-1])
    else:
        return int(dim[1+i])

def isDiag(M):
    i, j = M.shape
    assert i == j 
    test = M.reshape(-1)[:-1].reshape(i-1, j+1)
    return ~np.any(test[:, 1:])

def CFNcount(mat, Theta, m, dim, List_indices, assign = False):
    num_nodes = np.sum(m) - 1
    cfn_number = 0
    tolerance = pow(10, -6)
    is_sudoku = True
    for i in range(num_nodes-1):
        for j in range(i+1, num_nodes):
            ETable = np.asarray(Block_Extract(mat, i+1, j+1, List_indices), dtype=np.longdouble)
            if (np.max(abs(ETable.reshape(get(dim,i) * get(dim,j)))) >= tolerance):  # else useless zero function
                cfn_number += 1
                is_sudoku &= (get(dim,i) == get(dim,j) and isDiag(ETable) and (np.min(np.diag(ETable)) > tolerance))
    return cfn_number, is_sudoku

def toCFN(Z, Theta, m, dim, List_indices, btLimit = None, assign = None, weight = None, twoSol = False, min_arity = 2, max_arity = 2, harden = False, varnames = None, domains = None, configuration = False):
    myCFN = CFN.CFN(twoSol,btLimit,configuration)
    num_nodes = np.sum(m) - 1
    tolerance = pow(10, -6)
    # Constant term if desired
    if (min_arity <= 0 and max_arity >= 0):
        ETable = np.asarray((Block_Extract(Z, 0, 0, List_indices)))
        if (np.max(abs(ETable)) >= tolerance):
            myCFN.AddFunction([], ETable)

    # Variables
    for i in range(num_nodes):
        if varnames:
            var_name = varnames[i]
            if domains:
                values = ['v' + str(e) for e in domains[var_name]]
            else:
                values = [str(z) for z in range(1, get(dim, i) + 1)]
        else:
            var_name = "x"+str(i)
            values = [str(z) for z in range(1, get(dim,i)+1)]
        myCFN.AddVariable(var_name, values)

        ETable = np.full(get(dim,i),0.0,dtype=np.longdouble)
        # Unary terms if desired
        if (min_arity <= 1 and max_arity >= 1):
            ETable += np.asarray((Block_Extract(Z, 0, i+1, List_indices) +
                                  np.transpose(Block_Extract(Z, i+1, 0, List_indices)) +
                                  np.diag(Block_Extract(Z, i+1, i+1, List_indices)))[0], dtype=np.longdouble)
        # Value assignment if desired
        if (assign):
            val = int(assign[i]) if (i<len(assign)) else -1
            if (val >= 0 and val < get(dim,i)):
                ATable = np.full(get(dim,i), 1000.0, dtype=np.longdouble)
                ATable[val] = 0.0
                ETable += ATable
        # Unary weight bias if desired
        if (weight):
            if (weight[i]):
                ETable += weight[i]
        
        if (np.max(abs(ETable)) >= tolerance):
            myCFN.AddFunction([i], ETable)

    # Binary terms
    if (min_arity <= 2 and max_arity >= 2):
        for i in range(num_nodes-1):
            for j in range(i+1, num_nodes):
                ETable = np.asarray(Block_Extract(
                    Z, i+1, j+1, List_indices).reshape(get(dim,i) * get(dim,j)), dtype=np.longdouble)
                if (harden):
                    ETable[ ETable >= harden ] = 1000.0
                if (np.max(abs(ETable)) >= tolerance):
                    myCFN.AddFunction([i, j], ETable)
    return myCFN


def write_cfn_gzip(cfn, output_filename, comment="", indent=0):
    cfn_str = json.dumps(cfn, indent=indent,
                         separators=(',', ':'), use_decimal=True)
    cfn_bytes = cfn_str.encode('utf-8')
    with gzip.GzipFile(output_filename, 'w') as fout:
        if (comment):
            fout.write(('# '+comment+'\n').encode('utf-8'))
        fout.write(cfn_bytes)


def dump_mat(mat, m, dim, precision, List_indices):
    """
    dump the learnt MRF in to a JSON/CFN
    mat is the matrix (theta or Z)
    """

    energyformat = "{:."+str(precision)+"f}"
    tolerance = pow(10, -precision)
    num_nodes = np.sum(m) - 1
    ub = 0.0
    varsDict = OrderedDict()
    funsDict = OrderedDict()

    for i in range(num_nodes):
        varname = "x_"+str(i)
        varsDict[varname] = get(dim,i)
        funDict = OrderedDict()
        funDict['scope'] = [varname]
        ETable1 = [Decimal(energyformat.format(e)) for e in (Block_Extract(
            mat, 0, i+1, List_indices) + np.transpose(Block_Extract(mat, i+1, 0, List_indices)) + np.diag(Block_Extract(mat, i+1, i+1, List_indices)))[0]]
        ub += float(max(ETable1))
        funDict['costs'] = ETable1
        funsDict['E_'+str(i)] = funDict

        for j in range(i+1, num_nodes):
            ovarname = "x_"+str(j)
            temp = Block_Extract(mat, i+1, j+1, List_indices)
            if (np.max(abs(temp)) >= tolerance):  # else useless zero function
                funDict = OrderedDict()
                funDict['scope'] = [varname, ovarname]
                ETable2 = [Decimal(energyformat.format(e))
                           for e in Block_Extract(mat, i+1, j+1, List_indices).reshape(get(dim,i) * get(dim,j))]
                ub += float(max(ETable2))
                funDict['costs'] = ETable2
                funsDict['E_'+str(i)+"_"+str(j)] = funDict

    header = OrderedDict()
    header['name'] = "ADMM_MRF"
    header['mustbe'] = '<'+energyformat.format(ub)

    cfn = OrderedDict()
    cfn["problem"] = header
    cfn["variables"] = varsDict
    cfn["functions"] = funsDict
    return cfn


def Amatrix(data, m_vec, dim_vec, balance, c):
    """
    data: the sufficient statistic data
    m_vec: array containing the number of nodes for each type of distribution (plus initial value 1)
    dim_vec: array containing the dimension for each type of distribution (plus initial value 1)
    balance = [1, balance1, balance2, balance3, balance4]
    c: binary vector that indicates the nodes that are discrete random variables
    """

    num_sam, d_num = np.shape(data)

    balance_vec = np.repeat(balance, m_vec*dim_vec)
    balance_mat = np.tile(balance_vec, (num_sam, 1))

    data_2 = np.concatenate((np.ones([num_sam, 1]), data), axis=1)
    new_data = np.divide(data_2, balance_mat)

    M = np.dot(new_data.T, new_data)/num_sam

    A = M + np.diag(c)

    return A


def Block_Extract(Z, row_ind, col_ind, List_indices):
    """
    extracts the submatrix of Z for the row row_ind and the colonne col_ind
    """

    inter_row = List_indices[row_ind]
    inter_col = List_indices[col_ind]

    num_row = len(inter_row)
    num_col = len(inter_col)

    inter_row_2 = inter_row.reshape(num_row, 1)
    inter_col_2 = inter_col.reshape(num_col, 1)

    return Z[inter_row_2, inter_col_2.T]

def Zupdate(theta, U, Eta, m, dim, List_indices, norm_type):
    """
    Update each subblock of Z
    """

    num_nodes = np.sum(m)-1

    temp_matrix = theta + U

    B = [[] for i in range(num_nodes+1)]
    for i in range(num_nodes+1):  # iterate from 0 to the number of nodes
        for j in range(num_nodes+1):
            Block_ij = Block_Extract(temp_matrix, i, j, List_indices)
            if i == 0 or j == 0 or i == j:
                B[i].append(Block_ij)
            else:
                if norm_type == 'l1_l2':
                    # Proximal of the l1/l2 norm
                    gamma = la.norm(Block_ij, 'fro')
                    # Block-wise soft-thresholding
                    B[i].append((1 - Eta[i, j]/gamma).clip(0) * Block_ij)
                elif norm_type == 'l1':
                    # Proximal of the l1 norm
                    # Soft-thresholding
                    B[i].append((Block_ij - Eta[i, j]).clip(0) -
                                (-Block_ij - Eta[i, j]).clip(0))
                elif norm_type == 'l2':
                    B[i].append(1/(1+2*Eta[i, j]) * Block_ij)

    # now B is [ [B0,0 , B0,1 , ... , B0,31], [B1,0 , B1,1 , ... , B1,31], ..., [B31,0, ..., B31,31]]
    Z_new = np.block(B)
    Z_new = (Z_new + Z_new.T)/2

    return Z_new


def createEmatrix(theta, m, dim, tolerance, List_indices):
    """
    Convert the learnt theta parameters to an adjacency matrix
    """
    num_nodes = np.sum(m)-1

    E = np.zeros([num_nodes+1, num_nodes+1])
    for i in range(num_nodes+1):
        for j in range(num_nodes+1):
            temp = la.norm(Block_Extract(theta, i, j, List_indices), 'fro')
            if np.absolute(temp) >= tolerance:
                E[i, j] = np.absolute(temp)
            else:
                E[i, j] = 0

    return E[1:, 1:]


def information_criterion(Theta, A, Ematrix, num_sample):
    """
    Computes the AIC and BIC criteria
    """
    num_param = np.sum(Ematrix != 0)
    log_likelihood = - np.trace(np.dot(Theta, A)) + np.log(la.det(Theta))
    AIC = -2*log_likelihood + 2*num_param
    BIC = -2*log_likelihood + np.log(num_sample)*num_param

    return AIC, BIC, log_likelihood


def ADMM(A, Z_init, U_init, lamb, num_sam, m, dim, norm_type='l1_l2', save=0, precision=6, K=1000, epsilon_abs=1e-3, epsilon_rel=1e-3):
    """
    ADMM algorithm for PE-MRF
    A: matrix obtained using the function Amatrix
    Z_init, U_init: initial values for Z and U
    lamb: regularization parameter
    num_sam: number of samples
    m: array containing the number of nodes for each type of distribution (plus initial value 1)
    dim: array containing the dimension for each type of distribution (plus initial value 1)
    norm_type: 'l1_l2' (group lasso), 'l1' (sparse group lasso) or 'l2' (ridge).
    save: save as CFN file False/0: none, True/1: theta, 2: Z, 3: both, 4: return data to build a CFN
    K: maximum number of iterations
    epsilon_abs/epsilon_rel: absolute/relative tolerances used in the termination criteria
    """

    # Build the matrix W containing the weights used in the penalty
    Mvec = np.array(np.repeat(dim, m))[np.newaxis]
    W = np.sqrt(np.transpose(Mvec)@Mvec)

    # Initialization
    d = np.sum(np.multiply(m, dim))-1
    Z = np.copy(Z_init)
    U = np.copy(U_init)
    rho = 4  # initial step size

    # Construct a list that gives the indices corresponding to each node
    Mvec = np.array(np.repeat(dim, m))[np.newaxis]
    List_indices = np.split(np.arange(d+1), np.cumsum(Mvec))
    del List_indices[-1]

    # Iterations
    for k in range(K):
        print(k,end='\r')
        eta1 = rho/num_sam  # define eta for theta update
        eta2 = (lamb*W)/rho  # define eta2 for Z update

        # Book-keeping previous Z for computing the dual residual
        prev_Z = np.copy(Z)

        # Update Theta
        temp = eta1*(prev_Z-U) - A
        temp = (temp + temp.T)/2

        lambvec, Q = la.eigh(temp)  # eigenvalue decomposition

        D_lamb = np.diag(
            lambvec + np.sqrt(np.power(lambvec, 2) + 4*eta1 * np.ones(len(lambvec))))
        Theta = Q @ D_lamb @ (Q.T)
        Theta = (1/(2*eta1)) * Theta
        Theta = (Theta + Theta.T)/2

        # Update Z
        Z = Zupdate(Theta, U, eta2, m, dim, List_indices, norm_type)

        # Update U
        U += Theta - Z

        # Check convergence
        R_primal = la.norm(Theta - Z, 'fro')  # primal residual
        R_dual = rho * la.norm(Z - prev_Z, 'fro')  # dual residual
        epsilon_dual = (d+1) * epsilon_abs + epsilon_rel * \
            rho * la.norm(U, 'fro')
        epsilon_pri = (d+1) * epsilon_abs + epsilon_rel * \
            max(la.norm(Theta, 'fro'), la.norm(Z, 'fro'))

        if R_primal <= epsilon_pri and R_dual <= epsilon_dual:
            print('Primal and dual met after ', end='')
            break

        # Varying step size
        if R_primal > 10*R_dual:
            rho *= 2
            U /= 2
        elif R_dual > 10*R_primal:
            rho /= 2
            U *= 2

    # END of Iteration
    print(k, 'iterations')

    if (save & 1):
        write_cfn_gzip(dump_mat(Theta, m, dim, precision,
                                List_indices), 'Theta_' + norm_type + '_'+str(lamb) + '.cfn.gz')
    if (save & 2):
        write_cfn_gzip(dump_mat(Z, m, dim, precision,
                                List_indices), 'Z_' + norm_type+'_' + str(lamb) + '.cfn.gz')
    if (save & 4):
        return (Z, Theta, m, dim, List_indices)
    
    # Convert the learnt parameters Z to obtain the predicted adjacency matrix
    E = createEmatrix(Z, m, dim, 1e-18, List_indices)
    # normalization
    E = E / np.sqrt(np.tensordot(np.diag(E), np.diag(E), axes=0))

    return Theta, Z, U, E
def regul_path(lamb, A, num_sam, m, dim, norm_type):
    """
    Computes the regularization path using warm restart
        
    lamb: vector of lambda by increasing values
    """
    
    Z_init = np.zeros(A.shape)
    U_init = np.zeros(A.shape)
    
    Adj_matrix = []
    for p in range(len(lamb)):
        optimal_theta, optimal_Z, optimal_U, Ematrix = ADMM(A, Z_init, U_init, lamb[p], num_sam, m, dim, norm_type)
        
        # Warm restart is used to initialize Z and U for the next lambda
        Z_init = np.copy(optimal_Z)
        U_init = np.copy(optimal_U)
        
        Adj_matrix.append(Ematrix)
    
    return Adj_matrix

def stars_parameter_selection(lamb, data, m, dim, c, balance, norm_type='l1', beta=0.05, N=50):
    """
    Implementation of the StARS (Stability Approach to Regularization Selection) algorithm [Liu et al., 2010]
    that selects the regularization parameter for graph inference problems based on the stability
        
    lamb: vector of lambda by increasing values (for example lamb = np.logspace(-3,-1,100))
    data: sufficient statistics
    beta: cut point value
    N: number of bootstraps
    """
    
    num_nodes = np.sum(m)-1
    num_sample = data.shape[0]
    
    Adj_matrix = [np.zeros([num_nodes, num_nodes]) for j in range(len(lamb))]
    
    for ind_boot in range(N):
        
        boot_index = np.random.choice(num_sample, int(10*np.sqrt(num_sample)), replace=False) # Random subsample
        
        A_b = Amatrix(data[boot_index,:], m, dim, balance, c)
        
        path = regul_path(lamb, A_b, num_sample, m, dim, norm_type) # run PE-MRF for all the lambda values
        
        for j in range(len(lamb)):
            # for each edge, add 1 if it has been predicted as present in the network
            Adj_matrix[j] += 1*(path[j] > 0)

    # Compute the total instability for each lambda
    v = np.zeros(len(lamb))
    iu = np.triu_indices(num_nodes, 1)
    for j in range(len(lamb)):
        Adj_matrix[j] = Adj_matrix[j] / N
        D = 2*Adj_matrix[j]*(1-Adj_matrix[j]) # instability of the edges across subsamples
        v[j] = np.mean(D[iu]) # total instability
    
    # Determine the optimal lambda
    vcum = [np.max(v[rho:]) for rho in range(len(lamb))]
    lambda_opt = np.min(lamb[np.array(vcum) <= beta])
    
    return v, lambda_opt, Adj_matrix
