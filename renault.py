import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from PEMRF import *

Norms = ["l1", "l2", "l1_l2"]

if len(sys.argv) < 5:
    print("Bad number of arguments!")
    print("norm [0-2] lambda [0-infty] validation fold [0-9] instance [medium|big] minarity [1|2] maxarity [1|2] constraints [0|1] oracle [0|1] select_lambda [0|1]")
    exit()

norm_type = Norms[int(sys.argv[1])]
Lambda = float(sys.argv[2])
num_validation_fold = int(sys.argv[3])
problem = str(sys.argv[4])
if len(sys.argv) > 5:
    minarity = int(sys.argv[5])
else:
    minarity = 1
if len(sys.argv) > 6:
    maxarity = int(sys.argv[6])
else:
    maxarity = 2
if len(sys.argv) > 7:
    isconstraints = bool(int(sys.argv[7]))
else:
    isconstraints = True
if len(sys.argv) > 8:
    isoracle = bool(int(sys.argv[8]))
else:
    isoracle = False

if len(sys.argv) > 9:
    rndseed = int(sys.argv[9])
    np.random.seed(rndseed)
else:
    np.random.seed(1)

if len(sys.argv) > 10:
    select_lambda = bool(int(sys.argv[10]))
else:
    select_lambda = False

# Load data
df_filter = pd.read_csv('renault/config_' + problem + '_filtered.txt', sep=" ")
domain_dict = pickle.load(open('renault/domain_' + problem + '.pkl', "rb"))

# Divide in training and validation sets using 10-fold cross-validation
with open('renault/index_csv_' + problem + '.txt', 'r') as foldtxt:
    fold = foldtxt.read().replace('\n', '')

hist_tab_filter = df_filter.values[[i for i in df_filter.index if int(fold[i]) != num_validation_fold]]
num_sample = len(hist_tab_filter)
print("Number of training samples:", num_sample)
validation = df_filter.values[[i for i in df_filter.index if int(fold[i]) == num_validation_fold]]
num_validation = len(validation)
print("Number of test samples:", num_validation)
scope_filter = list(df_filter.columns)
num_hint = len(scope_filter)

# Oracle needs validation database
validationpd = df_filter.iloc[[i for i in df_filter.index if int(fold[i]) == num_validation_fold]]

hint_vars_indexes = []
for i in range(num_hint):
    hint_vars_indexes.append(i)

sorted_domain_dict = {}
for k in list(scope_filter):
    sorted_domain_dict[k] = np.sort(domain_dict[k])

# Basic marginals and pairwise conditional probabilities
trainingpd = df_filter.iloc[[i for i in df_filter.index if int(fold[i]) != num_validation_fold]]
marginal = {}
conditional = {}
if isoracle:
    for k in list(scope_filter):
        for val in sorted_domain_dict[k]:
            selectconditional = trainingpd.loc[trainingpd[k] == val]
            nbx = len(selectconditional)
            marginal[k + '=v' + str(val)] = float(nbx) / float(num_sample)
            for k2 in list(scope_filter):
                if k2 != k:
                    ydomsz = len(sorted_domain_dict[k2])
                    for val2 in sorted_domain_dict[k2]:
                        conditional[k2 + '=v' + str(val2) + '|' + k + '=v' + str(val)] = float(len(selectconditional.loc[selectconditional[k2] == val2]) + 1) / float(nbx + ydomsz)

# Computes the sufficient statistics using one-hot encoding
onehotencoder = OneHotEncoder(categories=[sorted_domain_dict[k] for k in list(scope_filter)])
data = onehotencoder.fit_transform(hist_tab_filter).toarray()
data = np.array(data, dtype=int)

num_sample_init, num_nodes = hist_tab_filter.shape

m = np.ones(int(num_nodes+1), dtype=int)  # number of nodes for each type of distribution (plus initial value 1)
list_dim = [1]
for k in list(scope_filter):
    list_dim.append(len(domain_dict[k]))
dim = np.array(list_dim)  # dimension for each type of distribution (plus initial value 1)

# vector that indicates the nodes that are discrete random variables
c = np.zeros(int(sum(m*dim)))
c[1:] = 1

d = np.sum(np.multiply(m, dim))-1

balance = np.ones(len(dim))

# Lambda selection using stability criterion
if select_lambda:
    lamb = np.logspace(-5,3,100)
    v, Lambda, Adj_matrix = stars_parameter_selection(lamb, data, m, dim, c, balance, norm_type=norm_type, beta=0.05, N=50)

# Build the A matrix
A = Amatrix(data, m, dim, balance, c)

# Run PE-MRF
Z_init = np.zeros([d+1, d+1])
U_init = np.zeros([d+1, d+1])
CFNdata = ADMM(A, Z_init, U_init, Lambda, num_sample, m, dim, norm_type, save=7)

print("The CFN has", CFNcount(*CFNdata)[0], "cost functions")

cfn = toCFN(*CFNdata, assign=None, btLimit=50000, min_arity=minarity, max_arity=maxarity, varnames=scope_filter, domains=sorted_domain_dict, configuration=True)   # create CFN from ADMM learnt Markov random field
cfn.Dump('renault_mrf.cfn')
if isconstraints:
    cfn.Read('renault/' + problem + '_domainsorted.xml')   # merge with mandatory constraints from Renault
cfn.Option.xmlflag = False  # avoid specific XML competition output messages
#cfn.Dump('renault_tmp.cfn')
initub = cfn.GetUB()
print('UB before preprocessing:', initub)
cfn.NoPreprocessing()
initub = cfn.SolveFirst()   # perform initial propagation and all preprocessing only once (but no variable elimination)
if initub is not None:
    print('UB after preprocessing:', initub)
else:
    print('Stop! No solution found in preprocessing!')
    exit(1)
cfn.Dump('renault_pre.cfn')
initdepth = cfn.Depth()
error = 0
maxbt = 0
nguess = [0] * num_hint
ndiff = [0] * num_hint
ndifforacle = [0] * num_hint
ndiffmarginal = [0] * num_hint
ndiffnaivebayes = [0] * num_hint
selectvalidation = validationpd
bad = []
for s, truth in enumerate(validation):
    ltruth = [int(list(sorted_domain_dict[scope_filter[i]]).index(v)) for i, v in enumerate(truth)]
    atruth = ',' + ','.join([scope_filter[i] + '=' + str(v) for i, v in enumerate(ltruth)])
    cfn.Store()
    cfn.SetUB(initub)
    try:
        cfn.Parse(atruth)
    except cfn.Contradiction:
        cfn.CFN.wcsp.whenContradiction()
        print('Sample', s, 'parse error', atruth)
        cfn.Restore(initdepth)
        continue
    res = cfn.SolveNext()
    if res:
        ub = res[1]
    else:
        print('Sample', s, 'evaluate error', atruth)
        cfn.Restore(initdepth)
        continue
    cfn.Restore(initdepth)
    print("UB:", ub)  # , " sol:", atruth)
    if isoracle:
        selectvalidation = validationpd
    permute = list(np.random.permutation(num_hint))
    for cur_hint in range(num_hint):
        nguess[cur_hint] += 1
        lhint = [(hint_vars_indexes[permute[i]], ltruth[hint_vars_indexes[permute[i]]]) for i in range(cur_hint)]
        ahint = ',' + ','.join([scope_filter[i] + '=' + str(v) for i, v in lhint])
        cur_var = hint_vars_indexes[permute[cur_hint]]
#        print(cur_var, ltruth[cur_var], ahint)

        # Oracle
        if isoracle:
            if cur_hint > 0:
                selectvalidation = selectvalidation.loc[selectvalidation[scope_filter[hint_vars_indexes[permute[cur_hint - 1]]]] == int(truth[hint_vars_indexes[permute[cur_hint - 1]]])]
            ntruehint = len(selectvalidation)
            bestval = None
            bestn = 0
            bestmarginal = 0.
            bestmarginalval = None
            bestnaivebayes = 0.
            bestnaivebayesval = None
            for val in sorted_domain_dict[scope_filter[cur_var]]:
                nbv = len(selectvalidation.loc[selectvalidation[scope_filter[cur_var]] == val])
                if nbv > bestn:
                    bestval = val
                    bestn = nbv
                marginalscore = marginal[scope_filter[cur_var] + '=v' + str(val)]
                if marginalscore > bestmarginal:
                    bestmarginal = marginalscore
                    bestmarginalval = val
                bayescore = marginalscore
                for i in range(cur_hint):
                    bayescore *= conditional[scope_filter[hint_vars_indexes[permute[i]]] + '=v' + str(truth[hint_vars_indexes[permute[i]]]) + '|' + scope_filter[cur_var] + '=v' + str(val)]
                if bayescore > bestnaivebayes:
                    bestnaivebayes = bayescore
                    bestnaivebayesval = val
            ndifforacle[cur_hint] += int(bestval != truth[cur_var])
            ndiffmarginal[cur_hint] += int(bestmarginalval != truth[cur_var])
            ndiffnaivebayes[cur_hint] += int(bestnaivebayesval != truth[cur_var])
#            print(s, cur_hint, ntruehint, bestn, bestval, truth[cur_var])

        cfn.Store()
        cfn.SetUB(ub+1e-6)
#        cfn.Dump('renault_test.cfn')
        try:
            cfn.Parse(ahint)
        except cfn.Contradiction:
            cfn.CFN.wcsp.whenContradiction()
            error += 1
            print('Hint parse error', error, ahint)
            cfn.Dump('renault_error' + str(error) + '.cfn')
            ndiff[cur_hint] += 1
            bad.append((lhint, ltruth[cur_var], None))
            cfn.Restore(initdepth)
            continue
        bt_prev = cfn.GetNbBacktracks()
        sol = cfn.SolveNext()
        if sol:
            diff = int(sol[0][cur_var] != ltruth[cur_var])
            ndiff[cur_hint] += diff
#            print(s, cur_hint, cur_var, sol[0][cur_var], ltruth[cur_var], diff)
            if diff > 0:
                solution = [sorted_domain_dict[scope_filter[i]][int(v)] for i, v in enumerate(list(sol[0])) if i < len(scope_filter)]
                bad.append((lhint, ltruth[cur_var], solution[cur_var]))
            bt = cfn.GetNbBacktracks()
            maxbt = max(maxbt, bt - bt_prev)
        else:
            error += 1
            print('Hint solve error', error, ahint)
            cfn.Dump('renault_error' + str(error) + '.cfn')
            ndiff[cur_hint] += 1
            bad.append((lhint, ltruth[cur_var], None))
        cfn.Restore(initdepth)

print(len(bad), "incorrect solutions")
print('PE-MRF:')
print('hint     precision (%)')
for i in range(num_hint):
    print(i, 100.*(nguess[i]-ndiff[i])/nguess[i])
print('Maximum number of backtracks: ', maxbt)
# print(bad)

if isoracle:
    print('Oracle:')
    print('hint     precision (%)')
    for i in range(num_hint):
        print(i, 100.*(nguess[i]-ndifforacle[i])/nguess[i])
    print('Marginals only:')
    print('hint     precision (%)')
    for i in range(num_hint):
        print(i, 100.*(nguess[i]-ndiffmarginal[i])/nguess[i])
    print('Naive Bayes:')
    print('hint     precision (%)')
    for i in range(num_hint):
        print(i, 100.*(nguess[i]-ndiffnaivebayes[i])/nguess[i])
