import pytoulbar2 as tb2

class CFN:
    def __init__(self, twoSolutions, btLimit, configuration):
        tb2.init()
        if btLimit:
            tb2.option.backtrackLimit = btLimit
        tb2.option.verbose = -1
        tb2.option.hbfs = True
        tb2.option.showSolutions = False
        tb2.option.weightedTightness = True
        if configuration:
            tb2.option.elimDegree_preprocessing = 1
            tb2.option.solutionBasedPhaseSaving = False     # WARNING! False: do not reuse previous complete solutions as hints during incremental solving used by structure learning evaluation procedure!

        tb2.option.decimalPoint = 6
        if twoSolutions:
            tb2.option.allSolutions = 2     # find two solutions and quit (incompatible with incremental solving)
        self.Variables = {}
        self.VariableIndices = {}
        self.Scopes = []
        self.VariableNames = []
        if configuration:
            self.CFN = tb2.Solver(1000000000)
        else:
            self.CFN = tb2.Solver()
        self.Contradiction = tb2.Contradiction
        self.SolverOut = tb2.SolverOut
        self.Option = tb2.option
        tb2.check()

    def __del__(self):
        del self.Scopes
        del self.Variables
        del self.VariableIndices
        del self.VariableNames
        del self.CFN

    def NoPreprocessing(self):
        tb2.option.elimDegree = -1
        tb2.option.elimDegree_preprocessing = -1
        tb2.option.preprocessTernaryRPC = 0
        tb2.option.preprocessFunctional = 0
        tb2.option.costfuncSeparate = False
        tb2.option.preprocessNary = 0
        tb2.option.DEE = 0
        tb2.option.MSTDAC = False
        tb2.option.trwsAccuracy = -1

    def AddVariable(self, name, values):
        if name in self.Variables:
            raise RuntimeError(name+" already defined")
        cardinality = len(values)
        self.Variables[name] = values
        vIdx = self.CFN.wcsp.makeEnumeratedVariable(name, 0, len(values)-1)
        self.VariableIndices[name] = vIdx
        for vn in values:
            self.CFN.wcsp.addValueName(vIdx, vn)
        self.VariableNames.append(name)

    def AddFunction(self, scope, costs):
        sscope = set(scope)
        if len(scope) != len(sscope):
            raise("Error: duplicate variable in scope")
        arity = len(scope)
        for i, v in enumerate(scope):
            if isinstance(v, str):
                v = self.VariableIndices[v]
            if (v < 0 or v >= len(self.VariableNames)):
                raise("Error: out of range variable index") 
            scope[i] = v
        if (len(scope) == 1):
            self.CFN.wcsp.postUnaryConstraint(scope[0], costs, False)
        elif (len(scope) == 2):
            self.CFN.wcsp.postBinaryConstraint(scope[0], scope[1], costs, False)
        else:
            raise NameError('Higher than 2 arity functions not implemented yet in Python layer.')
        self.Scopes.append(sscope)
        return

    def Read(self, problem):
        self.CFN.read(problem)

    def Parse(self, certificate):
        self.CFN.parse_solution(certificate, False)    # important! False: do not reuse certificate in future searches used by structure learning evaluation procedure!

    def Domain(self, varIndex):
        return self.CFN.wcsp.getEnumDomain(varIndex)

    def GetUB(self):
        return self.CFN.wcsp.getDPrimalBound()

    # decreasing upper bound only
    def UpdateUB(self, cost):
        icost = self.CFN.wcsp.DoubletoCost(cost)
        self.CFN.wcsp.updateUb(icost)
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    # important! used in incremental solving for changing initial upper bound up and down before adding any problem modifications
    def SetUB(self, cost):
        icost = self.CFN.wcsp.DoubletoCost(cost)
        self.CFN.wcsp.setUb(icost)  # must be done after problem loading
        self.CFN.wcsp.initSolutionCost()  # important to notify previous best found solution is no more valid
        self.CFN.wcsp.enforceUb()   # this might generate a Contradiction exception

    def Depth(self):
        return tb2.store.getDepth()

    # make a copy of the current problem and move to store.depth+1
    def Store(self):
        tb2.store.store()

    # restore previous copy made at a given depth
    def Restore(self, depth):
        tb2.store.restore(depth)

    def GetNbNodes(self):
        return self.CFN.getNbNodes()

    def GetNbBacktracks(self):
        return self.CFN.getNbBacktracks()

    # non-incremental solving method
    def Solve(self):
        self.CFN.wcsp.sortConstraints()
        solved = self.CFN.solve()
        if (solved):
            return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), len(self.CFN.solutions())
        else:
            return None
    # incremental solving: perform initial preprocessing before all future searches, return improved ub
    def SolveFirst(self):
        self.CFN.wcsp.sortConstraints()
        ub = self.CFN.wcsp.getUb()
        self.CFN.beginSolve(ub)
        try:
            ub = self.CFN.preprocessing(ub)
        except tb2.Contradiction:
            self.CFN.wcsp.whenContradiction()
            print('Problem has no solution!')
            return None
        return self.CFN.wcsp.Cost2ADCost(ub)

    # incremental solving: find the next optimum value after a problem modification (see also SetUB)
    def SolveNext(self):
        initub = self.CFN.wcsp.getUb()
        initdepth = tb2.store.getDepth()
        self.CFN.beginSolve(initub)
        tb2.option.hbfs = 1     # reinitialize this parameter which can be modified during hybridSolve()
        try:
            try:
                tb2.store.store()
                self.CFN.wcsp.propagate()
                lb, ub = self.CFN.hybridSolve()
            except tb2.Contradiction:
                self.CFN.wcsp.whenContradiction()
        except tb2.SolverOut:
            tb2.option.limit = False
        tb2.store.restore(initdepth)
        if self.CFN.wcsp.getSolutionCost() < initub:
            return self.CFN.solution(), self.CFN.wcsp.getDPrimalBound(), None   # warning! None: does not return number of found solutions because it is two slow to retrieve all solutions in python
        else:
            return None

    def Dump(self, problem):
        if '.wcsp' in problem:
            self.CFN.dump_wcsp(problem, True, 1)
        elif '.cfn' in problem:
            self.CFN.dump_wcsp(problem, True, 2)
        else:
            print('Error unknown format!')
