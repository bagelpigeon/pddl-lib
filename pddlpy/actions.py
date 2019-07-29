import itertools
import numpy as np
import copy

class Actions():
    def __init__(self):
        # hold the indices of state transitions for each action
        self.actionDict = {"dunk-package": 0}
        # list of matrices, access
        self.transitionMatrices = []
        self.predicateDict = {"bomb-in-package?pkg":0, "bomb-in-package2": 1, "toilet-clogged": 2, "bomb-defused":3}
        #self.predicateDict = {"bomb-in-package?pkg": 0, "toilet-clogged": 1, "bomb-defused": 2}
        self.stateEncodings = []
        self.createMatrices()
        self.TRUE = 1
        self.FALSE = 0

    def createMatrices(self):
        #create state encodings for lookup purposes
        numOfPredicates = len(self.predicateDict.keys())
        self.stateEncodings = list(itertools.product([0, 1], repeat=numOfPredicates))
        self.numOfStates = len(self.stateEncodings)
        #create empty transition matrices for each action
        for actions in range(len(self.actionDict.keys())):
            self.transitionMatrices.append(np.zeros((self.numOfStates, self.numOfStates)))

    def printInfo(self):
        print("Available Actions:", self.actionDict.keys())
        print("Available Predicates:", self.predicateDict.keys())
        print("Transition Matrices")
        for stateEncode in self.stateEncodings:
            print(stateEncode)
        for matrixIndex in range(len(self.transitionMatrices)):
            print(str(self.transitionMatrices[matrixIndex]))

    '''
    returns a true if the state encoding matches the vector pattern
    Vector pattern format is this
    x are the values for each predicate
    t is a flag indicating whether this pattern should be matched exactly or
    if the state vector should be matched as a notGoal
    This means that if there is only one predicate called isRaining and a state is supposed to match this vec
    it will be [1,1]
    Fro all other states, a vector match could be [1,0] where the isRaining is false in that state, but 0 indicates that
    the state isn't supposed to match this vector;
    [x, x, x, t]
    this has to be changed to take into account states that are not supposed to match
    '''
    def matchStateEncodings (self, stateVectorPattern, stateEncode):
        if stateVectorPattern[len(stateVectorPattern)-1] == 1:
            for trueFalseValueIndex in range (len(stateVectorPattern)-1):
                #ignore the -1 they are counted as wildcards
                if stateVectorPattern[trueFalseValueIndex] != -1:
                    if stateVectorPattern[trueFalseValueIndex] != stateEncode[trueFalseValueIndex]:
                        return False
        else:
            for trueFalseValueIndex in range (len(stateVectorPattern)-1):
                #ignore the -1 they are counted as wildcards
                if stateVectorPattern[trueFalseValueIndex] != -1:
                    if stateVectorPattern[trueFalseValueIndex] == stateEncode[trueFalseValueIndex]:
                        return False
        return True

    def matchStateEncodingsUnconditional(self, stateVectorPattern, stateEncode):
        for trueFalseValueIndex in range(len(stateVectorPattern)):
            # ignore the -1 they are counted as wildcards
            if stateVectorPattern[trueFalseValueIndex] != -1:
                if stateVectorPattern[trueFalseValueIndex] != stateEncode[trueFalseValueIndex]:
                    return False
    '''
    This matches a list of effect patterns
    Primarily used to search for states that match a list of goal descriptions
    '''
    def matchListOfStateEncodings(self, patternList, stateVector, unconditional=True):
        for pattern in patternList:
            if unconditional == False:
                if self.matchStateEncodings(pattern, stateVector) == False:
                    return False
            else:
                if self.matchStateEncodingsUnconditional(pattern, stateVector) == False:
                    return False
        return True

    '''
    only returns true if tstate encoding exactly matches the other
    '''
    def exactMatchStateEncodings(self, stateVector, otherStateVector):
        for trueFalseValueIndex in range (len(stateVector)):
            if (stateVector[trueFalseValueIndex] != otherStateVector[trueFalseValueIndex]):
                return False
        return True

    def getNextState(self, state, effectPatternList):
        newState = list(copy.deepcopy(state))
        for effectPattern in effectPatternList:
            for predIndex in range(len(state)):
                if effectPattern[predIndex] != -1:
                    newState[predIndex] = effectPattern[predIndex]
        return newState

    #def alterActionMatrix(self, nameOfAction, stateVectorPatternList, effectVectorPattern, prob):
    def alterActionMatrix(self, nameOfAction, effects):
        actionIndex = self.actionDict[nameOfAction]
        tMatrixOfAction = self.transitionMatrices[actionIndex]
        for effectComboIndex in range(len(effects.effectPatternList)):
            stateVectorPatternList = effects.goalPatternList[effectComboIndex]
            effectVectorPattern = effects.effectPatternList[effectComboIndex]
            probList = effects.probList[effectComboIndex]

            #calculate probability by looping over entire list and multiplying by all probs in list
            prob = 1.0
            for probability in probList:
                prob = prob * probability

            #get all states that match the stateVector pattern, -1s are wildcards
            rowsToChange = []
            columnsToChange = []
            for stateIndex in range(len(self.stateEncodings)):
                if self.matchListOfStateEncodings(stateVectorPatternList, self.stateEncodings[stateIndex], unconditional=False):
                    rowsToChange.append(stateIndex)
                if self.matchListOfStateEncodings(effectVectorPattern, self.stateEncodings[stateIndex]):
                    columnsToChange.append(stateIndex)
            #have issue, do columns and rows match up? how to match them?

            # have to be relative to each state! so for every state that will be affected, get the next state for each of them
            for i in range(len(rowsToChange)):
                row = rowsToChange[i]
                state = self.stateEncodings[i]
                nextState = self.getNextState(state, effectVectorPattern)

                #means the state did not change
                if self.exactMatchStateEncodings(nextState, state) != True:
                    for column in columnsToChange:
                        if self.exactMatchStateEncodings(self.stateEncodings[column], nextState):
                            #column = columnsToChange[j]
                            if tMatrixOfAction[row][column] == 0:
                                tMatrixOfAction[row][column] = prob
                            else:
                                tMatrixOfAction[row][column] *= prob
        print(self.printInfo())

    def getPredicateIndex(self, nameOfPredicate):
        if nameOfPredicate == 'null':
            return -1
        return self.predicateDict[nameOfPredicate]

    def getNumPredicates(self):
        return (len(self.predicateDict.keys()))