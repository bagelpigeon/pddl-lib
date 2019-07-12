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
    '''
    def matchStateEncodings (self, stateVectorPattern, stateEncode):
        for trueFalseValueIndex in range (len(stateVectorPattern)):
            #ignore the -1 they are counted as wildcards
            if stateVectorPattern[trueFalseValueIndex] != -1:
                if stateVectorPattern[trueFalseValueIndex]!=stateEncode[trueFalseValueIndex]:
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

    def getNextState(self, state, effectPattern):
        newState = list(copy.deepcopy(state))
        for predIndex in range(len(state)):
            if effectPattern[predIndex] != -1:
                newState[predIndex] = effectPattern[predIndex]
        return newState

    def alterActionMatrix(self, nameOfAction, stateVectorPattern, effectVectorPattern, prob):

        actionIndex = self.actionDict[nameOfAction]
        tMatrixOfAction = self.transitionMatrices[actionIndex]

        #get all states that match the stateVector pattern, -1s are wildcards
        rowsToChange = []
        columnsToChange = []
        for stateIndex in range(len(self.stateEncodings)):
            if self.matchStateEncodings(stateVectorPattern, self.stateEncodings[stateIndex]):
                rowsToChange.append(stateIndex)
            if self.matchStateEncodings(effectVectorPattern, self.stateEncodings[stateIndex]):
                columnsToChange.append(stateIndex)
        #have issue, do columns and rows match up? how to match them?

        # have to be relative to each state! so for every state that will be affected, get the next state for each of them
        for i in range(len(rowsToChange)):
            row = rowsToChange[i]
            state = self.stateEncodings[i]
            nextState = self.getNextState(state, effectVectorPattern)
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