import itertools
import numpy as np

class Actions():
    def __init__(self):
        # hold the indices of state transitions for each action
        self.actionDict = {"dunk-package": 0}
        # list of matrices, access
        self.transitionMatrices = []
        self.predicateDict = {"bomb-in-package?pkg":0, "bomb-in-package2": 1, "bomb-defused": 2, "toilet-clogged":3}
        self.stateEncodings = []
        self.createMatrices()

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
        for matrix in self.transitionMatrices:
            print(matrix)

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


        for i in range(len(rowsToChange)):
            row = rowsToChange[i]
            column = columnsToChange[i]
            if tMatrixOfAction[row][column] == 0:
                tMatrixOfAction[row][column] = prob
            else:
                tMatrixOfAction[row][column] *= prob

    def getPredicateIndex(self, nameOfPredicate):
        print (nameOfPredicate)
        return self.predicateDict[nameOfPredicate]

    def getNumPredicates(self):
        return (len(self.predicateDict.keys()))