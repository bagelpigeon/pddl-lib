#defines an effect

class Effect():
    def __init__(self, effectList, valueList, probList):
        self.valueList = valueList
        self.probList = probList
        #effectlist should be an actual list of effect objects
        self.effectList = effectList

    def printInfo (self):
        print("values:", self.valueList)
        print("probs:", self.probList)
        print("effectList:", self.effectList)