#defines an effect

class Effect():
    def __init__(self, effectList, valueList, probList):
        #do we also want a goal desc list where
        self.valueList = valueList
        self.probList = probList
        #effectlist should just be strings(do not want nested effects)
        self.effectList = effectList

    def printInfo (self):
        print("values:", self.valueList)
        print("probs:", self.probList)
        print("effectList:", self.effectList)