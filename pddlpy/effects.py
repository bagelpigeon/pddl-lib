#defines an effect

class Effects():
    def __init__(self):
        self.goalPatternList = []
        self.effectPatternList = []
        self.probList = []
        self.valueList = []

    def printInfo (self):
        print("values:", self.valueList)
        print("probs:", self.probList)
        print("effectList:", self.effectPatternList)
        print("goalPatternList:", self.goalPatternList)