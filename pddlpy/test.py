import unittest
from pddlpy import DomainProblem
'''
class TestStringMethods(unittest.TestCase):
    domainfile = "domain-0%d.pddl" % 1
    problemfile = "problem-0%d.pddl" % 1

    def test_ground(self):
        domprob = DomainProblem(self.domainfile, self.problemfile)
        freeop = domprob.domain.operators["op2"]
        all_grounded_opers = domprob.ground_operator("op2")
        for gop in all_grounded_opers:
            if gop.precondition_pos == set( [('S','R','C'),('S','R','S')] ):
                self.assertTrue(True)
                return
        self.assertFalse("Missed a value")
'''
def testppddl():

    domainfile = "domain-0%d.pddl" % 1
    problemfile = "problem-0%d.pddl" % 1


    domprob = DomainProblem(domainfile, problemfile)
    print(domprob.initialstate())

if __name__ == '__main__':
    testppddl()

