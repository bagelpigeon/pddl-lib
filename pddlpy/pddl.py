#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Copyright 2015 Hernán M. Foffani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

from antlr4 import *
from pddlLexer import pddlLexer
from pddlParser import pddlParser
from pddlListener import pddlListener
from pddlVisitor import pddlVisitor
from actions import Actions
from effect import Effect
from effects import Effects
import itertools
import numpy as np

import itertools
from itertools import permutations


class Atom():
    def __init__(self, predicate):
        self.predicate = predicate

    def __repr__(self):
        return str(tuple(self.predicate))

    def ground(self, varvals):
        g = [ varvals[v] if v in varvals else v for v in self.predicate ]
        return tuple(g)

class Scope():
    def __init__(self):
        self.atoms = []
        self.negatoms = []

    def addatom(self, atom):
        self.atoms.append(atom)
    def addnegatom(self, atom):
        self.negatoms.append(atom)


class Obj():
    def __init__(self):
        self.variable_list = {}

class Operator():
    """Represents and action. Can be grounded or ungrounded.
    Ungrounded operators have a '?' in names (unbound variables).
    Attributes:

        operator_name -- the name of operator (action in the domain.)
        variable_list -- a dictionary of key-value pairs where the key
                         is the variable name (with the '?') and the
                         value is the value of it when the operator is
                         grounded.
        precondition_pos -- a set of atoms corresponding to the
                            positive preconditions.
        precondition_neg -- a set of atoms corresponding to the
                            negative preconditions.
        effect_pos -- a set of atoms to add.
        effect_neg -- a set of atoms to delete.
    """
    def __init__(self, name):
        self.operator_name = name
        self.variable_list = {}
        self.precondition_pos = set()
        self.precondition_neg = set()
        self.effect_pos = set()
        self.effect_neg = set()

    def __str__(self):
        templ = "Operator Name: %s\n\tVariables: %s\n\t" + \
                "Positive Preconditions: %s\n\t" + \
                "Negative Preconditions: %s\n\t" + \
                "Positive Effects: %s\n\t" + \
                "Negative Effects: %s\n"
        return templ % ( self.operator_name, self.variable_list,
                         self.precondition_pos, self.precondition_neg,
                         self.effect_pos, self.effect_neg)


class DomainListener(pddlListener):
    def __init__(self):
        self.typesdef = False
        self.objects = {}
        self.operators = {}
        self.scopes = []
        self.negativescopes = []

    def enterActionDef(self, ctx):
        opname = ctx.actionSymbol().getText()
        opvars = {}
        self.scopes.append(Operator(opname))

    def exitActionDef(self, ctx):
        action = self.scopes.pop()
        self.operators[action.operator_name] = action

    def enterPredicatesDef(self, ctx):
        self.scopes.append(Operator(None))

    def exitPredicatesDef(self, ctx):
        dummyop = self.scopes.pop()

    def enterTypesDef(self, ctx):
        self.scopes.append(Obj())

    def exitTypesDef(self, ctx):
        self.typesdef = True
        self.scopes.pop()

    def enterTypedVariableList(self, ctx):
        # print("-> tvar")
        for v in ctx.VARIABLE():
            vname = v.getText()
            self.scopes[-1].variable_list[v.getText()] = None
        for vs in ctx.singleTypeVarList():
            t = vs.r_type().getText()
            for v in vs.VARIABLE():
                vname = v.getText()
                self.scopes[-1].variable_list[vname] = t

    def enterAtomicTermFormula(self, ctx):
        # print("-> terf")
        neg = self.negativescopes[-1]
        pred = []

        for c in ctx.getChildren():
            n = c.getText()
            if n == '(' or n == ')':
                continue
            pred.append(n)
        scope = self.scopes[-1]
        if not neg:
            scope.addatom(Atom(pred))
        else:
            scope.addnegatom(Atom(pred))

    def enterPrecondition(self, ctx):
        self.scopes.append(Scope())

    def exitPrecondition(self, ctx):
        scope = self.scopes.pop()
        self.scopes[-1].precondition_pos = set( scope.atoms )
        self.scopes[-1].precondition_neg = set( scope.negatoms )

    def enterEffect(self, ctx):
        self.scopes.append(Scope())

    def exitEffect(self, ctx):
        scope = self.scopes.pop()
        self.scopes[-1].effect_pos = set( scope.atoms )
        self.scopes[-1].effect_neg = set( scope.negatoms )

    def enterGoalDesc(self, ctx):
        negscope = bool(self.negativescopes and self.negativescopes[-1])
        for c in ctx.getChildren():
            if c.getText() == 'not':
                negscope = True
                break
        self.negativescopes.append(negscope)

    def exitGoalDesc(self, ctx):
        self.negativescopes.pop()

    def enterPEffect(self, ctx):
        negscope = False
        for c in ctx.getChildren():
            if c.getText() == 'not':
                negscope = True
                break
        self.negativescopes.append(negscope)

    def exitPEffect(self, ctx):
        self.negativescopes.pop()

    def enterTypedNameList(self, ctx):
        # print("-> tnam")
        for v in ctx.name():
            vname = v.getText()
            self.scopes[-1].variable_list[v.getText()] = None
        for vs in ctx.singleTypeNameList():
            t = vs.r_type().getText()
            for v in vs.name():
                vname = v.getText()
                self.scopes[-1].variable_list[vname] = t

    def enterConstantsDef(self, ctx):
        self.scopes.append(Obj())

    def exitConstantsDef(self, ctx):
        scope = self.scopes.pop()
        self.objects = scope.variable_list

    def exitDomain(self, ctx):
        if not self.objects and not self.typesdef:
            vs = set()
            for opn, oper in self.operators.items():
                alls = oper.precondition_pos | oper.precondition_neg | oper.effect_pos | oper.effect_neg
                for a in alls:
                    for s in a.predicate:
                        if s[0] != '?':
                            vs.add( (s, None) )
            self.objects = dict( vs)


class ProblemListener(pddlListener):

    def __init__(self):
        self.objects = {}
        self.initialstate = []
        self.goals = []
        self.scopes = []

    def enterInit(self, ctx):
        self.scopes.append(Scope())

    def exitInit(self, ctx):
        self.initialstate = set( self.scopes.pop().atoms )

    def enterGoal(self, ctx):
        self.scopes.append(Scope())

    def exitGoal(self, ctx):
        self.goals = set( self.scopes.pop().atoms )

    def enterAtomicNameFormula(self, ctx):
        pred = []
        for c in ctx.getChildren():
            n = c.getText()

            if n == '(' or n == ')':
                continue
            pred.append(n)
        scope = self.scopes[-1]
        scope.addatom(Atom(pred))

    def enterAtomicTermFormula(self, ctx):
        # with a NOT!
        pred = []
        for c in ctx.getChildren():
            n = c.getText()
            if n == '(' or n == ')':
                continue
            pred.append(n)
        scope = self.scopes[-1]
        scope.addatom(Atom(pred))

    def enterTypedNameList(self, ctx):
        for v in ctx.name():
            vname = v.getText()
            self.scopes[-1].variable_list[v.getText()] = None
        for vs in ctx.singleTypeNameList():
            t = vs.r_type().getText()
            for v in vs.name():
                vname = v.getText()
                self.scopes[-1].variable_list[vname] = t

    def enterObjectDecl(self, ctx):
        self.scopes.append(Obj())

    def exitObjectDecl(self, ctx):
        scope = self.scopes.pop()
        self.objects = scope.variable_list

    def exitProblem(self, ctx):
        if not self.objects:
            vs = set()
            for a in self.initialstate:
                for s in a.predicate:
                    vs.add( (s, None) )
            for a in self.goals:
                for s in a.predicate:
                    vs.add( (s, None) )
            self.objects = dict( vs )

class VisitorEvaluator(pddlVisitor):
    def __init__(self, actions):
        self.actions = actions

    def visitProbabilityEffect(self, ctx, combinedEffects, goal ):
        print("probability", ctx.getText())
        prob = self.visitPROB(ctx.probability())
        effect = self.visitCEffect(ctx.cEffect(), combinedEffects, goal, prob=prob)
        print(effect)
        return effect

    def visitProbEffect(self, ctx, combinedEffects, goal):
        prob = 0
        probValueList = []
        probEffectList = []
        probtfList = []
        effect = None
        #here, we want to know all of the probabilistic effects (there may be more than 1)
        for probEffectIndex in range (len(ctx.probabilityEffect())):
            effect = self.visitProbabilityEffect(ctx.probabilityEffect()[probEffectIndex], combinedEffects, goal)
            '''
            prob += probValue
            probValueList.append(probValue)
            probEffectList.append(probEffect)
            probtfList.append(True)
            '''
        print("total prob:", prob)
        '''
        if prob < 1.0:
            #handle the null action's prob
            nullProb = 1.0 - prob
            probValueList.append(nullProb)
            probEffectList.append('null')
            probtfList.append(True)
        '''
        return effect


    #may want to start in action (one level above, get name, find associated matrix)
    #otherwise this is start point
    def visitActionDefBody(self,ctx):
        #for effect in actiondefbody
        print("actiondef", ctx.getText())
        effect = Effects()
        self.visitEffect(ctx.effect(), effect)

    def combineEffects ( self, effect1: Effects, combinedEffects : Effects )-> Effects:
        newCombinedEffects = Effects()

        if combinedEffects.effectPatternList != []:
            for combination in itertools.product(effect1.effectPatternList, combinedEffects.effectPatternList):
                newCombinedEffects.effectPatternList.append(combination)
        else:
            newCombinedEffects.effectPatternList = effect1.effectPatternList

        if combinedEffects.probList != []:
            for combination in itertools.product(effect1.probList, combinedEffects.probList):
                newCombinedEffects.probList.append(combination)
        else:
            newCombinedEffects.probList = effect1.probList

        if combinedEffects.valueList != []:
            for combination in itertools.product(effect1.valueList, combinedEffects.valueList):
                newCombinedEffects.valueList.append(combination)
        else:
            newCombinedEffects.valueList = effect1.valueList

        if combinedEffects.goalPatternList != []:
            for combination in itertools.product(effect1.goalPatternList, combinedEffects.goalPatternList):
                newCombinedEffects.goalPatternList.append(combination)
        else:
            newCombinedEffects.goalPatternList = effect1.goalPatternList

        return newCombinedEffects
    '''
    effect
	    : '(' 'and' cEffect* ')' 
	    | cEffect
	    ;
    '''
    def visitEffect(self,ctx,combinedEffects):
        #within here we want a loop going over all of the effects
        #and indicates a list of effects
        #if and then do a forall
        #should return a single combinedEffects object

        listOfEffects = []
        if '(and' in ctx.getText():
            for effectIndex in range(len(ctx.cEffect())):
                print("effect")
                effect = self.visitCEffect(ctx.cEffect()[effectIndex], combinedEffects)
                effect.printInfo()
                listOfEffects.append(effect)
        else:
            self.visitCEffect(ctx.cEffect(), combinedEffects)
        newCombinedEffects = Effects()
        '''
        for combination in itertools.product(*list(effect.effectList for effect in combinedEffects.effectPatternList)):
            newCombinedEffects.effectPatternList.append(combination)

        for combination in itertools.product(*list(effect.probList for effect in combinedEffects.probList)):
            newCombinedEffects.probList.append(combination)

        for combination in itertools.product(*list(effect.effectList for effect in combinedEffects.goalPatternList)):
            newCombinedEffects.goalPatternList.append(combination)

        for combination in itertools.product(*list(effect.probList for effect in combinedEffects.valueList)):
            newCombinedEffects.valueList.append(combination)
        '''
        #get individual effect vectors for each combo of effects
        #get indiv prob for each effect vector (should be same length as number of individual effect vectors
        '''for probIndex in range(len(probEffectList)):
                targetPatternVector = np.full(self.actions.getNumPredicates(), -1)
                #must alter target pattern vector each time
                #have this as one (true for now)
                predIndex = self.actions.getPredicateIndex(probEffectList[probIndex])
                if predIndex != -1:
                    targetPatternVector[predIndex] = tfvalueList[probIndex]
                print(probValueList[probIndex])
                self.actions.alterActionMatrix("dunk-package", goalPatternVector, targetPatternVector, probValueList[probIndex])
        #if no and then just handle one effect
        '''
        return newCombinedEffects

    def visitPEffect(self,ctx):
        return ctx.getText()

    '''
    cEffect
        : '(' 'forall' '(' typedVariableList ')' effect ')'
        | '(' 'when' goalDesc condEffect ')'
        | pEffect
        | probEffect
        ;
    '''
    def visitCEffect(self,ctx, combinedEffects, goalPatternVector = None, prob=1):
        if goalPatternVector is None:
            goalPatternVector = np.full(self.actions.getNumPredicates(), -1)
        #if this is a when clause meaning it has an if condition
        condEffect = Effects()
        if ctx.goalDesc() is not None:
            goalIndex, value = self.visitGoalDesc(ctx.goalDesc())
            goalPatternVector[goalIndex] = value
            condEffect = self.visitCondEffect(ctx.condEffect(),combinedEffects, goalPatternVector)
            return condEffect

        #else if it is an effect that will happen regardless then no goal desc at all
        #else just consider as a regular 100% will happen effect
        if ctx.pEffect() is not None:
            return self.visitPEffect(ctx.pEffect(), combinedEffects, goalPatternVector, prob)

        #if ctx.effect() is not None:
        #    return self.visitEffect(ctx.effect())

        if ctx.probEffect() is not None:

            probEffect = self.visitProbEffect(ctx.probEffect(), combinedEffects, goalPatternVector)
            #create probEffect
            #probEffect = Effect(probEffectList, tfvalueList, probValueList)
            return probEffect
            '''for probIndex in range(len(probEffectList)):
                targetPatternVector = np.full(self.actions.getNumPredicates(), -1)
                #must alter target pattern vector each time
                #have this as one (true for now)
                predIndex = self.actions.getPredicateIndex(probEffectList[probIndex])
                if predIndex != -1:
                    targetPatternVector[predIndex] = tfvalueList[probIndex]
                print(probValueList[probIndex])
                self.actions.alterActionMatrix("dunk-package", goalPatternVector, targetPatternVector, probValueList[probIndex])
            '''

    def visitPredicate(self, ctx):
        return ctx.getText()

    def visitTerm(self,ctx):
        if ctx.getText() == None:
            return ""
        return ctx.getText()

    def visitAtomicTermFormula(self,ctx):
        if ctx.term() == []:
            term = ""
        else:
            term = self.visitTerm(ctx.term()[0])
        pred = self.visitPredicate(ctx.predicate())
        return pred + term

    def visitGoalDesc(self,ctx):
        #account for all objects later

        text = self.visitAtomicTermFormula(ctx.atomicTermFormula())
        print("goaldesc", text)
        predIndex = self.actions.getPredicateIndex(text)
        #-1 as a fill value for all values that are allowed to be wildcards
        goalStateEncode = np.full(self.actions.getNumPredicates(), -1)

        #if goal desc has a not, this should return a false!
        if '(not' in ctx.getText():
            value = False
        else:
            value = True
        return predIndex, value

    #might have to be changed later on for probability
    def visitCondEffect(self,ctx, combinedEffects, condition):
        return self.visitPEffect(ctx.pEffect()[0], combinedEffects, condition, 1)

    def visitPROB(self,ctx):
        return float(ctx.getText())

    '''
    pEffect
        : '(' assignOp fHead fExp ')'
        | '(' 'not' atomicTermFormula ')'
        | atomicTermFormula
        ;
    '''
    #apply to matrices here at the lowest level so that when unrolling can remove the effects
    #how to adjust for (not (negatives?
    def visitPEffect(self,ctx, combinedEffects, condition, prob):
        #check if pEffect is true/false
        peffect = self.visitAtomicTermFormula(ctx.atomicTermFormula())
        print("peffect", peffect)
        peffectObj = Effects()
        #take current combined effects list and
        peffectObj.goalPatternList.append(condition)
        peffectObj.effectPatternList.append(ctx.getText())
        peffectObj.valueList.append(1)
        peffectObj.probList.append(prob)
        #no condition for this effect
        combinedEffects = self.combineEffects(peffectObj, combinedEffects)
        #combinedEffects.goalPatternList.append(condition)
        #combinedEffects.effectPatternList.append(ctx.getText())
        #combinedEffects.valueList.append(1)
        #combinedEffects.probList.append(1.0)

        #temp true for now
        return combinedEffects

class DomainProblem():

    def __init__(self, domainfile, problemfile):
        """Parses a PDDL domain and problem files and
        returns an object representing them.

        domainfile -- path for the PDDL domain file
        problemfile -- path for the PDDL problem file
        """
        # domain
        inp = FileStream(domainfile)
        lexer = pddlLexer(inp)
        stream = CommonTokenStream(lexer)
        parser = pddlParser(stream)
        self.tree = parser.domain()
        self.domain = DomainListener()
        walker = ParseTreeWalker()
        walker.walk(self.domain, self.tree)
        # problem
        inp = FileStream(problemfile)
        lexer = pddlLexer(inp)
        stream = CommonTokenStream(lexer)
        parser = pddlParser(stream)
        tree = parser.problem()
        self.problem = ProblemListener()
        walker = ParseTreeWalker()
        walker.walk(self.problem, tree)
        actions = self.createActions()
        self.visitor = VisitorEvaluator(actions)
        # variable ground space for each operator.
        # a dict where keys are op names and values
        # a dict where keys are var names and values
        # a list of possible symbols.
        self.vargroundspace = {}

    def operators(self):
        """Returns an iterator of the names of the actions defined in
        the domain file.
        """
        return self.domain.operators.keys()

    def ground_operator(self, op_name):
        """Returns an interator of Operator instances. Each item of the iterator
        is a grounded instance.

        returns -- An iterator of Operator instances.
        """
        op = self.domain.operators[op_name]
        self._set_operator_groundspace( op_name, op.variable_list.items() )
        for ground in self._instantiate( op_name ):
            # print('grounded', ground)
            st = dict(ground)
            gop = Operator(op_name)
            gop.variable_list = st
            gop.precondition_pos = set( [ a.ground( st ) for a in op.precondition_pos ] )
            gop.precondition_neg = set( [ a.ground( st ) for a in op.precondition_neg ] )
            gop.effect_pos = set( [ a.ground( st ) for a in op.effect_pos ] )
            gop.effect_neg = set( [ a.ground( st ) for a in op.effect_neg ] )
            yield gop

    def _typesymbols(self, t):
        return ( k for k,v in self.worldobjects().items() if v == t )

    def _set_operator_groundspace(self, opname, variables):
        # cache the variables ground space for each operator.
        if opname not in self.vargroundspace:
            d = self.vargroundspace.setdefault(opname, {})
            for vname, t in variables:
                for symb in self._typesymbols(t):
                    d.setdefault(vname, []).append(symb)

    def _instantiate(self, opname):
        d = self.vargroundspace[opname]
        # expands the dict to something like:
        #[ [('?x1','A'),('?x1','B')..], [('?x2','M'),('?x2','N')..],..]
        expanded = [ [ (vname, symb) for symb in d[vname] ] for vname in d ]
        # cartesian product.
        return itertools.product(*expanded)

    def initialstate(self):
        """Returns a set of atoms (tuples of strings) corresponding to the intial
        state defined in the problem file.
        """
        return self.problem.initialstate

    def goals(self):
        """Returns a set of atoms (tuples of strings) corresponding to the goals
        defined in the problem file.
        """
        return self.problem.goals

    def worldobjects(self):
        """Returns a dictionary of key value pairs where the key is the name of
        an object and the value is it's type (None in case is untyped.)
        """
        return dict( self.domain.objects.items() | self.problem.objects.items() )

    def teststuff(self):
        self.visitor.visit(self.tree)

    def createActions(self):
        self.actions = Actions()
        return self.actions

def getCombinations(n):
    lst = list(itertools.product([0, 1], repeat=n))
    return lst

if __name__ == '__main__':
    domainfile = "bomb-toilet-domain.pddl"
    problemfile = "bomb-toilet-problem.pddl"

    domprob = DomainProblem(domainfile, problemfile)
    domprob.createActions()
    #print(list(domprob.ground_operator('dunk-package'))[0])
    #print(list(domprob.ground_operator('move'))[0].effect_neg)
    domprob.teststuff()
    #print(list(domprob.ground_operator("dunk-package"))[0])
