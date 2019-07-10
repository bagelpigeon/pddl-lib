# Generated from pddl.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .pddlParser import pddlParser
else:
    from pddlParser import pddlParser

# This class defines a complete generic visitor for a parse tree produced by pddlParser.

class pddlVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by pddlParser#pddlDoc.
    def visitPddlDoc(self, ctx:pddlParser.PddlDocContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#domain.
    def visitDomain(self, ctx:pddlParser.DomainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#domainName.
    def visitDomainName(self, ctx:pddlParser.DomainNameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#requireDef.
    def visitRequireDef(self, ctx:pddlParser.RequireDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#typesDef.
    def visitTypesDef(self, ctx:pddlParser.TypesDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#typedNameList.
    def visitTypedNameList(self, ctx:pddlParser.TypedNameListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#singleTypeNameList.
    def visitSingleTypeNameList(self, ctx:pddlParser.SingleTypeNameListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#r_type.
    def visitR_type(self, ctx:pddlParser.R_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#primType.
    def visitPrimType(self, ctx:pddlParser.PrimTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#functionsDef.
    def visitFunctionsDef(self, ctx:pddlParser.FunctionsDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#functionList.
    def visitFunctionList(self, ctx:pddlParser.FunctionListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#atomicFunctionSkeleton.
    def visitAtomicFunctionSkeleton(self, ctx:pddlParser.AtomicFunctionSkeletonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#functionSymbol.
    def visitFunctionSymbol(self, ctx:pddlParser.FunctionSymbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#functionType.
    def visitFunctionType(self, ctx:pddlParser.FunctionTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#constantsDef.
    def visitConstantsDef(self, ctx:pddlParser.ConstantsDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#predicatesDef.
    def visitPredicatesDef(self, ctx:pddlParser.PredicatesDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#atomicFormulaSkeleton.
    def visitAtomicFormulaSkeleton(self, ctx:pddlParser.AtomicFormulaSkeletonContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#predicate.
    def visitPredicate(self, ctx:pddlParser.PredicateContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#typedVariableList.
    def visitTypedVariableList(self, ctx:pddlParser.TypedVariableListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#singleTypeVarList.
    def visitSingleTypeVarList(self, ctx:pddlParser.SingleTypeVarListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#constraints.
    def visitConstraints(self, ctx:pddlParser.ConstraintsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#structureDef.
    def visitStructureDef(self, ctx:pddlParser.StructureDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#actionDef.
    def visitActionDef(self, ctx:pddlParser.ActionDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#actionSymbol.
    def visitActionSymbol(self, ctx:pddlParser.ActionSymbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#actionDefBody.
    def visitActionDefBody(self, ctx:pddlParser.ActionDefBodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#precondition.
    def visitPrecondition(self, ctx:pddlParser.PreconditionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#goalDesc.
    def visitGoalDesc(self, ctx:pddlParser.GoalDescContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fComp.
    def visitFComp(self, ctx:pddlParser.FCompContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#atomicTermFormula.
    def visitAtomicTermFormula(self, ctx:pddlParser.AtomicTermFormulaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#term.
    def visitTerm(self, ctx:pddlParser.TermContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#durativeActionDef.
    def visitDurativeActionDef(self, ctx:pddlParser.DurativeActionDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#daDefBody.
    def visitDaDefBody(self, ctx:pddlParser.DaDefBodyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#daGD.
    def visitDaGD(self, ctx:pddlParser.DaGDContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#prefTimedGD.
    def visitPrefTimedGD(self, ctx:pddlParser.PrefTimedGDContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#timedGD.
    def visitTimedGD(self, ctx:pddlParser.TimedGDContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#timeSpecifier.
    def visitTimeSpecifier(self, ctx:pddlParser.TimeSpecifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#interval.
    def visitInterval(self, ctx:pddlParser.IntervalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#derivedDef.
    def visitDerivedDef(self, ctx:pddlParser.DerivedDefContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fExp.
    def visitFExp(self, ctx:pddlParser.FExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fExp2.
    def visitFExp2(self, ctx:pddlParser.FExp2Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fHead.
    def visitFHead(self, ctx:pddlParser.FHeadContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#effect.
    def visitEffect(self, ctx:pddlParser.EffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#cEffect.
    def visitCEffect(self, ctx:pddlParser.CEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#pEffect.
    def visitPEffect(self, ctx:pddlParser.PEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#probEffect.
    def visitProbEffect(self, ctx:pddlParser.ProbEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#probabilityEffect.
    def visitProbabilityEffect(self, ctx:pddlParser.ProbabilityEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#PROB.
    def visitPROB(self, ctx:pddlParser.PROBContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#condEffect.
    def visitCondEffect(self, ctx:pddlParser.CondEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#binaryOp.
    def visitBinaryOp(self, ctx:pddlParser.BinaryOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#binaryComp.
    def visitBinaryComp(self, ctx:pddlParser.BinaryCompContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#assignOp.
    def visitAssignOp(self, ctx:pddlParser.AssignOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#durationConstraint.
    def visitDurationConstraint(self, ctx:pddlParser.DurationConstraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#simpleDurationConstraint.
    def visitSimpleDurationConstraint(self, ctx:pddlParser.SimpleDurationConstraintContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#durOp.
    def visitDurOp(self, ctx:pddlParser.DurOpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#durValue.
    def visitDurValue(self, ctx:pddlParser.DurValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#daEffect.
    def visitDaEffect(self, ctx:pddlParser.DaEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#timedEffect.
    def visitTimedEffect(self, ctx:pddlParser.TimedEffectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fAssignDA.
    def visitFAssignDA(self, ctx:pddlParser.FAssignDAContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#fExpDA.
    def visitFExpDA(self, ctx:pddlParser.FExpDAContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#assignOpT.
    def visitAssignOpT(self, ctx:pddlParser.AssignOpTContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#problem.
    def visitProblem(self, ctx:pddlParser.ProblemContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#problemDecl.
    def visitProblemDecl(self, ctx:pddlParser.ProblemDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#problemDomain.
    def visitProblemDomain(self, ctx:pddlParser.ProblemDomainContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#objectDecl.
    def visitObjectDecl(self, ctx:pddlParser.ObjectDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#init.
    def visitInit(self, ctx:pddlParser.InitContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#initEl.
    def visitInitEl(self, ctx:pddlParser.InitElContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#probInitEl.
    def visitProbInitEl(self, ctx:pddlParser.ProbInitElContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#nameLiteral.
    def visitNameLiteral(self, ctx:pddlParser.NameLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#atomicNameFormula.
    def visitAtomicNameFormula(self, ctx:pddlParser.AtomicNameFormulaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#goal.
    def visitGoal(self, ctx:pddlParser.GoalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#probConstraints.
    def visitProbConstraints(self, ctx:pddlParser.ProbConstraintsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#prefConGD.
    def visitPrefConGD(self, ctx:pddlParser.PrefConGDContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#metricSpec.
    def visitMetricSpec(self, ctx:pddlParser.MetricSpecContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#optimization.
    def visitOptimization(self, ctx:pddlParser.OptimizationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#metricFExp.
    def visitMetricFExp(self, ctx:pddlParser.MetricFExpContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#conGD.
    def visitConGD(self, ctx:pddlParser.ConGDContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by pddlParser#name.
    def visitName(self, ctx:pddlParser.NameContext):
        return self.visitChildren(ctx)



del pddlParser