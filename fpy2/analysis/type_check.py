"""Type checking for FPy programs."""

from ..ast import *
from .define_use import DefineUse, DefineUseAnalysis, Definition

_TCtx = dict[NamedId, TypeAnn]

class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""
    func: FuncDef
    def_use: DefineUseAnalysis
    types: dict[Definition, TypeAnn]

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        super().__init__()
        self.func = func
        self.def_use = def_use
        self.types = {}

    def analyze(self):
        return self._visit_function(self.func, {})

    def _visit_var(self, e: Var, tctx: _TCtx):
        raise NotImplementedError

    def _visit_bool(self, e: BoolVal, tctx: _TCtx):
        raise NotImplementedError

    def _visit_foreign(self, e: ForeignVal, tctx: _TCtx):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, tctx: _TCtx):
        raise NotImplementedError

    def _visit_hexnum(self, e: Hexnum, tctx: _TCtx):
        raise NotImplementedError

    def _visit_integer(self, e: Integer, tctx: _TCtx):
        raise NotImplementedError

    def _visit_rational(self, e: Rational, tctx: _TCtx):
        raise NotImplementedError

    def _visit_digits(self, e: Digits, tctx: _TCtx):
        raise NotImplementedError

    def _visit_constant(self, e: Constant, tctx: _TCtx):
        raise NotImplementedError

    def _visit_unaryop(self, e: UnaryOp, tctx: _TCtx):
        raise NotImplementedError

    def _visit_binaryop(self, e: BinaryOp, tctx: _TCtx):
        raise NotImplementedError

    def _visit_ternaryop(self, e: TernaryOp, tctx: _TCtx):
        raise NotImplementedError

    def _visit_naryop(self, e: NaryOp, tctx: _TCtx):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, tctx: _TCtx):
        raise NotImplementedError

    def _visit_call(self, e: Call, tctx: _TCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, tctx: _TCtx):
        raise NotImplementedError

    def _visit_comp_expr(self, e: CompExpr, tctx: _TCtx):
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, tctx: _TCtx):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, tctx: _TCtx):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, tctx: _TCtx):
        raise NotImplementedError

    def _visit_context_expr(self, e: ContextExpr, tctx: _TCtx):
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, tctx: _TCtx):
        raise NotImplementedError

    def _visit_indexed_assign(self, stmt: IndexedAssign, tctx: _TCtx):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_while(self, stmt: WhileStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_return(self, stmt: ReturnStmt, tctx: _TCtx):
        raise NotImplementedError

    def _visit_block(self, block: StmtBlock, tctx: _TCtx):
        raise NotImplementedError

    def _visit_function(self, func: FuncDef, tctx: _TCtx):


        raise NotImplementedError


class TypeCheck:
    """
    Type checker for the FPy language.

    Unlike Python, FPy is statically typed.

    When the `@fpy` decorator runs, it also type checks the function
    and raises an error if the function is not well-typed.
    """

    @staticmethod
    def check(func):
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        return _TypeCheckInstance(func, def_use).analyze()

