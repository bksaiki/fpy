"""
This module defines subsitution for FPy AST.
"""

from ..ast import *
from ..utils import Gensym

from .matcher import LocatedMatch, ExprMatch, StmtMatch
from .pattern import Pattern, ExprPattern, StmtPattern
from .subst import Subst


class SubstitutionError(Exception):
    """
    Exception raised when a substitution fails.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'SubstitutionError: {self.message}'

class _ExprApplierInst(DefaultAstTransformVisitor):
    """
    FPy pattern match applier instance for expressions.

    Takes a pattern and a substitution and applies the substitution
    to produce a program (or program fragment).
    """

    pattern: ExprPattern
    pmatch: ExprMatch

    def __init__(self, pattern: ExprPattern, pmatch: ExprMatch):
        self.pattern = pattern
        self.pmatch = pmatch

    def run(self):
        return self._visit_expr(self.pattern.expr, dict())

    def _visit_var(self, e: Var, ctx: None):
        if e.name in self.pmatch.subst:
            # lookup name in the substitution
            return self.pmatch.subst[e.name]
        else:
            # otherwise, name is an inserted variable
            return super()._visit_var(e, ctx)


class _StmtApplierInst(DefaultAstTransformVisitor):
    """
    FPy pattern match applier instance for statements.

    Takes a pattern and a substitution and applies the substitution
    to produce a program (or program fragment).
    """

    pattern: StmtPattern
    pmatch: StmtMatch
    free: dict[NamedId, NamedId]

    def __init__(self, pattern: StmtPattern, pmatch: StmtMatch):
        self.pattern = pattern
        self.pmatch = pmatch
        self.free = dict()
        for pvar in pattern.vars() - pmatch.subst.vars():
            # TODO: generate a fresh identifier
            self.free[pvar] = pvar

    def run(self):
        # apply substitution
        return self._visit_block(self.pattern.block, dict())

    def _visit_target(self, name: NamedId):
        if name in self.pmatch.subst:
            # name in the substitution
            e = self.pmatch.subst[name]
            if not isinstance(e, Var):
                raise TypeError(f'Expected \'Var\', got {type(e)} for {e}')
            return e.name
        else:
            # variable if free in the pattern
            return self.free[name]

    def _visit_id(self, ident: Id):
        match ident:
            case NamedId():
                return self._visit_target(ident)
            case _:
                return ident

    def _visit_var(self, e: Var, ctx: None):
        if e.name in self.pmatch.subst:
            # name in the substitution
            return self.pmatch.subst[e.name]
        else:
            # variable is free in the pattern
            return Var(self.free[e.name], None)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: None):
        ident = self._visit_id(stmt.var)
        expr = self._visit_expr(stmt.expr, None)
        return SimpleAssign(ident, expr, stmt.ann, stmt.loc)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        raise NotImplementedError(stmt)

    def _visit_return(self, stmt, ctx):
        raise NotImplementedError(stmt)


class Applier:
    """
    FPy subsitution applier.

    Takes a pattern and a substitution and applies the substitution
    to produce a program (or program fragment).
    """

    pattern: Pattern

    def __init__(self, pattern: Pattern):
        if not isinstance(pattern, Pattern):
            raise TypeError(f'Expected \'Pattern\', got {type(pattern)}')
        self.pattern = pattern

    def apply(self, pmatch: LocatedMatch):
        """
        Applies the substitution to the pattern.
        The result is always a valid IR fragment (including locally SSA).
        """
        # check that type of pattern matches type of pattern match
        match self.pattern:
            case ExprPattern():
                if not isinstance(pmatch, ExprMatch):
                    raise TypeError(f'Expected \'ExprMatch\', got {type(pmatch)}')
                self._check_valid_subst(pmatch.subst)
                return _ExprApplierInst(self.pattern, pmatch).run()
            case StmtPattern():
                if not isinstance(pmatch, StmtMatch):
                    raise TypeError(f'Expected \'StmtMatch\', got {type(pmatch)}')
                self._check_valid_subst(pmatch.subst)
                return _StmtApplierInst(self.pattern, pmatch).run()
            case _:
                raise RuntimeError(f'unreachable case: {self.pattern}')

    def _check_valid_subst(self, subst: Subst):
        """Checks that the substitution is valid."""
        for pvar in self.pattern.vars():
            if pvar not in subst:
                raise SubstitutionError(f'variable \'{pvar}\' not in substitution {subst}')
