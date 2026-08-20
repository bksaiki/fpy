"""
This module defines subsitution for FPy AST.
"""

from ..ast import *
from ..transform import clone
from .pattern import ExprPattern, Pattern, StmtPattern
from .subst import Subst


class SubstitutionError(Exception):
    """A rule whose replacement names a variable its pattern never bound.

    Outside the :class:`TransformError` hierarchy: a malformed *rule*, not a bad
    reference into a program, so a try/fallback schedule must not swallow it.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f'SubstitutionError: {self.message}'

class _ExprApplierInst(DefaultTransformVisitor):
    """
    FPy pattern match applier instance for expressions.

    Takes a pattern and a substitution and applies the substitution
    to produce a program (or program fragment).
    """

    pattern: ExprPattern
    subst: Subst

    def __init__(self, pattern: ExprPattern, subst: Subst):
        self.pattern = pattern
        self.subst = subst

    def run(self):
        return self._visit_expr(self.pattern.expr, {})

    def _visit_var(self, e: Var, ctx: None):
        if e.name in self.subst:
            # a replacement may use a variable twice, and two program points
            # must not share AST nodes
            return clone(self.subst[e.name])
        else:
            # otherwise, name is an inserted variable
            return super()._visit_var(e, ctx)


class _StmtApplierInst(DefaultTransformVisitor):
    """
    FPy pattern match applier instance for statements.

    Takes a pattern and a substitution and applies the substitution
    to produce a program (or program fragment).
    """

    pattern: StmtPattern
    subst: Subst
    free: dict[NamedId, NamedId]

    def __init__(self, pattern: StmtPattern, subst: Subst):
        self.pattern = pattern
        self.subst = subst
        self.free = {}
        for pvar in pattern.vars() - subst.vars():
            # TODO: generate a fresh identifier
            self.free[pvar] = pvar

    def run(self):
        # apply substitution
        block, _ = self._visit_block(self.pattern.block, None)
        return block

    def _visit_target(self, name: NamedId):
        if name in self.subst:
            # name in the substitution
            e = self.subst[name]
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
        if e.name in self.subst:
            return clone(self.subst[e.name])
        else:
            # variable is free in the pattern
            return Var(self.free[e.name], None)

    def _visit_list_comp(self, e: ListComp, ctx: None):
        targets: list[Id | TupleBinding] = []
        for target in e.targets:
            match target:
                case Id():
                    targets.append(self._visit_id(target))
                case TupleBinding():
                    targets.append(self._visit_tuple_binding(target, ctx))
                case _:
                    raise RuntimeError(f'unreachable case: {target}')

        iterables = [self._visit_expr(e, None) for e in e.iterables]
        elt = self._visit_expr(e.elt, None)
        return ListComp(targets, iterables, elt, None)

    def _visit_binding(self, binding: Id | TupleBinding, ctx: None):
        match binding:
            case Id():
                return self._visit_id(binding)
            case TupleBinding():
                return self._visit_tuple_binding(binding, ctx)
            case _:
                raise RuntimeError(f'unreachable case: {binding}')

    def _visit_tuple_binding(self, binding: TupleBinding, ctx: None):
        return TupleBinding([binding for binding in binding.elts], None)

    def _visit_assign(self, stmt: Assign, ctx: None):
        binding = self._visit_binding(stmt.target, None)
        expr = self._visit_expr(stmt.expr, None)
        s =  Assign(binding, stmt.type, expr, None)
        return s, None

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        var = self._visit_id(stmt.var)
        slices = [self._visit_expr(s, None) for s in stmt.indices]
        expr = self._visit_expr(stmt.expr, None)
        s = IndexedAssign(var, slices, expr, None)
        return s, None

    def _visit_if(self, stmt: IfStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, None)
        ift, _ = self._visit_block(stmt.ift, None)
        if stmt.iff is None:
            s = IfStmt(cond, ift, None, None)
        else:
            iff, _ = self._visit_block(stmt.iff, None)
            s = IfStmt(cond, ift, iff, None)
        return s, None

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, None)
        body, _ = self._visit_block(stmt.body, None)
        s = WhileStmt(cond, body, None)
        return s, None

    def _visit_for(self, stmt: ForStmt, ctx: None):
        target = self._visit_binding(stmt.target, None)
        iterable = self._visit_expr(stmt.iterable, None)
        body, _ = self._visit_block(stmt.body, None)
        s = ForStmt(target, iterable, body, None)
        return s, None

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        context = self._visit_expr(stmt.ctx, None)
        body, _ = self._visit_block(stmt.body, None)
        if stmt.target is None:
            s = ContextStmt(None, context, body, None)
        else:
            name = self._visit_id(stmt.target)
            s = ContextStmt(name, context, body, None)
            return s, None

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        # TODO: message?
        test = self._visit_expr(stmt.test, None)
        s = AssertStmt(test, stmt.msg, None)
        return s, None

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        expr = self._visit_expr(stmt.expr, None)
        s = EffectStmt(expr, None)
        return s, None

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        expr = self._visit_expr(stmt.expr, None)
        s = ReturnStmt(expr, None)
        return s, None


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

    def apply(self, subst: Subst):
        """
        Applies the substitution to the pattern.
        The result is always a valid IR fragment (including locally SSA).
        """
        self._check_valid_subst(subst)
        match self.pattern:
            case ExprPattern():
                return _ExprApplierInst(self.pattern, subst).run()
            case StmtPattern():
                return _StmtApplierInst(self.pattern, subst).run()
            case _:
                raise RuntimeError(f'unreachable case: {self.pattern}')

    def _check_valid_subst(self, subst: Subst):
        """Checks that the substitution is valid."""
        for pvar in self.pattern.vars():
            if pvar not in subst:
                raise SubstitutionError(f'variable \'{pvar}\' not in substitution {subst}')
