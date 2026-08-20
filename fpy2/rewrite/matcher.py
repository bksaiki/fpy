"""
This module defines pattern matching facilities for FPy AST.
"""

from dataclasses import dataclass
from fractions import Fraction

from ..ast import *
from ..function import Function
from ..number import Float
from ..transform import (
    BlockCursor,
    Cursor,
    ExprCursor,
    StmtCursor,
    StmtPath,
    walk_blocks,
    walk_exprs,
)
from ..utils import default_repr, sliding_window
from .pattern import ExprPattern, Pattern, StmtPattern
from .subst import Subst


def _real_eq(a: 'Fraction | Float', b: 'Fraction | Float') -> bool:
    """Signed-zero-aware equality of two exact-real literal values.

    `RationalVal.as_real` returns a `Float` only for negative zero, which is
    otherwise `==` to `+0.0` under IEEE semantics. Treat a negative-zero literal
    as equal only to another negative zero, so a rule keyed on `+0.0` does not
    match `-0.0` (which would be unsound, e.g. `x + 0.0 -> x` at `x = -0.0`)."""
    a_negzero = isinstance(a, Float)
    b_negzero = isinstance(b, Float)
    if a_negzero or b_negzero:
        return a_negzero and b_negzero
    return a == b

class _MatchFailure(Exception):
    """
    Exception raised when a match fails.
    """
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

@dataclass(frozen=True)
class Match:
    """Where a pattern matched, and what its variables bound to.

    The cursor's kind is the pattern's kind: an :class:`ExprCursor` for an
    expression pattern, a :class:`StmtCursor` or :class:`BlockCursor` for a
    statement pattern of one or several statements.
    """

    pattern: Pattern
    subst: Subst
    cursor: Cursor


class _MatcherInst(Visitor):
    """
    FPy pattern matching instance for a pattern and sub-program.

    Unlike `_MatcherEngine`, the matcher attempts to matching
    starting at the top of the program slice.
    """
    pattern: Pattern
    ast: StmtBlock | Expr
    subst: Subst

    def __init__(self, pattern: Pattern, ast: StmtBlock | Expr):
        self.pattern = pattern
        self.ast = ast
        self.subst = Subst()

    def match(self) -> Subst | None:
        """What the pattern's variables bind to here, or `None` where it does
        not match."""
        try:
            match self.pattern:
                case ExprPattern():
                    if not isinstance(self.ast, Expr):
                        raise TypeError(f'Expected \'Expr\', got {type(self.ast)} for {self.ast}')
                    self._visit_expr(self.ast, self.pattern.expr)
                case StmtPattern():
                    if not isinstance(self.ast, StmtBlock):
                        raise TypeError(f'Expected \'StmtBlock\', got {type(self.ast)} for {self.ast}')
                    self._visit_block(self.ast, self.pattern.block)
                case _:
                    raise RuntimeError(f'unreachable case: {self.pattern}')
            return self.subst
        except _MatchFailure as _:
            # match failed
            return None

    def _bind_expr(self, name: NamedId, e: Expr):
        # check if pattern is already bound
        if name in self.subst:
            # check if the current binding is the same as `e`
            bound = self.subst[name]
            if not e.is_equiv(bound):
                raise _MatchFailure(f'conflicting bindings for {name}: {bound} != {e}')
        else:
            # insert a new binding
            self.subst[name] = e

    def _visit_target(self, name: Id, pat: Id):
        """
        Visit the left-hand side of an assignment.
        The left-hand side is an `Id` while in an expression it is a `Var`.
        """
        match pat, name:
            case UnderscoreId(), _:
                # wildcard => ignore
                pass
            case NamedId(), NamedId():
                # pattern variable
                self._bind_expr(pat, Var(name, None))
            case NamedId(), UnderscoreId():
                raise NotImplementedError(pat, name)
            case _:
                raise RuntimeError('unreachable', pat, name)

    def _visit_var(self, e: Var, pat: Var):
        raise RuntimeError('do not call')

    def _visit_bool(self, e: BoolVal, pat: BoolVal):
        if e.val != pat.val:
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_foreign(self, e: ForeignVal, pat: ForeignVal):
        if e.val != pat.val:
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_decnum(self, e: Decnum, pat: Decnum):
        # semantic match, not syntactic — but signed-zero aware: `-0.0` and
        # `+0.0` are distinct (see `_real_eq` / `RationalVal.as_real`).
        if not _real_eq(e.as_real(), pat.as_real()):
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_hexnum(self, e: Hexnum, pat: Hexnum):
        # semantic match, not syntactic — signed-zero aware (see `_visit_decnum`).
        if not _real_eq(e.as_real(), pat.as_real()):
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_integer(self, e: Integer, pat: Integer):
        if e.val != pat.val:
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_rational(self, e: Rational, pat: Rational):
        # this is a semantic match, not a syntactic match!
        if e.as_rational() != pat.as_rational():
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_digits(self, e: Digits, pat: Digits):
        # this is a semantic match, not a syntactic match!
        if e.as_rational() != pat.as_rational():
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_nullaryop(self, e: NullaryOp, pat: NullaryOp):
        if type(e) is not type(pat):
            raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_unaryop(self, e: UnaryOp, pat: UnaryOp):
        if type(e) is not type(pat):
            raise _MatchFailure(f'matching {pat} against {e}')
        self._visit_expr(e.arg, pat.arg)

    def _visit_binaryop(self, e: BinaryOp, pat: BinaryOp):
        if type(e) is not type(pat):
            raise _MatchFailure(f'matching {pat} against {e}')
        self._visit_expr(e.first, pat.first)
        self._visit_expr(e.second, pat.second)

    def _visit_ternaryop(self, e: TernaryOp, pat: TernaryOp):
        if type(e) is not type(pat):
            raise _MatchFailure(f'matching {pat} against {e}')
        self._visit_expr(e.first, pat.first)
        self._visit_expr(e.second, pat.second)
        self._visit_expr(e.third, pat.third)

    def _visit_naryop(self, e: NaryOp, pat: NaryOp):
        if type(e) is not type(pat):
            raise _MatchFailure(f'matching {pat} against {e}')
        for c1, c2 in zip(e.args, pat.args):
            self._visit_expr(c1, c2)

    def _visit_call(self, e: Call, pat: Call):
        # check shape of call
        if len(e.args) != len(pat.args) or len(e.kwargs) != len(pat.kwargs):
            raise _MatchFailure(f'matching {pat} against {e}')

        # check function symbol
        match e.fn, pat.fn:
            case None, None:
                if e.func != pat.func:
                    raise _MatchFailure(f'matching {pat} against {e}')
            case _, _:
                if e.fn != pat.fn:
                    raise _MatchFailure(f'matching {pat} against {e}')

        # check arguments
        for c1, c2 in zip(e.args, pat.args):
            self._visit_expr(c1, c2)

        # check keyword arguments
        for (k1, v1), (k2, v2) in zip(e.kwargs, pat.kwargs):
            if k1 != k2:
                raise _MatchFailure(f'matching {pat} against {e}')
            self._visit_expr(v1, v2)

    def _visit_compare(self, e: Compare, pat: Compare):
        # TODO: is matching on a subset of operations valid?
        # check if the operation sequence is the same
        if len(e.args) != len(pat.args) or e.ops != pat.ops:
            raise _MatchFailure(f'matching {pat} against {e}')
        # check if args are the same
        for c1, c2 in zip(e.args, pat.args):
            self._visit_expr(c1, c2)

    def _visit_tuple_expr(self, e: TupleExpr, pat: TupleExpr):
        # check if #elements are the same
        if len(e.elts) != len(pat.elts):
            raise _MatchFailure(f'matching {pat} against {e}')
        # check if elements are the same
        for c1, c2 in zip(e.elts, pat.elts):
            self._visit_expr(c1, c2)

    def _visit_list_expr(self, e: ListExpr, pat: ListExpr):
        # check if #elements are the same
        if len(e.elts) != len(pat.elts):
            raise _MatchFailure(f'matching {pat} against {e}')
        # check if elements are the same
        for c1, c2 in zip(e.elts, pat.elts):
            self._visit_expr(c1, c2)

    def _visit_list_ref(self, e: ListRef, pat: ListRef):
        self._visit_expr(e.value, pat.value)
        self._visit_expr(e.index, pat.index)

    def _visit_list_slice(self, e: ListSlice, pat: ListSlice):
        self._visit_expr(e.value, pat.value)
        match e.start, pat.start:
            case None, None:
                pass
            case Expr(), Expr():
                self._visit_expr(e.start, pat.start)
            case _:
                raise _MatchFailure(f'matching {pat} against {e}')
        match e.stop, pat.stop:
            case None, None:
                pass
            case Expr(), Expr():
                self._visit_expr(e.stop, pat.stop)
            case _:
                raise _MatchFailure(f'matching {pat} against {e}')

    def _visit_list_comp(self, e: ListComp, pat: ListComp):
        if len(e.iterables) != len(pat.iterables):
            raise _MatchFailure(f'matching {pat} against {e}')
        for target, ptarget in zip(e.targets, pat.targets):
            match target, ptarget:
                case Id(), Id():
                    self._visit_target(target, ptarget)
                case TupleBinding(), TupleBinding():
                    self._visit_tuple_binding(target, ptarget)
                case _:
                    raise _MatchFailure(f'matching {ptarget} against {target}')
        for it, pit in zip(e.iterables, pat.iterables):
            self._visit_expr(it, pit)
        self._visit_expr(e.elt, pat.elt)

    def _visit_if_expr(self, e: IfExpr, pat: IfExpr):
        self._visit_expr(e.cond, pat.cond)
        self._visit_expr(e.ift, pat.ift)
        self._visit_expr(e.iff, pat.iff)

    def _visit_attribute(self, e: Attribute, pat: Attribute):
        if e.attr != pat.attr:
            raise _MatchFailure(f'matching {pat} against {e}')
        self._visit_expr(e.value, pat.value)

    def _visit_binding(self, binding: Id | TupleBinding, pat: Id | TupleBinding):
        match binding, pat:
            case Id(), Id():
                self._visit_target(binding, pat)
            case TupleBinding(), TupleBinding():
                self._visit_tuple_binding(binding, pat)
            case _:
                raise _MatchFailure(f'matching {pat} against {binding}')

    def _visit_assign(self, stmt: Assign, pat: Assign):
        self._visit_binding(stmt.target, pat.target)
        self._visit_expr(stmt.expr, pat.expr)

    def _visit_tuple_binding(self, binding: TupleBinding, pat: TupleBinding):
        if len(binding.elts) != len(pat.elts):
            raise _MatchFailure(f'matching {pat} against {binding}')
        for elt, p in zip(binding.elts, pat.elts):
            match elt, p:
                case UnderscoreId(), _:
                    # ignore
                    pass
                case NamedId(), Id():
                    # pattern variable
                    self._visit_target(elt, p)
                case TupleBinding(), TupleBinding():
                    # pattern variable
                    self._visit_tuple_binding(elt, p)
                case _:
                    raise _MatchFailure(f'matching {p} against {elt}')

    def _visit_indexed_assign(self, stmt: IndexedAssign, pat: IndexedAssign):
        self._visit_target(stmt.var, pat.var)
        for e, p in zip(stmt.indices, pat.indices):
            self._visit_expr(e, p)
        self._visit_expr(stmt.expr, pat.expr)

    def _visit_if1(self, stmt: If1Stmt, pat: IfStmt):
        self._visit_expr(stmt.cond, pat.cond)
        self._visit_block(stmt.body, pat.ift)

    def _visit_if(self, stmt: IfStmt, pat: IfStmt):
        self._visit_expr(stmt.cond, pat.cond)
        self._visit_block(stmt.ift, pat.ift)
        self._visit_block(stmt.iff, pat.iff)

    def _visit_while(self, stmt: WhileStmt, pat: WhileStmt):
        self._visit_expr(stmt.cond, pat.cond)
        self._visit_block(stmt.body, pat.body)

    def _visit_for(self, stmt: ForStmt, pat: ForStmt):
        self._visit_binding(stmt.target, pat.target)
        self._visit_expr(stmt.iterable, pat.iterable)
        self._visit_block(stmt.body, pat.body)

    def _visit_context(self, stmt: ContextStmt, pat: ContextStmt):
        self._visit_expr(stmt.ctx, pat.ctx)
        self._visit_block(stmt.body, pat.body)

    def _visit_assert(self, stmt: AssertStmt, pat: AssertStmt):
        # TODO: message?
        self._visit_expr(stmt.test, pat.test)

    def _visit_effect(self, stmt: EffectStmt, pat: EffectStmt):
        self._visit_expr(stmt.expr, pat.expr)

    def _visit_return(self, stmt: ReturnStmt, pat: ReturnStmt):
        self._visit_expr(stmt.expr, pat.expr)

    def _visit_pass(self, stmt: PassStmt, pat: ReturnStmt):
        pass

    def _visit_block(self, block: StmtBlock, pat: StmtBlock):
        # check length of block
        if len(block.stmts) != len(pat.stmts):
            raise _MatchFailure(f'matching {pat} against {block}')

        # check if statements are the same
        for s1, s2 in zip(block.stmts, pat.stmts):
            if type(s1) is not type(s2):
                raise _MatchFailure(f'matching {pat} against {s1}')
            self._visit_statement(s1, s2)

    def _visit_function(self, func: FuncDef, pat: FuncDef):
        raise RuntimeError('do not call')

    def _visit_expr(self, e: Expr, pat: Expr):
        match pat:
            case Var():
                # pattern variable
                self._bind_expr(pat.name, e)
            case _:
                # check if expressions are the same type
                if type(e) is not type(pat):
                    raise _MatchFailure(f'matching {type(pat)} against {type(e)}')
                return super()._visit_expr(e, pat)

    def _visit_statement(self, stmt: Stmt, pat: Stmt):
        # check if statements are the same type
        if type(stmt) is not type(pat):
            raise _MatchFailure(f'matching {type(pat)} against {type(stmt)}')
        return super()._visit_statement(stmt, pat)


def _match_exprs(pattern: ExprPattern, func: FuncDef) -> list[Match]:
    """Every expression of *func* the pattern matches, outermost first."""
    out: list[Match] = []
    for path, e in walk_exprs(func):
        subst = _MatcherInst(pattern, e).match()
        if subst is not None:
            out.append(Match(pattern, subst, ExprCursor(func, path)))
    return out


def _match_stmts(pattern: StmtPattern, func: FuncDef) -> list[Match]:
    """Every run of statements the pattern matches: a window of *k* consecutive
    statements, at every position of every block.

    Windows overlap, so two matches can name statements in common -- fine for a
    cursor, but a rewrite cannot apply to both.
    """
    k = len(pattern.block.stmts)
    out: list[Match] = []
    for path, block in walk_blocks(func):
        for i, window in enumerate(sliding_window(block.stmts, k)):
            subst = _MatcherInst(pattern, StmtBlock(list(window))).match()
            if subst is not None:
                cursor: Cursor = (
                    StmtCursor(func, StmtPath(path, i)) if k == 1
                    else BlockCursor(func, path, range(i, i + k))
                )
                out.append(Match(pattern, subst, cursor))
    return out


class Matcher:
    """
    FPy pattern matcher.

    Matches each instance of `self.pattern` for a program
    and returns the list of subsitutions for each match.
    """

    pattern: Pattern

    def __init__(self, pattern: Pattern):
        if not isinstance(pattern, Pattern):
            raise TypeError(f'Expected \'Pattern\', got {type(pattern)}')
        self.pattern = pattern

    def match(self, func: Function) -> list[Match]:
        """Every match in *func*, each with the cursor naming where it is."""
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)}')
        match self.pattern:
            case ExprPattern():
                return _match_exprs(self.pattern, func.ast)
            case StmtPattern():
                return _match_stmts(self.pattern, func.ast)
            case _:
                raise RuntimeError(f'unreachable case: {self.pattern}')

    def match_exact(self, e: StmtBlock | Expr) -> Subst | None:
        """What the pattern binds to at exactly this node, or `None`."""
        if not isinstance(e, StmtBlock | Expr):
            raise TypeError(f'Expected \'StmtBlock\' or \'Expr\', got {type(e)}')
        return _MatcherInst(self.pattern, e).match()
