"""
This module defines a rewrite rule.
"""

from ..ast import *
from ..runtime import Function
from ..utils import default_repr, sliding_window

from .applier import Applier
from .matcher import Matcher, ExprMatch, StmtMatch
from .pattern import Pattern, ExprPattern, StmtPattern


class _RewriteEngine(DefaultAstTransformVisitor):
    """Rewrite rule applier for a given rewrite rule."""

    matcher: Matcher
    """rewrite rule applier"""
    applier: Applier
    """rewrite rule applier"""

    times_matched: int
    """number of times the rewrite rule was matched"""
    times_applied: int
    """number of times the rewrite rule was applied"""

    def __init__(self, lhs: Pattern, rhs: Pattern):
        self.matcher = Matcher(lhs)
        self.applier = Applier(rhs)
        self.times_matched = 0
        self.times_applied = 0

    def apply(self, func: FuncDef, occurence: int | None = None):
        # reset counters
        self.times_matched = 0
        self.times_applied = 0
        # apply the rewrite rule
        ast = self._visit_function(func, occurence)
        return ast, self.times_matched, self.times_applied

    def _visit_expr(self, e: Expr, occurence: int | None):
        e = super()._visit_expr(e, occurence)
        if isinstance(self.matcher.pattern, ExprPattern):
            # check if rewrite applies here
            pmatch = self.matcher.match_exact(e)
            if pmatch:
                if not isinstance(pmatch, ExprMatch):
                    raise TypeError(f'Matcher produced \'ExprMatch\', got {type(pmatch)} for {pmatch}')
                if occurence is None or self.times_matched == occurence:
                    e = self.applier.apply(pmatch)
                    if not isinstance(e, Expr):
                        raise TypeError(f'Substitution produced \'Expr\', got {type(e)} for {e}')
                    self.times_applied += 1
                self.times_matched += 1
        return e

    def _visit_block(self, block: StmtBlock, occurence: int | None):
        pattern = self.matcher.pattern
        block, _ = super()._visit_block(block, occurence)
        if isinstance(pattern, StmtPattern):
            # check if rewrite applies here
            pattern_size = len(pattern.block.stmts)
            iterator = sliding_window(block.stmts, pattern_size)
            new_block = StmtBlock([])
            try:
                # termination guaranteed by finitely-sized iterator
                while True:
                    stmts = next(iterator)
                    pmatch = self.matcher.match_exact(StmtBlock(stmts))
                    if pmatch:
                        if not isinstance(pmatch, StmtMatch):
                            raise TypeError(f'Matcher produced \'StmtMatch\', got {type(pmatch)} for {pmatch}')
                        if occurence is None or self.times_matched == occurence:
                            # apply the substitution
                            rw = self.applier.apply(pmatch)
                            if not isinstance(rw, StmtBlock):
                                raise TypeError(f'Substitution produced \'StmtBlock\', got {type(rw)} for {rw}')
                            new_block.stmts.extend(rw.stmts)
                            self.times_applied += 1
                            # skip rest of the block
                            for _ in range(pattern_size - 1):
                                next(iterator)
                        else:
                            self.times_matched += 1
                    else:
                        # rewrite does not apply
                        new_block.stmts.append(stmts[0])
            except StopIteration:
                # end of the block to check
                # we are missing the last N - 1 statements
                # where N is the size of the pattern
                if pattern_size > 1:
                    # sanity check
                    end_stmts = block.stmts[-(pattern_size - 1):]
                    assert len(end_stmts) == pattern_size - 1
                    # add the last N - 1 statements
                    new_block.stmts.extend(block.stmts[-(pattern_size - 1):])
            return new_block, occurence
        else:
            # pattern does not apply
            return block, occurence


class RewriteError(Exception):
    """Exception raised when a rewrite rule fails to apply."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@default_repr
class Rewrite:
    """A rewrite rule from L to R."""

    lhs: Pattern
    """the matching side of the rewrite rules"""

    rhs: Pattern
    """the substitution side of the rewrite rule"""

    name: str | None
    """the name of the rewrite rule"""

    _engine: _RewriteEngine
    """underlying rewrite rule applier"""

    def __init__(self, lhs: Pattern, rhs: Pattern, *, name: str | None = None):
        """
        Initialize a rewrite rule.

        Args:
            lhs (Pattern): the matching side of the rewrite rule.
            rhs (Pattern): the substitution side of the rewrite rule.
        """
        if type(lhs) is not type(rhs):
            raise ValueError(f'patterns must be of the same type: {lhs} => {rhs}')

        self.lhs = lhs
        self.rhs = rhs
        self.name = name
        self._engine = _RewriteEngine(lhs, rhs)

    def apply(self, func: Function, occurence: int = 0):
        """
        Applies the rewrite rule to the given pattern.
        Optionally, specify which match occurence, in traversal order,
        to apply the rewrite rule to. By default, the first match is used.

        Raises `ValueError` if the rewrite rule does not apply.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)}')
        if not isinstance(occurence, int):
            raise TypeError(f'Expected \'int\', got {type(occurence)}')
        if occurence < 0:
            raise ValueError(f'Expected non-negative integer, got {occurence}')

        ast, _, times_applied = self._engine.apply(func.ast, occurence)
        if times_applied == 0:
            raise RewriteError(f'could not apply rewrite rule: {self.lhs.format()} => {self.rhs.format()}')
        return func.with_ast(ast)


    def apply_all(self, func: Function):
        """
        Applies the rewrite rule to all matching patterns in the given function.

        Raises `ValueError` if the rewrite rule does not apply.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)}')

        ast, _, times_applied = self._engine.apply(func.ast)
        if times_applied == 0:
            raise RewriteError(f'could not apply rewrite rule: {self.lhs} => {self.rhs}')
        return func.with_ast(ast)
