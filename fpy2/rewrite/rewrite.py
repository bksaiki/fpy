"""
This module defines a rewrite rule.
"""

from ..ast import *
from ..function import Function
from ..utils import default_repr, sliding_window

from .applier import Applier
from .matcher import Matcher, ExprMatch, StmtMatch
from .pattern import Pattern, ExprPattern, StmtPattern

@default_repr
class _RewriteContext:
    """Static options"""
    occurence: int | None
    repeat: int
    is_nested: bool
    """Counters"""
    times_matched: int

    def __init__(self, occurence: int | None, repeat: int, *, is_nested: bool = False):
        self.occurence = occurence
        self.repeat = repeat
        self.is_nested = is_nested
        self.times_matched = 0

    @staticmethod
    def default() -> '_RewriteContext':
        return _RewriteContext(occurence=0, repeat=1)



class _RewriteEngine(DefaultTransformVisitor):
    """Rewrite rule applier for a given rewrite rule."""

    matcher: Matcher
    """rewrite rule applier"""
    applier: Applier
    """rewrite rule applier"""

    times_applied: int
    """number of times the rewrite rule was applied"""

    def __init__(self, lhs: Pattern, rhs: Pattern):
        self.matcher = Matcher(lhs)
        self.applier = Applier(rhs)

        self.times_applied = 0

    def apply(
        self,
        func: FuncDef, *,
        occurence: int | None = None,
        repeat: int = 1
    ):
        # reset counters
        self.times_applied = 0
        # apply the rewrite rule
        options = _RewriteContext(occurence, repeat)
        ast = self._visit_function(func, options)
        return ast, self.times_applied

    def _nested_applier(self, num_times: int):
        if num_times == 1:
            return self.applier
        else:
            pattern = self.applier.pattern
            for _ in range(1, num_times):
                match pattern:
                    case ExprPattern():
                        # run the matcher on the applier pattern
                        # this is a hacky way to emulate taint
                        repeat_opt = _RewriteContext(0, 1, is_nested=True)
                        expr = self._visit_expr(pattern.expr, repeat_opt)
                        # TODO: this is a bit messy
                        ast = pattern.to_ast()
                        ast.body.stmts[0] = EffectStmt(expr, None)
                        pattern = ExprPattern(ast)
                    case StmtPattern():
                        repeat_opt = _RewriteContext(0, 1, is_nested=True)
                        block, _ = self._visit_block(pattern.block, repeat_opt)
                        # TODO: this is a bit messy
                        ast = pattern.to_ast()
                        ast.body = block
                        pattern = StmtPattern(ast)
                    case _:
                        raise RuntimeError(f'unreachable case: {pattern}')
            return Applier(pattern)

    def _visit_expr(self, e: Expr, ctx: _RewriteContext):
        e = super()._visit_expr(e, ctx)
        if isinstance(self.matcher.pattern, ExprPattern):
            # check if rewrite applies here
            pmatch = self.matcher.match_exact(e)
            if pmatch:
                if not isinstance(pmatch, ExprMatch):
                    raise TypeError(f'Matcher produced \'ExprMatch\', got {type(pmatch)} for {pmatch}')
                if ctx.occurence is None or ctx.times_matched == ctx.occurence:
                    e = self.applier.apply(pmatch)
                    if not isinstance(e, Expr):
                        raise TypeError(f'Substitution produced \'Expr\', got {type(e)} for {e}')
                    self.times_applied += 1
                ctx.times_matched += 1
        return e

    def _visit_block(self, block: StmtBlock, ctx: _RewriteContext):
        pattern = self.matcher.pattern
        block, _ = super()._visit_block(block, ctx)
        if isinstance(pattern, StmtPattern):
            # check if rewrite applies here
            pattern_size = len(pattern.block.stmts)
            iterator = sliding_window(block.stmts, pattern_size)
            new_stmts: list[Stmt] = []
            try:
                # termination guaranteed by finitely-sized iterator
                while True:
                    stmts = list(next(iterator))
                    pmatch = self.matcher.match_exact(StmtBlock(stmts))
                    if pmatch:
                        if not isinstance(pmatch, StmtMatch):
                            raise TypeError(f'Matcher produced \'StmtMatch\', got {type(pmatch)} for {pmatch}')
                        if ctx.occurence is None or ctx.times_matched == ctx.occurence:
                            # apply the substitution
                            applier = self._nested_applier(1 if ctx.is_nested else ctx.repeat)
                            rw = applier.apply(pmatch)
                            if not isinstance(rw, StmtBlock):
                                raise TypeError(f'Substitution produced \'StmtBlock\', got {type(rw)} for {rw}')
                            new_stmts.extend(rw.stmts)
                            self.times_applied += 1
                            # skip rest of the block
                            for _ in range(pattern_size - 1):
                                next(iterator)
                        ctx.times_matched += 1
                    else:
                        # rewrite does not apply
                        new_stmts.append(stmts[0])
            except StopIteration:
                # end of the block to check
                # we are missing the last N - 1 statements
                # where N is the size of the pattern
                if pattern_size > 1:
                    # sanity check
                    end_stmts = block.stmts[-(pattern_size - 1):]
                    assert len(end_stmts) == pattern_size - 1
                    # add the last N - 1 statements
                    new_stmts.extend(end_stmts)

            new_block = StmtBlock(new_stmts)
            return new_block, None
        else:
            # pattern does not apply
            return block, None


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

    def apply(self, func: Function, *, occurence: int = 0, repeat: int = 1):
        """
        Applies the rewrite rule to the given pattern.

        Optionally specify:
        - `occurence`: which match occurence, in traversal order, to rewrite
        - `repeat`: how many times to apply the rewrite rule once a match occurs

        Raises `ValueError` if the rewrite rule does not apply.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)} for {func}')
        if not isinstance(occurence, int):
            raise TypeError(f'Expected \'int\', got {type(occurence)} for {occurence}')
        if not isinstance(repeat, int):
            raise TypeError(f'Expected \'int\', got {type(repeat)} for {repeat}')
        if occurence < 0:
            raise ValueError(f'Expected non-negative integer, got {occurence}')
        if repeat < 1:
            raise ValueError(f'Expected positive integer, got {repeat}')

        ast, times_applied = self._engine.apply(func.ast, occurence=occurence, repeat=repeat)
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

        ast, times_applied = self._engine.apply(func.ast)
        if times_applied == 0:
            raise RewriteError(f'could not apply rewrite rule: {self.lhs} => {self.rhs}')
        return func.with_ast(ast)
