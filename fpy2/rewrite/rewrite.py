"""
This module defines a rewrite rule.
"""

from itertools import combinations

from ..analysis import SyntaxCheck
from ..ast import *
from ..function import Function
from ..transform import BlockCursor, EditLog, TransformDeclined, TransformReferenceError
from ..transform.utils import SiteRewriter
from ..utils import default_repr, sliding_window
from .applier import Applier
from .find import find_all
from .matcher import Matcher
from .pattern import ExprPattern, Pattern, StmtPattern


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



class _RewriteEngine(SiteRewriter):
    """Rewrite rule applier for a given rewrite rule."""

    matcher: Matcher
    """matcher for the rule's left-hand side"""
    applier: Applier
    """applier for the rule's right-hand side"""

    times_applied: int
    """number of times the rewrite rule was applied"""

    def __init__(self, lhs: Pattern, rhs: Pattern):
        self.matcher = Matcher(lhs)
        self.applier = Applier(rhs)
        self.where = None       # a `Rewrite` is aimed by its pattern (and phase 5)
        self.site_idx = 0
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
        return ast, self.times_applied, options.times_matched

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
            subst = self.matcher.match_exact(e)
            if subst:
                if ctx.occurence is None or ctx.times_matched == ctx.occurence:
                    e = self.applier.apply(subst)
                    if not isinstance(e, Expr):
                        raise TypeError(f'Substitution produced \'Expr\', got {type(e)} for {e}')
                    self.times_applied += 1
                    if not ctx.is_nested:
                        # the statement survives with an expression rewritten, so
                        # no edit -- but an expression cursor in it is stale
                        self._mark_exprs(*self._site)
                ctx.times_matched += 1
        return e

    def _visit_block(self, block: StmtBlock, ctx: _RewriteContext):
        pattern = self.matcher.pattern
        # the path of the block being visited, before the visit rebuilds it
        path = self._paths.get(id(block))

        rebuilt: list[Stmt] = []
        for pos, stmt in enumerate(block.stmts):
            # `_visit_expr` marks the statement it rewrote an expression of
            self._site = (block, pos)
            s, _ = self._visit_statement(stmt, ctx)
            rebuilt.append(s)
        block = StmtBlock(rebuilt)

        if isinstance(pattern, StmtPattern):
            # every window of `k` consecutive statements is a candidate; a match
            # that is rewritten consumes its whole window, and one that is not
            # keeps its statements
            k = len(pattern.block.stmts)
            stmts = block.stmts
            new_stmts: list[Stmt] = []
            pos = 0
            while pos + k <= len(stmts):
                subst = self.matcher.match_exact(StmtBlock(list(stmts[pos:pos + k])))
                if subst is not None:
                    selected = ctx.occurence is None or ctx.times_matched == ctx.occurence
                    ctx.times_matched += 1
                    if selected:
                        applier = self._nested_applier(1 if ctx.is_nested else ctx.repeat)
                        rw = applier.apply(subst)
                        if not isinstance(rw, StmtBlock):
                            raise TypeError(f'Substitution produced \'StmtBlock\', got {type(rw)} for {rw}')
                        new_stmts.extend(rw.stmts)
                        self.times_applied += 1
                        if path is not None:
                            self._record_at(path, pos, len(rw.stmts), removed=k)
                        pos += k
                        continue
                new_stmts.append(stmts[pos])
                pos += 1
            # the trailing `k - 1` statements no window covered
            new_stmts.extend(stmts[pos:])

            new_block = StmtBlock(new_stmts)
            return new_block, None
        else:
            # pattern does not apply
            return block, None


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

        return func.with_edits(
            self.apply_with_edits(func, occurence=occurence, repeat=repeat)
        )


    def apply_all(self, func: Function):
        """
        Applies the rewrite rule to all matching patterns in the given function.

        Raises `ValueError` if the rewrite rule does not apply.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)}')

        return func.with_edits(self.apply_with_edits(func, occurence=None))

    def apply_with_edits(
        self,
        func: Function, *,
        occurence: int | None = 0,
        repeat: int = 1,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced.

        A statement rule records the window it replaced; an expression rule
        records no edit -- the statement survives -- and marks it as one whose
        expressions moved.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)} for {func}')
        if occurence is not None and (not isinstance(occurence, int) or occurence < 0):
            raise TypeError(f'Expected a non-negative \'int\' or None, got {occurence}')
        if not isinstance(repeat, int) or repeat < 1:
            raise TypeError(f'Expected a positive \'int\' for repeat, got {repeat}')

        if occurence is None:
            self._check_disjoint(func)

        ast, applied, matched = self._engine.apply(
            func.ast, occurence=occurence, repeat=repeat
        )
        if applied == 0:
            raise self._no_match(func, matched, occurence)
        # a user rewrite is unverified, so at least hold it to a valid program
        SyntaxCheck.check(ast, ignore_unknown=True)
        return EditLog(
            func.ast, ast, tuple(self._engine.edits),
            exprs_rewritten=tuple(self._engine.dirty_exprs),
            exprs_preserved=True,
        )

    def _check_disjoint(self, func: Function) -> None:
        """Declines where two matches share a statement.

        A statement pattern is matched by a sliding window, so windows *i* and
        *i+1* can both match; both cannot be rewritten, and rewriting one and
        skipping the other would quietly do less than asked.
        """
        if not isinstance(self.lhs, StmtPattern):
            return
        found = [c for c in find_all(self.lhs, func) if isinstance(c, BlockCursor)]
        for a, b in combinations(found, 2):
            if a.block_path == b.block_path and set(a.span) & set(b.span):
                raise TransformDeclined(
                    f'matches of `{self.lhs.name}` overlap (`{a}` and `{b}`), so '
                    'they cannot both be rewritten; narrow the pattern or aim one '
                    'match'
                )

    def _no_match(
        self, func: Function, matched: int, occurence: int | None
    ) -> TransformReferenceError:
        """Why nothing was rewritten: the pattern named no place, or the
        occurrence index named no match."""
        if matched == 0:
            return TransformReferenceError(
                f'pattern `{self.lhs.name}` matches nothing in `{func.name}`'
            )
        return TransformReferenceError(
            f'occurence={occurence} does not correspond to a match of '
            f'`{self.lhs.name}`; it matches {matched} place(s) in `{func.name}`'
        )
