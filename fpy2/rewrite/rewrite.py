"""
This module defines a rewrite rule.
"""

from itertools import combinations

from ..analysis import SyntaxCheck
from ..ast import *
from ..function import Function
from ..transform import (
    BlockCursor,
    Cursor,
    EditLog,
    SiteRewriter,
    TransformDeclined,
    TransformReferenceError,
    check_where,
)
from ..utils import default_repr, sliding_window
from .applier import Applier
from .matcher import Matcher
from .pattern import ExprPattern, Pattern, StmtPattern
from .search import find_all


@default_repr
class _RewriteContext:
    """How many times to expand the replacement, and whether this run is
    expanding the *rule* rather than rewriting the program."""

    repeat: int
    is_nested: bool

    def __init__(self, repeat: int, *, is_nested: bool = False):
        self.repeat = repeat
        self.is_nested = is_nested



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
        # the sites are the pattern's matches, which are expressions for an
        # expression rule and statements for a statement one
        self._expr_sited = isinstance(lhs, ExprPattern)
        self.where = None
        self.site_idx = 0
        self.times_applied = 0

    def apply(
        self,
        func: FuncDef, *,
        where: int | Cursor | None = None,
        repeat: int = 1
    ):
        self.where = where
        self.times_applied = 0
        ast = self._visit_function(func, _RewriteContext(repeat))
        return ast, self.times_applied, self.site_idx

    def _nested_applier(self, num_times: int):
        if num_times == 1:
            return self.applier
        else:
            # expanding the rule is its own pass: it counts its own matches, so
            # the program-level count is neither consumed nor inflated
            outer, self.site_idx = self.site_idx, 0
            try:
                return Applier(self._expand(num_times))
            finally:
                self.site_idx = outer

    def _expand(self, num_times: int) -> Pattern:
        """The replacement, with the rule applied to itself `num_times - 1`
        more times."""
        pattern = self.applier.pattern
        for _ in range(1, num_times):
            # each round is a pass in its own right, matching from zero
            self.site_idx = 0
            nested = _RewriteContext(1, is_nested=True)
            match pattern:
                case ExprPattern():
                    # run the rule over its own replacement, to emulate taint
                    expr = self._visit_expr(pattern.expr, nested)
                    ast = pattern.to_ast()
                    ast.body.stmts[0] = EffectStmt(expr, None)
                    pattern = ExprPattern(ast)
                case StmtPattern():
                    block, _ = self._visit_block(pattern.block, nested)
                    ast = pattern.to_ast()
                    ast.body = block
                    pattern = StmtPattern(ast)
                case _:
                    raise RuntimeError(f'unreachable case: {pattern}')
        return pattern

    def _visit_expr(self, e: Expr, ctx: _RewriteContext):
        # an expression cursor names the *source* node, so keep it before the
        # visit rebuilds this one
        src = e
        e = super()._visit_expr(e, ctx)
        if isinstance(self.matcher.pattern, ExprPattern):
            # check if rewrite applies here
            subst = self.matcher.match_exact(e)
            if subst:
                idx = self.site_idx
                self.site_idx += 1
                # expanding the rule takes its first match, whatever the
                # program-level `where` says
                if idx == 0 if ctx.is_nested else self._selects_expr_src(src, idx):
                    e = self.applier.apply(subst)
                    if not isinstance(e, Expr):
                        raise TypeError(f'Substitution produced \'Expr\', got {type(e)} for {e}')
                    self.times_applied += 1
                    if not ctx.is_nested:
                        # the statement survives with an expression rewritten, so
                        # no edit -- but an expression cursor in it is stale
                        self._mark_exprs(*self._site)
        return e

    def _selects_expr_src(self, src: Expr, idx: int) -> bool:
        """Whether the match at *src* is the one this rewrite is aimed at."""
        if self._target_expr is not None:
            return src is self._target_expr
        block, pos = self._site
        return self._selects(block, pos, idx)

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
                    idx = self.site_idx
                    self.site_idx += 1
                    selected = (
                        idx == 0 if ctx.is_nested
                        else self._selects_at(path, pos, idx)
                    )
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

    def apply(
        self,
        func: Function,
        where: int | Cursor | None = None,
        *,
        repeat: int = 1,
    ) -> Function:
        """
        Applies the rewrite rule to `func`.

        Parameters
        ----------
        func : Function
            The function to rewrite.
        where : int | Cursor | None
            Which match to rewrite: an index counting the pattern's matches in
            visit order, outermost-first, or a cursor naming a program point,
            which takes the match at or beneath it. A cursor from an earlier
            program is forwarded to this one first. If `None`, rewrite every
            match.
        repeat : int
            How many times to expand the replacement once a match occurs.

        Returns
        -------
        Function
            The rewritten function.

        Raises
        ------
        TransformReferenceError
            If the pattern matches nothing, or `where` names no match.
        TransformDeclined
            If two matches overlap, so they cannot both be rewritten.
        """
        return func.with_edits(self.apply_with_edits(func, where, repeat=repeat))

    def apply_with_edits(
        self,
        func: Function,
        where: int | Cursor | None = None,
        *,
        repeat: int = 1,
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced.

        A statement rule records the window it replaced; an expression rule
        records no edit -- the statement survives -- and marks it as one whose
        expressions moved.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected \'Function\', got {type(func)} for {func}')
        check_where(where)
        if not isinstance(repeat, int) or repeat < 1:
            raise TypeError(f'Expected a positive \'int\' for repeat, got {repeat}')

        where = func.rebase(where)
        if where is None:
            self._check_disjoint(func)

        ast, applied, matched = self._engine.apply(
            func.ast, where=where, repeat=repeat
        )
        if applied == 0:
            raise self._no_match(func, matched, where)
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
        self, func: Function, matched: int, where: int | Cursor | None
    ) -> TransformReferenceError:
        """Why nothing was rewritten: the pattern named no place, or `where`
        named no match."""
        if matched == 0:
            return TransformReferenceError(
                f'pattern `{self.lhs.name}` matches nothing in `{func.name}`'
            )
        aim = f'where={where}' if isinstance(where, int) else f'`{where}`'
        return TransformReferenceError(
            f'{aim} does not correspond to a match of `{self.lhs.name}`; '
            f'it matches {matched} place(s) in `{func.name}`'
        )
