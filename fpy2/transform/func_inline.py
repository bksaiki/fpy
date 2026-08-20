"""
Function inlining.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ..analysis import (
    AssignDef,
    CallGraph,
    DefineUse,
    DefineUseAnalysis,
    PhiDef,
    Reachability,
    ReachingDefs,
    SyntaxCheck,
)
from ..ast.fpyast import *
from ..env import ForeignEnv
from ..function import Function
from ..number import REAL
from ..utils import Gensym
from .cursor import Cursor, Edit, EditLog, ExprCursor, expr_sites
from .path import StmtPath
from .rename_target import RenameTarget
from .utils import SiteRewriter, check_where


def _replace_ret(block: StmtBlock, new_var: NamedId):
    last_stmt = block.stmts[-1]
    match last_stmt:
        case ReturnStmt():
            new_stmt = Assign(new_var, None, last_stmt.expr, last_stmt.loc)
            block.stmts[-1] = new_stmt
        case ContextStmt():
            _replace_ret(last_stmt.body, new_var)
        case _:
            raise RuntimeError(f'expected a `return` or `with` statement, got `{last_stmt}`')


@dataclass
class _Ctx:
    stmts: list[Stmt]
    is_ctx_expr: bool
    in_while_cond: bool = False

    @staticmethod
    def default():
        return _Ctx(stmts=[], is_ctx_expr=False)


class _FuncInline(SiteRewriter):
    """Function inline visitor."""

    _expr_sited = True   # the candidates are `Call` expressions

    func: FuncDef
    def_use: DefineUseAnalysis
    funcs: set[Function] | None
    inlined: dict[FuncDef, FuncDef]
    recursive: bool
    where: int | Cursor | None

    gensym: Gensym
    free_vars: set[NamedId]
    env: ForeignEnv

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        funcs: set[Function] | None,
        inlined: dict[FuncDef, FuncDef] | None = None,
        recursive: bool = True,
        where: int | Cursor | None = None
    ):
        self.func = func
        self.def_use = def_use
        self.funcs = funcs
        self.where = where
        # Cache of fully-inlined callee bodies, keyed by callee
        # ``FuncDef``.  ``_visit_call`` reuses a cached body across call
        # sites instead of re-inlining the whole callee subtree each
        # time.  ``FuncInline.apply`` may pre-populate it in leaves-first
        # order (so every lookup is a hit); otherwise it fills lazily on
        # first use.  Either way it is never re-built per call site.
        self.inlined = {} if inlined is None else inlined
        self.recursive = recursive

        self.gensym = Gensym(self.def_use.names())
        self.free_vars = set(func.free_vars)
        self.env = func.env.copy()

    def _visit_call(self, e: Call, ctx: _Ctx):
        if not isinstance(e.fn, Function):
            # not calling a function so no inlining
            return super()._visit_call(e, ctx)
        if self.funcs is not None and e.fn not in self.funcs:
            # not a candidate for inlining
            return super()._visit_call(e, ctx)

        idx = self.site_idx
        self.site_idx += 1
        if not self._selects_expr(e, idx):
            # a candidate site, but not the selected one
            return super()._visit_call(e, ctx)
        self._matched += 1
        if ctx.in_while_cond:
            raise RuntimeError(
                f'cannot inline call to `{e.fn.name}` in a `while` condition: '
                f'the spliced body would be evaluated only once, before the loop'
            )

        # Inline the callee body.  Acyclicity is guaranteed by the
        # `CallGraph` guard in `FuncInline.apply`, so this terminates.
        if self.recursive:
            if e.fn.ast in self.inlined:
                # cached
                ast = self.inlined[e.fn.ast]
            else:
                # first time we see this callee, inline it and cache the result
                ast = FuncInline.apply(e.fn.ast, recursive=True)
                self.inlined[e.fn.ast] = ast

            def_use = DefineUse.analyze(ast)
            self.gensym.reserve(*def_use.names())
        else:
            ast = e.fn.ast

        # ASSUME: exactly one trailing return statement in the
        # function body.  Inlining works by replacing the trailing
        # return with an assignment to a fresh temp (see
        # ``_replace_ret``); zero returns leaves nothing to rewrite,
        # and multiple returns would leave non-trailing returns in
        # the inlined block, which would prematurely exit the
        # *caller*.  Reject explicitly with a clear error.
        ret_check = Reachability.analyze(ast)
        n_rets = len(ret_check.ret_stmts)
        if n_rets != 1:
            raise RuntimeError(
                f'cannot inline function `{e.fn.name}`: expected exactly '
                f'one trailing return statement, found {n_rets}.  Zero '
                f'returns leave nothing to rewrite; multiple returns '
                f'would emit non-trailing returns into the caller and '
                f'exit it prematurely.'
            )

        # first, rename all variables in the function body
        reachability = ReachingDefs.analyze(ast)
        subst: dict[NamedId, NamedId] = {}
        for d in reachability.defs:
            if isinstance(d, AssignDef) and not d.is_free:
                subst[d.name] = self.gensym.refresh(d.name)
        ast = RenameTarget.apply(ast, subst)

        # merge free variables
        for name in ast.free_vars:
            if str(name) in self.env:
                # already in the environment, check that it is the same
                val = self.env.get(str(name))
                if val != e.fn.env.get(str(name)):
                    raise RuntimeError(f'cannot inline function `{e.fn.name}` due to conflicting free variable `{name}`')
        self.env = self.env.merge(ast.env, keys=map(str, ast.free_vars))
        self.free_vars |= ast.free_vars

        # bind arguments to parameters
        for arg, param in zip(e.args, ast.args):
            arg = self._visit_expr(arg, ctx)
            if isinstance(param.name, NamedId):
                name = subst.get(param.name, param.name)
                ctx.stmts.append(Assign(name, param.type, arg, e.loc))

        # bind the return value to a fresh variable and splice into the current block
        t = self.gensym.fresh('t')
        _replace_ret(ast.body, t)
        if ast.ctx is not None:
            # overriding context
            stmt = ContextStmt(UnderscoreId(), ForeignVal(ast.ctx, None), ast.body, ast.loc)
            ctx.stmts.append(stmt)
        elif ctx.is_ctx_expr:
            # overriding context must be `RealContext` since
            # we are in a context expression `with e: ...`
            stmt = ContextStmt(UnderscoreId(), ForeignVal(REAL, None), ast.body, ast.loc)
            ctx.stmts.append(stmt)
        else:
            # no overriding context
            ctx.stmts.extend(ast.body.stmts)

        # return the bound value
        return Var(t, e.loc)


    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        cond = self._visit_expr(stmt.cond, _Ctx(ctx.stmts, False, in_while_cond=True))
        body, _ = self._visit_block(stmt.body, ctx)
        return WhileStmt(cond, body, stmt.loc), None

    def _visit_context(self, stmt: ContextStmt, ctx: _Ctx):
        ctx_e = self._visit_expr(stmt.ctx, _Ctx(ctx.stmts, True))
        body, _ = self._visit_block(stmt.body, None)
        s = ContextStmt(stmt.target, ctx_e, body, stmt.loc)
        return s, None

    def _visit_block(self, block: StmtBlock, ctx: _Ctx | None):
        block_ctx = _Ctx.default()
        for pos, stmt in enumerate(block.stmts):
            self._site = (block, pos)
            before = len(block_ctx.stmts)
            stmt, _ = self._visit_statement(stmt, block_ctx)
            block_ctx.stmts.append(stmt)
            # the callee's body is spliced in *ahead* of the statement that held
            # the call, which survives with the call replaced by a variable
            spliced = len(block_ctx.stmts) - before - 1
            if spliced > 0:
                self._record(block, pos, spliced, removed=0)
                self._mark_exprs(block, pos)
        b = StmtBlock(block_ctx.stmts)
        return b, None

    def _visit_function(self, func: FuncDef, ctx: None):
        self._begin(func)
        body, _ = self._visit_block(func.body, None)
        meta = FuncMeta(self.free_vars, func.meta.ctx, func.meta.spec, func.meta.props, self.env)
        return FuncDef(func.name, func.args, body, meta, loc=func.loc)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)


class FuncInline:
    """
    Function inlining.
    """

    @staticmethod
    def sites(
        func: FuncDef,
        within: Cursor | None = None,
        *,
        funcs: Iterable[Function] | None = None,
    ) -> list[ExprCursor]:
        """The candidate call sites of `func`, in visit order: what a `where`
        index counts.  `funcs` filters as it does for :meth:`apply`."""
        keep = None if funcs is None else set(funcs)
        return expr_sites(
            func,
            lambda e: (
                isinstance(e, Call) and isinstance(e.fn, Function)
                and (keep is None or e.fn in keep)
            ),
            within,
        )

    @staticmethod
    def apply(
        func: FuncDef, *,
        def_use: DefineUseAnalysis | None = None,
        funcs: Iterable[Function] | None = None,
        recursive: bool = True,
        where: int | Cursor | None = None
    ) -> FuncDef:
        """
        Applies function inlining to `func` returning the transformed function.

        `where` selects candidate call sites -- calls to a `Function` that pass
        the `funcs` filter -- by index, in visit order (outermost-first), or by a
        cursor.  If `None`, every candidate site is inlined.  With
        `recursive=True` the selected site's callee is still fully flattened.

        Raises `CallGraphError` if the call graph reachable from `func`
        contains a cycle (FPy forbids recursion; inlining a recursive
        call would not terminate).
        """
        return FuncInline.apply_with_edits(
            func,
            def_use=def_use,
            funcs=funcs,
            recursive=recursive,
            where=where,
        ).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        funcs: Iterable[Function] | None = None,
        recursive: bool = True,
        where: int | Cursor | None = None
    ) -> EditLog:
        """:meth:`apply`, with an :class:`EditLog` of what it replaced."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        check_where(where)

        # Recursion guard — see the method docstring.  Also gives us
        # the leaves-first order for the bottom-up path below.
        cg = CallGraph.analyze(func)

        if funcs is not None:
            funcs = set(funcs)

        if recursive and funcs is None and where is None:
            # Bottom-up: inline each reachable function exactly once in
            # leaves-first order, reusing cached results, instead of
            # re-inlining the whole callee subtree at every call site.
            # Entries are finished as they are built: later functions'
            # free-variable merges consume the pruned free-var sets.
            inlined: dict[FuncDef, FuncDef] = {}
            edits: tuple[Edit, ...] = ()
            dirty: tuple[StmtPath, ...] = ()
            for fdef in cg.order:
                fdef_du = DefineUse.analyze(fdef)
                vtor = _FuncInline(
                    fdef, fdef_du, None, recursive=True, inlined=inlined,
                )
                out = FuncInline._finish(vtor.apply())
                if fdef is func:
                    # only this function's own rewrites forward its cursors
                    edits, dirty = tuple(vtor.edits), tuple(vtor.dirty_exprs)
                inlined[fdef] = out
            return EditLog(
                func, inlined[func], edits,
                exprs_rewritten=dirty, exprs_preserved=True,
            )
        else:
            # One-level inlining, or selective inlining via `funcs` /
            # `where`: keep the per-call-site strategy.
            if def_use is None:
                def_use = DefineUse.analyze(func)
            vtor = _FuncInline(func, def_use, funcs, recursive=recursive, where=where)
            result = vtor.apply()
            # `site_idx` is the true candidate count: spliced callee bodies are
            # never re-visited
            vtor.check_site('a call site')
            return EditLog(
                func, FuncInline._finish(result), tuple(vtor.edits),
                exprs_rewritten=tuple(vtor.dirty_exprs), exprs_preserved=True,
            )

    @staticmethod
    def _finish(result: FuncDef) -> FuncDef:
        """Validate an inlined function and prune stale callee names.

        Inlining removes references to callee names, but the visitor only
        ever *adds* free variables — prune function-valued free variables
        with no remaining uses.  Data free variables are DCE's business,
        not ours.  The syntax check runs first so a malformed result
        fails with a syntax error rather than inside `DefineUse`.
        """
        SyntaxCheck.check(result, ignore_unknown=True)

        fn_names = {
            fv for fv in result.free_vars
            if isinstance(result.env.get(str(fv)), Function)
        }
        if not fn_names:
            return result

        du = DefineUse.analyze(result)
        unused: set[NamedId] = set()
        for name in fn_names:
            for d in du.name_to_defs.get(name, ()):
                if (
                    isinstance(d, AssignDef)
                    and d.is_free
                    and not du.uses[d]
                    # a phi successor means the name still flows into a merge
                    and not any(isinstance(s, PhiDef) for s in du.successors[d])
                ):
                    unused.add(name)
        if not unused:
            return result

        meta = FuncMeta(
            result.free_vars - unused, result.meta.ctx,
            result.meta.spec, result.meta.props, result.env,
        )
        return FuncDef(result.name, result.args, result.body, meta, loc=result.loc)
