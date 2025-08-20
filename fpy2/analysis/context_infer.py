"""
Context inference.

TODO: unification for context variables
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Self, TypeAlias, cast

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context
from ..primitive import Primitive
from ..utils import Gensym, NamedId, Unionfind, default_repr
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite


@default_repr
class TupleContext:
    """Tracks the contexts of tuples"""

    elts: tuple[Self | NamedId | Context | None, ...]

    def __init__(self, elts: Iterable[Self | NamedId | Context | None]):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, TupleContext) and self.elts == other.elts

    def __hash__(self):
        return hash(self.elts)

_Context: TypeAlias = NamedId | Context | TupleContext | None

@default_repr
class FunctionContext:
    """Tracks the context of a function"""

    ctx: _Context # global context
    args: tuple[_Context, ...]
    ret: _Context

    def __init__(self, ctx: _Context, args: Iterable[_Context], ret: _Context):
        self.ctx = ctx
        self.args = tuple(args)
        self.ret = ret

    def __eq__(self, other):
        return (
            isinstance(other, FunctionContext)
            and self.ctx == other.ctx
            and self.args == other.args
            and self.ret == other.ret
        )

    def __hash__(self):
        return hash((self.ctx, self.args, self.ret))


class ContextInferError(Exception):
    """Context inference error for FPy programs."""
    pass

@dataclass(frozen=True)
class ContextAnalysis:
    func_ctx: FunctionContext
    by_def: dict[Definition, _Context]
    by_expr: dict[Expr, _Context]

    @property
    def body_ctx(self):
        return self.func_ctx.ctx

    @property
    def arg_ctxs(self):
        return self.func_ctx.args

    @property
    def ret_ctx(self):
        return self.func_ctx.ret


class _ContextInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    by_def: dict[Definition, _Context]
    by_expr: dict[Expr, _Context]
    ret_ctx: _Context
    rvars: Unionfind[_Context]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.by_def = {}
        self.by_expr = {}
        self.ret_ctx = None
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def infer(self):
        f_ctx = self._visit_function(self.func, None)
        by_defs = {
            d: self._resolve_context(ctx)
            for d, ctx in self.by_def.items()
        }
        by_expr = {
            e: self._resolve_context(ctx)
            for e, ctx in self.by_expr.items()
        }
        return ContextAnalysis(f_ctx, by_defs, by_expr)

    def _set_context(self, site: Definition, ctx: _Context):
        self.by_def[site] = ctx

    def _fresh_context_var(self) -> NamedId:
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _free_vars(self, ctx: _Context | FunctionContext):
        match ctx:
            case None | Context():
                return set()
            case NamedId():
                return { ctx }
            case TupleContext():
                fvs: set[NamedId] = set()
                for elt in ctx.elts:
                    fvs |= self._free_vars(elt)
                return fvs
            case FunctionContext():
                fvs = self._free_vars(ctx.ctx)
                for arg in ctx.args:
                    fvs |= self._free_vars(arg)
                fvs |= self._free_vars(ctx.ret)
                return fvs
            case _:
                raise RuntimeError(f'unknown context: {ctx}')

    def _subst_vars(self, ctx: _Context | FunctionContext, subst: Mapping[NamedId, _Context]):
        match ctx:
            case None | Context():
                return ctx
            case NamedId():
                return subst.get(ctx, ctx)
            case TupleContext():
                return TupleContext(self._subst_vars(elt, subst) for elt in ctx.elts)
            case FunctionContext():
                return FunctionContext(
                    self._subst_vars(ctx.ctx, subst),
                    (self._subst_vars(arg, subst) for arg in ctx.args),
                    self._subst_vars(ctx.ret, subst)
                )
            case _:
                raise RuntimeError(f'unknown context: {ctx}')

    def _instantiate(self, ctx: FunctionContext):
        subst: dict[NamedId, NamedId] = {}
        for fv in sorted(self._free_vars(ctx)):
            subst[fv] = self._fresh_context_var()
        return self._subst_vars(ctx, subst)

    def _generalize(self, ctx: FunctionContext):
        subst: dict[NamedId, _Context] = {}
        for i, fv in enumerate(sorted(self._free_vars(ctx))):
            t = self.rvars.find(fv)
            match t:
                case NamedId():
                    subst[fv] = NamedId(f'r{i + 1}')
                case _:
                    subst[fv] = t
        return self._subst_vars(ctx, subst)

    def _resolve_context(self, ctx: _Context):
        match ctx:
            case None | Context():
                return ctx
            case NamedId():
                return self.rvars.get(ctx, ctx)
            case TupleContext():
                elts = (self._resolve_context(elt) for elt in ctx.elts)
                return self.rvars.add(TupleContext(elts))
            case _:
                raise RuntimeError(f'unknown context: {ctx}')

    def _unify(self, a_ctx: _Context, b_ctx: _Context) -> _Context:
        a_ctx = self.rvars.get(a_ctx, a_ctx)
        b_ctx = self.rvars.get(b_ctx, b_ctx)
        match a_ctx, b_ctx:
            case (None, _) | (None, _):
                return None
            case _, NamedId():
                a_ctx = self.rvars.add(a_ctx)
                return self.rvars.union(a_ctx, b_ctx)
            case NamedId(), _:
                b_ctx = self.rvars.add(b_ctx)
                return self.rvars.union(b_ctx, a_ctx)
            case Context(), Context():
                if not a_ctx.is_equiv(b_ctx):
                    raise ContextInferError(f'incompatible context types: {a_ctx} != {b_ctx}')
                return a_ctx
            case TupleContext(), TupleContext():
                if len(a_ctx.elts) != len(b_ctx.elts):
                    raise ContextInferError(f'incompatible context types: {a_ctx} != {b_ctx}')
                elts = (self._unify(a_elt, b_elt) for a_elt, b_elt in zip(a_ctx.elts, b_ctx.elts))
                ctx = self.rvars.add(TupleContext(elts))
                ctx = self.rvars.union(ctx, self.rvars.add(a_ctx))
                ctx = self.rvars.union(ctx, self.rvars.add(b_ctx))
                return ctx
            case _:
                return None

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ctx: _Context):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self._set_context(d, ctx)
            case UnderscoreId():
                pass
            case TupleBinding():
                match ctx:
                    case None:
                        for elt in target.elts:
                            self._visit_binding(site, elt, None)
                    case TupleContext():
                        for elt, elt_ctx in zip(target.elts, ctx.elts):
                            self._visit_binding(site, elt, elt_ctx)
                    case _:
                        raise RuntimeError(f'unexpected context for binding {ctx}')
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_var(self, e: Var, ctx: _Context):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: _Context):
        return ctx

    def _visit_foreign(self, e: ForeignVal, ctx: _Context):
        return None

    def _visit_decnum(self, e: Decnum, ctx: _Context):
        return ctx

    def _visit_hexnum(self, e: Hexnum, ctx: _Context):
        return ctx

    def _visit_integer(self, e: Integer, ctx: _Context):
        return ctx

    def _visit_rational(self, e: Rational, ctx: _Context):
        return ctx

    def _visit_digits(self, e: Digits, ctx: _Context):
        return ctx

    def _visit_nullaryop(self, e: NullaryOp, ctx: _Context):
        return ctx

    def _visit_unaryop(self, e: UnaryOp, ctx: _Context):
        self._visit_expr(e.arg, ctx)
        return ctx

    def _visit_binaryop(self, e: BinaryOp, ctx: _Context):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return ctx

    def _visit_ternaryop(self, e: TernaryOp, ctx: _Context):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return ctx

    def _visit_naryop(self, e: NaryOp, ctx: _Context):
        match e:
            case Zip():
                return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)
            case _:
                for arg in e.args:
                    self._visit_expr(arg, ctx)
                return ctx

    def _visit_compare(self, e: Compare, ctx: _Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return ctx

    def _visit_call(self, e: Call, ctx: _Context):
        # get around circular imports
        from ..function import Function

        match e.fn:
            case None:
                # unbound call
                return None
            case Primitive():
                # calling a primitive => can't conclude anything
                # TODO: annotations to attach context info to primitives
                return None
            case Function():
                # calling a function
                # TODO: guard against recursion
                fn_info = ContextInfer.infer(e.fn.ast)
                if len(fn_info.arg_ctxs) != len(e.args):
                    raise ContextInferError(
                        f'function {e.fn} expects {len(fn_info.arg_ctxs)} arguments, '
                        f'got {len(e.args)}'
                    )

                # instantiate the function context
                fn_ctx = cast(FunctionContext, self._instantiate(fn_info.func_ctx))
                # merge caller context
                self._unify(ctx, fn_ctx.ctx)
                # merge arguments
                for arg, expect_ty in zip(e.args, fn_ctx.args):
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, expect_ty)

                return fn_ctx.ret
            case _:
                raise NotImplementedError(f'cannot type check {e.fn} {e.func}')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _Context):
        return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)

    def _visit_list_expr(self, e: ListExpr, ctx: _Context):
        if len(e.args) == 0:
            return ctx
        else:
            elt_ctx = self._visit_expr(e.args[0], ctx)
            for arg in e.args[1:]:
                elt_ctx = self._unify(elt_ctx, self._visit_expr(arg, ctx))
            return elt_ctx

    def _visit_list_comp(self, e: ListComp, ctx: _Context):
        for target, iterable in zip(e.targets, e.iterables):
            iter_ctx = self._visit_expr(iterable, ctx)
            self._visit_binding(e, target, iter_ctx)

        elt_ctx = self._visit_expr(e.elt, None)
        return elt_ctx

    def _visit_list_ref(self, e: ListRef, ctx: _Context):
        value_ctx = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        return value_ctx

    def _visit_list_slice(self, e: ListSlice, ctx: _Context):
        value_ctx = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return value_ctx

    def _visit_list_set(self, e: ListSet, ctx: _Context):
        arr_ctx = self._visit_expr(e.array, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.value, ctx)
        return arr_ctx

    def _visit_if_expr(self, e: IfExpr, ctx: _Context):
        self._visit_expr(e.cond, ctx)
        ift_ctx = self._visit_expr(e.ift, ctx)
        iff_ctx = self._visit_expr(e.iff, ctx)
        return self._unify(ift_ctx, iff_ctx)

    def _visit_context_expr(self, e: ContextExpr, ctx: _Context):
        return None

    def _visit_assign(self, stmt: Assign, ctx: _Context):
        e_ctx = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.binding, e_ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _Context):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _select_def_repr(self, d: Definition):
        return d.assigns()[0]

    def _visit_if(self, stmt: IfStmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # need to merge variables introduced on both sides
        defs_in_ift, defs_out_ift = self.def_use.blocks[stmt.ift]
        defs_in_iff, defs_out_iff = self.def_use.blocks[stmt.iff]
        intros_ift = defs_in_ift.fresh_in(defs_out_ift)
        intros_iff = defs_in_iff.fresh_in(defs_out_iff)
        for intro in intros_ift & intros_iff:
            ift_def = self._select_def_repr(defs_out_ift[intro])
            iff_def = self._select_def_repr(defs_out_iff[intro])
            phi_ctx = self._unify(self.by_def[ift_def], self.by_def[iff_def])
            if phi_ctx is None:
                # if we can't infer a context, clear whatever we inferred in the branches
                # TODO: technically this is wrong; we know the types in the branches
                # but the type of the merged variable is unknown
                self.by_def[ift_def] = None
                self.by_def[iff_def] = None

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: _Context):
        iter_ctx = self._visit_expr(stmt.iterable, ctx)
        self._visit_binding(stmt, stmt.target, iter_ctx)
        self._visit_block(stmt.body, ctx)
        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: _Context):
        match stmt.ctx:
            case ForeignVal():
                if isinstance(stmt.ctx.val, Context):
                    body_ctx = stmt.ctx.val
                else:
                    body_ctx = None
            case Var() | ForeignAttribute() | ContextExpr():
                body_ctx = None
            case _:
                raise RuntimeError(f'unreachable: {stmt.ctx}')

        self._visit_block(stmt.body, body_ctx)
        return ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: _Context):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _Context):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _Context):
        ret_ctx = self._visit_expr(stmt.expr, ctx)
        self.ret_ctx = ret_ctx
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: _Context):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                body_ctx: _Context = self._fresh_context_var()
            case FPCoreContext():
                body_ctx = None
            case _:
                body_ctx = func.ctx

        # generate context variables for each argument
        arg_ctxs: list[_Context] = []
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                ctx_var = self._fresh_context_var()
                self._set_context(d, ctx_var)
                arg_ctxs.append(ctx_var)
            else:
                arg_ctxs.append(None)

        # generate context variables for each free variables
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            self._set_context(d, self._fresh_context_var())

        self._visit_block(func.body, body_ctx)

        # generalize the function context
        arg_ctxs = [self._resolve_context(ctx) for ctx in arg_ctxs]
        ret_ctx = self._resolve_context(self.ret_ctx)
        ty = FunctionContext(body_ctx, arg_ctxs, ret_ctx)
        return cast(FunctionContext, self._generalize(ty))

    def _visit_expr(self, expr: Expr, ctx: _Context) -> _Context:
        ret_ctx = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ret_ctx
        return ret_ctx


class ContextInfer:
    """
    Context inference.

    Like type checking but for rounding contexts.
    Most rounding contexts are statically known, so we
    can assign every statement (or expression) a rounding context
    if it can be determined.
    """

    #
    # <context> ::= C_i
    #             | <context> x <context>
    #             | [<context>] <context> -> <context>

    @staticmethod
    def infer(func: FuncDef):
        """
        Performs rounding context inference.

        Produces a map from definition sites to their rounding contexts.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        inst = _ContextInferInstance(func, def_use)
        return inst.infer()
