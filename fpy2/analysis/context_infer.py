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

    elts: tuple[Self | NamedId | Context, ...]

    def __init__(self, elts: Iterable[Self | NamedId | Context]):
        self.elts = tuple(elts)

    def __eq__(self, other):
        return isinstance(other, TupleContext) and self.elts == other.elts

    def __hash__(self):
        return hash(self.elts)

TypeContext: TypeAlias = NamedId | Context | TupleContext

@default_repr
class FunctionContext:
    """Tracks the context of a function"""

    ctx: TypeContext # global context
    args: tuple[TypeContext, ...]
    ret: TypeContext

    def __init__(self, ctx: TypeContext, args: Iterable[TypeContext], ret: TypeContext):
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
    by_def: dict[Definition, TypeContext]
    by_expr: dict[Expr, TypeContext]

    @property
    def body_ctx(self):
        return self.func_ctx.ctx

    @property
    def arg_ctxs(self):
        return self.func_ctx.args

    @property
    def return_ctx(self):
        return self.func_ctx.ret


class ContextTypeInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    by_def: dict[Definition, TypeContext]
    by_expr: dict[Expr, TypeContext]
    ret_ctx: TypeContext | None
    rvars: Unionfind[TypeContext]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.by_def = {}
        self.by_expr = {}
        self.ret_ctx = None
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def _set_context(self, site: Definition, ctx: TypeContext):
        self.by_def[site] = ctx

    def _fresh_context_var(self) -> NamedId:
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _free_vars(self, ctx: TypeContext | FunctionContext):
        match ctx:
            case Context():
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

    def _subst_vars(self, ctx: TypeContext | FunctionContext, subst: Mapping[NamedId, TypeContext]):
        match ctx:
            case Context():
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

    def _generalize(self, ctx: FunctionContext) -> tuple[TypeContext | FunctionContext, dict[NamedId, TypeContext]]:
        subst: dict[NamedId, TypeContext] = {}
        for i, fv in enumerate(sorted(self._free_vars(ctx))):
            t = self.rvars.find(fv)
            match t:
                case NamedId():
                    subst[fv] = NamedId(f'r{i + 1}')
                case _:
                    subst[fv] = t
        c = self._subst_vars(ctx, subst)
        return c, subst

    def _resolve_context(self, ctx: TypeContext):
        match ctx:
            case Context():
                return ctx
            case NamedId():
                return self.rvars.get(ctx, ctx)
            case TupleContext():
                elts = (self._resolve_context(elt) for elt in ctx.elts)
                return self.rvars.add(TupleContext(elts))
            case _:
                raise RuntimeError(f'unknown context: {ctx}')

    def _unify(self, a_ctx: TypeContext, b_ctx: TypeContext) -> TypeContext:
        a_ctx = self.rvars.get(a_ctx, a_ctx)
        b_ctx = self.rvars.get(b_ctx, b_ctx)
        match a_ctx, b_ctx:
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
                raise ContextInferError(f'incompatible context types: {a_ctx} != {b_ctx}')

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ctx: TypeContext):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self._set_context(d, ctx)
            case UnderscoreId():
                pass
            case TupleBinding():
                if not isinstance(ctx, TupleContext) or len(ctx.elts) != len(target.elts):
                    raise ContextInferError(f'cannot unpack context type `{ctx}` into `{target.format()}`')
                for elt, elt_ctx in zip(target.elts, ctx.elts):
                    self._visit_binding(site, elt, elt_ctx)
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_var(self, e: Var, ctx: TypeContext):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: TypeContext):
        return ctx

    def _visit_foreign(self, e: ForeignVal, ctx: TypeContext):
        return self._fresh_context_var()

    def _visit_decnum(self, e: Decnum, ctx: TypeContext):
        return ctx

    def _visit_hexnum(self, e: Hexnum, ctx: TypeContext):
        return ctx

    def _visit_integer(self, e: Integer, ctx: TypeContext):
        return ctx

    def _visit_rational(self, e: Rational, ctx: TypeContext):
        return ctx

    def _visit_digits(self, e: Digits, ctx: TypeContext):
        return ctx

    def _visit_nullaryop(self, e: NullaryOp, ctx: TypeContext):
        return ctx

    def _visit_unaryop(self, e: UnaryOp, ctx: TypeContext):
        arg_ctx = self._visit_expr(e.arg, ctx)
        match e:
            case Enumerate():
                # a -> tuple[a, ctx]
                return TupleContext([ctx, arg_ctx])
            case _:
                return ctx

    def _visit_binaryop(self, e: BinaryOp, ctx: TypeContext):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return ctx

    def _visit_ternaryop(self, e: TernaryOp, ctx: TypeContext):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return ctx

    def _visit_naryop(self, e: NaryOp, ctx: TypeContext):
        match e:
            case Zip():
                return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)
            case _:
                for arg in e.args:
                    self._visit_expr(arg, ctx)
                return ctx

    def _visit_compare(self, e: Compare, ctx: TypeContext):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return ctx

    def _visit_call(self, e: Call, ctx: TypeContext):
        # get around circular imports
        from ..function import Function

        match e.fn:
            case None:
                # calling None => can't conclude anything
                return self._fresh_context_var()
            case Primitive():
                # calling a primitive => can't conclude anything
                # TODO: annotations to attach context info to primitives
                fn_ctx = ContextInfer.primitive(e.fn)
                # instantiate the function context
                fn_ctx = cast(FunctionContext, self._instantiate(fn_ctx))
                # merge caller context
                self._unify(ctx, fn_ctx.ctx)
                # merge arguments
                if len(fn_ctx.args) != len(e.args):
                    raise ContextInferError(
                        f'primitive {e.fn} expects {len(fn_ctx.args)} arguments, '
                        f'got {len(e.args)}'
                    )
                for arg, expect_ty in zip(e.args, fn_ctx.args):
                    ty = self._visit_expr(arg, ctx)
                    self._unify(ty, expect_ty)

                return fn_ctx.ret
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
                    ty = self._visit_expr(arg, ctx)
                    self._unify(ty, expect_ty)

                return fn_ctx.ret
            case type() if issubclass(e.fn, Context):
                # calling context constructor
                # TODO: can infer if the arguments are statically known
                raise ContextInferError(f'cannot infer context `{e.fn}`')
            case _:
                raise ContextInferError(f'cannot infer context for call with `{e.fn}`')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: TypeContext):
        return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)

    def _visit_list_expr(self, e: ListExpr, ctx: TypeContext):
        if len(e.args) == 0:
            return ctx
        else:
            elt_ctx = self._visit_expr(e.args[0], ctx)
            for arg in e.args[1:]:
                elt_ctx = self._unify(elt_ctx, self._visit_expr(arg, ctx))
            return elt_ctx

    def _visit_list_comp(self, e: ListComp, ctx: TypeContext):
        for target, iterable in zip(e.targets, e.iterables):
            iter_ctx = self._visit_expr(iterable, ctx)
            self._visit_binding(e, target, iter_ctx)

        elt_ctx = self._visit_expr(e.elt, ctx)
        return elt_ctx

    def _visit_list_ref(self, e: ListRef, ctx: TypeContext):
        value_ctx = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        return value_ctx

    def _visit_list_slice(self, e: ListSlice, ctx: TypeContext):
        value_ctx = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return value_ctx

    def _visit_list_set(self, e: ListSet, ctx: TypeContext):
        arr_ctx = self._visit_expr(e.array, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.value, ctx)
        return arr_ctx

    def _visit_if_expr(self, e: IfExpr, ctx: TypeContext):
        self._visit_expr(e.cond, ctx)
        ift_ctx = self._visit_expr(e.ift, ctx)
        iff_ctx = self._visit_expr(e.iff, ctx)
        return self._unify(ift_ctx, iff_ctx)

    def _visit_attribute(self, e: Attribute, ctx: TypeContext):
        raise NotImplementedError(e)

    def _visit_assign(self, stmt: Assign, ctx: TypeContext):
        e_ctx = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.binding, e_ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: TypeContext):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: TypeContext):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: TypeContext):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # need to merge variables introduced on both sides
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: TypeContext):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: TypeContext):
        iter_ctx = self._visit_expr(stmt.iterable, ctx)
        self._visit_binding(stmt, stmt.target, iter_ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: TypeContext):
        if not isinstance(stmt.ctx, ForeignVal) or not isinstance(stmt.ctx.val, Context):
            raise ContextInferError(f'cannot infer context for `{stmt.ctx}`')

        body_ctx = stmt.ctx.val
        self._visit_block(stmt.body, body_ctx)
        return ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: TypeContext):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: TypeContext):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: TypeContext):
        ret_ctx = self._visit_expr(stmt.expr, ctx)
        self.ret_ctx = ret_ctx
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: TypeContext):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                body_ctx: TypeContext = self._fresh_context_var()
            case FPCoreContext():
                body_ctx = func.ctx.to_context()
            case _:
                body_ctx = func.ctx

        # generate context variables for each argument
        arg_ctxs: list[TypeContext] = []
        for arg in func.args:
            ctx_var = self._fresh_context_var()
            arg_ctxs.append(ctx_var)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_context(d, ctx_var)

        # generate context variables for each free variables
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            self._set_context(d, self._fresh_context_var())

        self._visit_block(func.body, body_ctx)
        assert self.ret_ctx is not None # function has no return statement

        # generalize the function context
        arg_ctxs = [self._resolve_context(ctx) for ctx in arg_ctxs]
        ret_ctx = self._resolve_context(self.ret_ctx)
        return FunctionContext(body_ctx, arg_ctxs, ret_ctx)

    def _visit_expr(self, expr: Expr, ctx: TypeContext) -> TypeContext:
        ret_ctx = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ret_ctx
        return ret_ctx

    def infer(self):
        # context inference on body
        ctx = self._visit_function(self.func, None)

        # generalize the output context
        fn_ctx, subst = self._generalize(ctx)
        fn_ctx = cast(FunctionContext, fn_ctx)

        # rename unbound context variables
        for t in self.rvars:
            if isinstance(t, NamedId) and t not in subst:
                subst[t] = NamedId(f'r{len(subst) + 1}')

        # resolve definition/expr contexts
        by_defs = {
            d: self._subst_vars(self._resolve_context(ctx), subst)
            for d, ctx in self.by_def.items()
        }
        by_expr = {
            e: self._subst_vars(self._resolve_context(ctx), subst)
            for e, ctx in self.by_expr.items()
        }
        return ContextAnalysis(fn_ctx, by_defs, by_expr)



class ContextTypeInferPrimitive:
    """
    Context inference for primitives.

    This is a simpler version of context inference that only
    interprets the context annotations on primitives.
    """
    
    prim: Primitive
    gensym: Gensym
    subst: dict[str, NamedId]

    def __init__(self, prim: Primitive):
        self.prim = prim
        self.gensym = Gensym()
        self.subst = {}

    def _arg_ctx(self, ty: TypeAnn, ctx: str | tuple) -> TypeContext:
        match ty:
            case None | RealTypeAnn() | BoolTypeAnn() | AnyTypeAnn():
                if not isinstance(ctx, str):
                    raise ValueError(f"expected context variable for argument of type {ty}, got {ctx}")
                if ctx not in self.subst:
                    self.subst[ctx] = self.gensym.fresh('r')
                return self.subst[ctx]
            case TupleTypeAnn():
                if not isinstance(ctx, tuple):
                    raise ValueError(f"expected tuple context for argument of type {ty}, got {ctx}")
                if len(ty.elts) != len(ctx):
                    raise ValueError(f"tuple context length mismatch: expected {len(ty.elts)}, got {len(ctx)}")
                elts = [self._arg_ctx(t, c) for t, c in zip(ty.elts, ctx)]
                return TupleContext(elts)
            case ListTypeAnn():
                return self._arg_ctx(ty.elt, ctx)
            case _:
                raise RuntimeError(f'unknown type: {ty}')

    def _cvt_return_ctx(self, ctx: Context | str | tuple) -> TypeContext:
        match ctx:
            case str():
                if ctx not in self.subst:
                    raise ContextInferError(f'unbound context variable in ret_ctx: {ctx}')
                return self.subst[ctx]
            case Context():
                return ctx
            case tuple():
                elts = [self._cvt_return_ctx(c) for c in ctx]
                return TupleContext(elts)
            case _:
                raise ValueError(f"invalid context in ret_ctx: {ctx}")

    def _default_return_ctx(self, ty: TypeAnn) -> TypeContext:
        match ty:
            case None | RealTypeAnn() | BoolTypeAnn() | AnyTypeAnn():
                return self.gensym.fresh('r')
            case TupleTypeAnn():
                elts = [self._default_return_ctx(t) for t in ty.elts]
                return TupleContext(elts)
            case ListTypeAnn():
                return self._default_return_ctx(ty.elt)
            case _:
                raise RuntimeError(f'unknown type: {ty}')

    def infer(self) -> FunctionContext:
        # interpret primitive context
        ctx = self.gensym.fresh('r')
        if self.prim.ctx is not None:
            self.subst[self.prim.ctx] = ctx

        # interpret argument contexts
        arg_ctxs: list[TypeContext] = []
        if self.prim.arg_ctxs is None:
            for _ in self.prim.arg_types:
                arg_ctxs.append(self.gensym.fresh('r'))
        else:
            # none specified
            assert len(self.prim.arg_ctxs) == len(self.prim.arg_types)
            for ty, arg_ctx in zip(self.prim.arg_types, self.prim.arg_ctxs):
                arg_ctxs.append(self._arg_ctx(ty, arg_ctx))

        # interpret return context
        if self.prim.ret_ctx is None:
            ret_ctx = self.gensym.fresh('r')
        else:
            ret_ctx = self._cvt_return_ctx(self.prim.ret_ctx)

        return FunctionContext(ctx, arg_ctxs, ret_ctx)

###########################################################
# Context inference

class ContextInfer:
    """
    Context inference.

    Like type checking but for rounding contexts.
    Most rounding contexts are statically known, so we
    can assign every statement (or expression) a rounding context
    if it can be determined.
    """

    @staticmethod
    def infer(func: FuncDef, def_use: DefineUseAnalysis | None = None):
        """
        Performs rounding context inference.

        Produces a map from definition sites to their rounding contexts.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        if def_use is None:
            def_use = DefineUse.analyze(func)
        inst = ContextTypeInferInstance(func, def_use)
        return inst.infer()

    @staticmethod
    def primitive(prim: Primitive) -> FunctionContext:
        """
        Infers the context of a primitive.
        """
        if not isinstance(prim, Primitive):
            raise TypeError(f'expected a \'Primitive\', got {prim}')
        return ContextTypeInferPrimitive(prim).infer()
