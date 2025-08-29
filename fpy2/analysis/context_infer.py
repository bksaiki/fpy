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

ContextType: TypeAlias = NamedId | Context | TupleContext | None

@default_repr
class FunctionContext:
    """Tracks the context of a function"""

    ctx: ContextType # global context
    args: tuple[ContextType, ...]
    ret: ContextType

    def __init__(self, ctx: ContextType, args: Iterable[ContextType], ret: ContextType):
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
    by_def: dict[Definition, ContextType]
    by_expr: dict[Expr, ContextType]

    @property
    def body_ctx(self):
        return self.func_ctx.ctx

    @property
    def arg_ctxs(self):
        return self.func_ctx.args

    @property
    def return_ctx(self):
        return self.func_ctx.ret


class _ContextInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    by_def: dict[Definition, ContextType]
    by_expr: dict[Expr, ContextType]
    ret_ctx: ContextType
    rvars: Unionfind[ContextType]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.by_def = {}
        self.by_expr = {}
        self.ret_ctx = None
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def _set_context(self, site: Definition, ctx: ContextType):
        self.by_def[site] = ctx

    def _fresh_context_var(self) -> NamedId:
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _free_vars(self, ctx: ContextType | FunctionContext):
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

    def _subst_vars(self, ctx: ContextType | FunctionContext, subst: Mapping[NamedId, ContextType]):
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

    def _generalize(self, ctx: FunctionContext) -> tuple[ContextType | FunctionContext, dict[NamedId, ContextType]]:
        subst: dict[NamedId, ContextType] = {}
        for i, fv in enumerate(sorted(self._free_vars(ctx))):
            t = self.rvars.find(fv)
            match t:
                case NamedId():
                    subst[fv] = NamedId(f'r{i + 1}')
                case _:
                    subst[fv] = t
        c = self._subst_vars(ctx, subst)
        return c, subst

    def _resolve_context(self, ctx: ContextType):
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

    def _unify(self, a_ctx: ContextType, b_ctx: ContextType) -> ContextType:
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

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ctx: ContextType):
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

    def _visit_var(self, e: Var, ctx: ContextType):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: ContextType):
        return ctx

    def _visit_foreign(self, e: ForeignVal, ctx: ContextType):
        return None

    def _visit_decnum(self, e: Decnum, ctx: ContextType):
        return ctx

    def _visit_hexnum(self, e: Hexnum, ctx: ContextType):
        return ctx

    def _visit_integer(self, e: Integer, ctx: ContextType):
        return ctx

    def _visit_rational(self, e: Rational, ctx: ContextType):
        return ctx

    def _visit_digits(self, e: Digits, ctx: ContextType):
        return ctx

    def _visit_nullaryop(self, e: NullaryOp, ctx: ContextType):
        return ctx

    def _visit_unaryop(self, e: UnaryOp, ctx: ContextType):
        arg_ctx = self._visit_expr(e.arg, ctx)
        match e:
            case Enumerate():
                # a -> tuple[a, ctx]
                return TupleContext([ctx, arg_ctx])
            case _:
                return ctx

    def _visit_binaryop(self, e: BinaryOp, ctx: ContextType):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        return ctx

    def _visit_ternaryop(self, e: TernaryOp, ctx: ContextType):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return ctx

    def _visit_naryop(self, e: NaryOp, ctx: ContextType):
        match e:
            case Zip():
                return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)
            case _:
                for arg in e.args:
                    self._visit_expr(arg, ctx)
                return ctx

    def _visit_compare(self, e: Compare, ctx: ContextType):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return ctx

    def _visit_call(self, e: Call, ctx: ContextType):
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

    def _visit_tuple_expr(self, e: TupleExpr, ctx: ContextType):
        return TupleContext(self._visit_expr(arg, ctx) for arg in e.args)

    def _visit_list_expr(self, e: ListExpr, ctx: ContextType):
        if len(e.args) == 0:
            return ctx
        else:
            elt_ctx = self._visit_expr(e.args[0], ctx)
            for arg in e.args[1:]:
                elt_ctx = self._unify(elt_ctx, self._visit_expr(arg, ctx))
            return elt_ctx

    def _visit_list_comp(self, e: ListComp, ctx: ContextType):
        for target, iterable in zip(e.targets, e.iterables):
            iter_ctx = self._visit_expr(iterable, ctx)
            self._visit_binding(e, target, iter_ctx)

        elt_ctx = self._visit_expr(e.elt, None)
        return elt_ctx

    def _visit_list_ref(self, e: ListRef, ctx: ContextType):
        value_ctx = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        return value_ctx

    def _visit_list_slice(self, e: ListSlice, ctx: ContextType):
        value_ctx = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return value_ctx

    def _visit_list_set(self, e: ListSet, ctx: ContextType):
        arr_ctx = self._visit_expr(e.array, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.value, ctx)
        return arr_ctx

    def _visit_if_expr(self, e: IfExpr, ctx: ContextType):
        self._visit_expr(e.cond, ctx)
        ift_ctx = self._visit_expr(e.ift, ctx)
        iff_ctx = self._visit_expr(e.iff, ctx)
        return self._unify(ift_ctx, iff_ctx)

    def _visit_context_expr(self, e: ContextExpr, ctx: ContextType):
        return None

    def _visit_assign(self, stmt: Assign, ctx: ContextType):
        e_ctx = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.binding, e_ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: ContextType):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: ContextType):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: ContextType):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # need to merge variables introduced on both sides
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: ContextType):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: ContextType):
        iter_ctx = self._visit_expr(stmt.iterable, ctx)
        self._visit_binding(stmt, stmt.target, iter_ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            ctx = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ctx)

        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: ContextType):
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

    def _visit_assert(self, stmt: AssertStmt, ctx: ContextType):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: ContextType):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: ContextType):
        ret_ctx = self._visit_expr(stmt.expr, ctx)
        self.ret_ctx = ret_ctx
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: ContextType):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                body_ctx: ContextType = self._fresh_context_var()
            case FPCoreContext():
                body_ctx = None
            case _:
                body_ctx = func.ctx

        # generate context variables for each argument
        arg_ctxs: list[ContextType] = []
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
        return FunctionContext(body_ctx, arg_ctxs, ret_ctx)

    def _visit_expr(self, expr: Expr, ctx: ContextType) -> ContextType:
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
    def infer(func: FuncDef, def_use: DefineUseAnalysis | None = None):
        """
        Performs rounding context inference.

        Produces a map from definition sites to their rounding contexts.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        if def_use is None:
            def_use = DefineUse.analyze(func)
        inst = _ContextInferInstance(func, def_use)
        return inst.infer()
