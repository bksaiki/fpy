"""
Context inference.

TODO: unification for context variables
"""

from dataclasses import dataclass
from typing import cast

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context
from ..primitive import Primitive
from ..utils import Gensym, NamedId, Unionfind

from .context_types import *
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite
from .type_check import TypeCheck, TypeAnalysis
from .types import *

class ContextInferError(Exception):
    """Context inference error for FPy programs."""
    pass

@dataclass(frozen=True)
class ContextAnalysis:
    func_ty: FunctionTypeContext
    by_def: dict[Definition, TypeContext]
    by_expr: dict[Expr, TypeContext]

    @property
    def body_ctx(self):
        return self.func_ty.ctx

    @property
    def arg_types(self):
        return self.func_ty.args

    @property
    def return_types(self):
        return self.func_ty.ret


class ContextTypeInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis

    by_def: dict[Definition, TypeContext]
    by_expr: dict[Expr, TypeContext]
    ret_ty: TypeContext | None
    rvars: Unionfind[ContextParam]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, type_info: TypeAnalysis):
        self.func = func
        self.def_use = def_use
        self.type_info = type_info
        self.by_def = {}
        self.by_expr = {}
        self.ret_ty = None
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def _lookup_ty(self, e: Expr):
        return self.type_info.by_expr[e]

    def _from_scalar(self, ty: Type, ctx: ContextParam):
        match ty:
            case BoolType():
                return BoolTypeContext()
            case RealType():
                return RealTypeContext(ctx)
            case ContextType():
                return ContextTypeContext()
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _set_context(self, site: Definition, ty: TypeContext):
        self.by_def[site] = ty

    def _fresh_context_var(self) -> NamedId:
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _resolve_context(self, ctx: ContextParam) -> ContextParam:
        return self.rvars.get(ctx, ctx)

    def _resolve(self, ty: TypeContext) -> TypeContext:
        match ty:
            case BoolTypeContext() | ContextTypeContext() | VarTypeContext():
                return ty
            case RealTypeContext():
                ctx = self._resolve_context(ty.ctx)
                return RealTypeContext(ctx)
            case TupleTypeContext():
                elts = (self._resolve(elt) for elt in ty.elts)
                return TupleTypeContext(*elts)
            case ListTypeContext():
                elt = self._resolve(ty.elt)
                return ListTypeContext(elt)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _instantiate(self, ty: TypeContext) -> TypeContext:
        subst: dict[NamedId, ContextParam] = {}
        for fv in sorted(ty.free_vars()):
            subst[fv] = self.gensym.fresh()
        return ty.subst(subst)

    def _generalize(self, ty: TypeContext) -> tuple[TypeContext, dict[NamedId, ContextParam]]:
        subst: dict[NamedId, ContextParam] = {}
        for i, fv in enumerate(sorted(ty.free_vars())):
            t = self.rvars.find(fv)
            match t: 
                case NamedId():
                    subst[fv] = NamedId(f't{i + 1}')
                case _:
                    subst[fv] = t
        ty = ty.subst(subst)
        return ty, subst

    def _unify_contexts(self, a: ContextParam, b: ContextParam) -> ContextParam:
        match a, b:
            case _, NamedId():
                a = self.rvars.add(a)
                return self.rvars.union(a, b)
            case NamedId(), _:
                b = self.rvars.add(b)
                return self.rvars.union(b, a)
            case Context(), Context():
                if not a.is_equiv(b):
                    raise ContextInferError(f'incompatible contexts: {a} != {b}')
                return a
            case _:
                raise RuntimeError(f'unreachable case: {a}, {b}')

    def _unify(self, a_ty: TypeContext, b_ty: TypeContext) -> TypeContext:
        match a_ty, b_ty:
            case VarTypeContext(), VarTypeContext():
                if a_ty != b_ty:
                    raise ContextInferError(f'incompatible types: {a_ty} != {b_ty}')
                return a_ty
            case (BoolTypeContext(), BoolTypeContext()) | (ContextTypeContext(), ContextTypeContext()):
                return a_ty
            case RealTypeContext(), RealTypeContext():
                ctx = self._unify_contexts(a_ty.ctx, b_ty.ctx)
                return RealTypeContext(ctx)
            case TupleTypeContext(), TupleTypeContext():
                assert len(a_ty.elts) == len(b_ty.elts)
                elts = [self._unify(a_elt, b_elt) for a_elt, b_elt in zip(a_ty.elts, b_ty.elts)]
                return TupleTypeContext(*elts)
            case ListTypeContext(), ListTypeContext():
                elt = self._unify(a_ty.elt, b_ty.elt)
                return ListTypeContext(elt)
            case _:
                raise RuntimeError(f'unreachable: {a_ty}, {b_ty}')

    def _cvt_arg_type(self, ty: Type):
        match ty:
            case VarType():
                return VarTypeContext(ty.name)
            case BoolType():
                return BoolTypeContext()
            case RealType():
                return RealTypeContext(self._fresh_context_var())
            case ContextType():
                return ContextTypeContext()
            case TupleType():
                elts = [self._cvt_arg_type(elt) for elt in ty.elts]
                return TupleTypeContext(*elts)
            case ListType():
                elt = self._cvt_arg_type(ty.elt)
                return ListTypeContext(elt)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ty: TypeContext):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self._set_context(d, ty)
            case UnderscoreId():
                pass
            case TupleBinding():
                assert isinstance(ty, TupleTypeContext) and len(ty.elts) == len(target.elts)
                for elt, elt_ctx in zip(target.elts, ty.elts):
                    self._visit_binding(site, elt, elt_ctx)
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_var(self, e: Var, ctx: ContextParam):
        #   x : T \in Γ
        # ---------------
        #  C, Γ |- x : T
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: ContextParam):
        # C, Γ |- e : bool
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, BoolTypeContext) # type checking should have concluded this
        return ty

    def _visit_foreign(self, e: ForeignVal, ctx: ContextParam):
        return self._cvt_arg_type(self._lookup_ty(e))

    def _visit_decnum(self, e: Decnum, ctx: ContextParam):
        # C, Γ |- e : real C
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, RealTypeContext) # type checking should have concluded this
        return ty

    def _visit_hexnum(self, e: Hexnum, ctx: ContextParam):
        # C, Γ |- e : real C
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, RealTypeContext) # type checking should have concluded this
        return ty

    def _visit_integer(self, e: Integer, ctx: ContextParam):
        # C, Γ |- e : real C
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, RealTypeContext) # type checking should have concluded this
        return ty

    def _visit_rational(self, e: Rational, ctx: ContextParam):
        # C, Γ |- e : real C
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, RealTypeContext) # type checking should have concluded this
        return ty

    def _visit_digits(self, e: Digits, ctx: ContextParam):
        # C, Γ |- e : real C
        ty = self._from_scalar(self._lookup_ty(e), ctx)
        assert isinstance(ty, RealTypeContext) # type checking should have concluded this
        return ty

    def _visit_nullaryop(self, e: NullaryOp, ctx: ContextParam):
        #   Γ |- real : T         Γ |- bool : T
        # ----------------      ------------------
        #  C, Γ |- e : real C    C, Γ |- e : bool
        return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_unaryop(self, e: UnaryOp, ctx: ContextParam):
        arg_ty = self._visit_expr(e.arg, ctx)
        match e:
            case Len() | Dim() | Sum():
                # length / dimension / sum
                # C, Γ |- len e : real C
                return RealTypeContext(ctx)
            case Range():
                # range operator
                # C, Γ |- range e : list real C
                return ListTypeContext(RealTypeContext(ctx))
            case Empty():
                # empty operator
                # C, Γ |- empty e : list T
                raise NotImplementedError(e)
            case Enumerate():
                # enumerate operator
                #          C, Γ |- e : list T
                # -----------------------------------------
                #  C, Γ |- enumerate e : list [real C] x T
                assert isinstance(arg_ty, ListTypeContext)
                return ListTypeContext(TupleTypeContext(RealTypeContext(ctx), arg_ty.elt))
            case _:
                #   Γ |- real : T         Γ |- bool : T
                # ----------------      ------------------
                #  C, Γ |- e : real C    C, Γ |- e : bool
                return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: ContextParam):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        match e:
            case Size():
                # size operator
                # C, Γ |- size e : real C
                return RealTypeContext(ctx)
            case _:
                #   Γ |- real : T         Γ |- bool : T
                # ----------------      ------------------
                #  C, Γ |- e : real C    C, Γ |- e : bool
                return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_ternaryop(self, e: TernaryOp, ctx: ContextParam):
        #   Γ |- real : T         Γ |- bool : T
        # ----------------      ------------------
        #  C, Γ |- e : real C    C, Γ |- e : bool
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_naryop(self, e: NaryOp, ctx: ContextParam):
        arg_tys = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case Min() | Max():
                # min / max operator
                #  C, Γ |- e_1 : real C ... C, Γ |- e_n : real C
                # -----------------------------------------------
                #               C, Γ |- e : real C
                ty = arg_tys[0]
                for e_ty in arg_tys[1:]:
                    ty = self._unify(ty, e_ty)
                return ty
            case And() | Or():
                # and / or operator
                # C, Γ |- e : bool
                return BoolTypeContext()
            case Zip():
                # zip operator
                #  C, Γ |- e_1 : list T_1 ... C, Γ |- e_n : list T_n
                # ---------------------------------------------------
                #        C, Γ |- e : list [T_1 x ... x T_n]
                elt_tys = []
                for arg_ty in arg_tys:
                    assert isinstance(arg_ty, ListTypeContext)
                    elt_tys.append(arg_ty.elt)
                return ListTypeContext(TupleTypeContext(*elt_tys))
            case _:
                raise ValueError(f'unknown n-ary operator: {type(e)}')

    def _visit_compare(self, e: Compare, ctx: ContextParam):
        # C, Γ |- e : bool
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return BoolTypeContext()

    def _visit_call(self, e: Call, ctx: ContextParam):
        # get around circular imports
        from ..function import Function

        match e.fn:
            case None:
                # calling None => can't conclude anything
                return self._fresh_context_var()
            case Primitive():
                # calling a primitive => can't conclude anything
                # TODO: annotations to attach context info to primitives
                fn_ty = ContextInfer.primitive(e.fn)
                # instantiate the function context
                fn_ty = cast(FunctionTypeContext, self._instantiate(fn_ty))
                # merge caller context
                self._unify_contexts(ctx, fn_ty.ctx)
                # merge arguments
                if len(fn_ty.args) != len(e.args):
                    raise ContextInferError(
                        f'primitive {e.fn} expects {len(fn_ty.args)} arguments, '
                        f'got {len(e.args)}'
                    )
                for arg, expect_ty in zip(e.args, fn_ty.args):
                    ty = self._visit_expr(arg, ctx)
                    self._unify(ty, expect_ty)

                return fn_ty.ret
            case Function():
                # calling a function
                # TODO: guard against recursion
                fn_info = ContextInfer.infer(e.fn.ast)
                if len(fn_info.arg_types) != len(e.args):
                    raise ContextInferError(
                        f'function {e.fn} expects {len(fn_info.arg_types)} arguments, '
                        f'got {len(e.args)}'
                    )

                # instantiate the function context
                fn_ctx = cast(FunctionTypeContext, self._instantiate(fn_info.func_ty))
                # merge caller context
                self._unify_contexts(ctx, fn_ctx.ctx)
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

    def _visit_tuple_expr(self, e: TupleExpr, ctx: ContextParam):
        #  C, Γ |- e_1 : T_1 ... C, Γ |- e_n : T_n
        # -----------------------------------------
        #        C, Γ |- e : T_1 x ... x T_n
        arg_tys = [self._visit_expr(arg, ctx) for arg in e.elts]
        return TupleTypeContext(*arg_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: ContextParam):
        #  C, Γ |- e_1 : T ... C, Γ |- e_n : T
        # -------------------------------------
        #         C, Γ |- e : list T
        if len(e.elts) == 0:
            raise NotImplementedError
        else:
            # type checking ensures the base type is the same
            elts = [self._visit_expr(arg, ctx) for arg in e.elts]
            return ListTypeContext(elts[0])

    def _visit_list_comp(self, e: ListComp, ctx: ContextParam):
        #       C, Γ |- elt: T
        # --------------------------
        #  C, Γ |- for ... : list T
        for target, iterable in zip(e.targets, e.iterables):
            iterable_ty = self._visit_expr(iterable, ctx)
            assert isinstance(iterable_ty, ListTypeContext)
            self._visit_binding(e, target, iterable_ty.elt)
        ty = self._visit_expr(e.elt, ctx)
        return ListTypeContext(ty)

    def _visit_list_ref(self, e: ListRef, ctx: ContextParam):
        #      C, Γ |- e: list T       C, Γ |- i : real C
        # -----------------------------------------------
        #            C, Γ |- e[i] : T
        ty = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(ty, ListTypeContext)
        return ty.elt

    def _visit_list_slice(self, e: ListSlice, ctx: ContextParam):
        #      C, Γ |- e: list T       C, Γ |- i,j : real C
        # -----------------------------------------------
        #           C, Γ |- e[i:j] : list T
        ty = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        return ty

    def _visit_list_set(self, e: ListSet, ctx: ContextParam):
        #      C, Γ |- e: list T       C, Γ |- i_1,...,i_n : real C
        # -----------------------------------------------
        #        C, Γ |- set(e, (i_1,...,i_n), v) : list T
        ty = self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.expr, ctx)
        return ty

    def _visit_if_expr(self, e: IfExpr, ctx: ContextParam):
        #     C, Γ |- cond: bool       C, Γ |- ift: T       C, Γ |- iff: T
        # -------------------------------------------------------
        #                 C, Γ |- e : T
        self._visit_expr(e.cond, ctx)
        ift_ty = self._visit_expr(e.ift, ctx)
        iff_ty = self._visit_expr(e.iff, ctx)
        return self._unify(ift_ty, iff_ty)

    def _visit_attribute(self, e: Attribute, ctx: ContextParam):
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, ctx: ContextParam):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: ContextParam):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: ContextParam):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            ty = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ty)
        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: ContextParam):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # need to merge variables introduced on both sides
        for phi in self.def_use.phis[stmt]:
            ty = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ty)

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: ContextParam):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            ty = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ty)
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: ContextParam):
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, ListTypeContext)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            ty = self._unify(self.by_def[phi.lhs], self.by_def[phi.rhs])
            self._set_context(phi, ty)

        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: ContextParam):
        if not isinstance(stmt.ctx, ForeignVal) or not isinstance(stmt.ctx.val, Context):
            raise ContextInferError(f'cannot infer context for `{stmt.ctx}`')

        body_ctx = stmt.ctx.val
        self._visit_block(stmt.body, body_ctx)
        return ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: ContextParam):
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: ContextParam):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: ContextParam):
        self.ret_ty = self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: ContextParam):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # function can have an overriding context
        match func.ctx:
            case None:
                body_ctx: ContextParam = self._fresh_context_var()
            case FPCoreContext():
                body_ctx = func.ctx.to_context()
            case _:
                body_ctx = func.ctx

        # generate context variables for each argument
        arg_types: list[TypeContext] = []
        for arg, ty in zip(func.args, self.type_info.arg_types):
            arg_ty = self._cvt_arg_type(ty)
            arg_types.append(arg_ty)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_context(d, arg_ty)

        # generate context variables for each free variables
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            ty = self._cvt_arg_type(self.type_info.by_def[d])
            self._set_context(d, ty)

        # visit body
        self._visit_block(func.body, body_ctx)
        assert self.ret_ty is not None # function has no return statement

        # generalize the function context
        arg_types = [self._resolve(ty) for ty in arg_types]
        ret_ty = self._resolve(self.ret_ty)
        return FunctionTypeContext(body_ctx, arg_types, ret_ty)

    def _visit_expr(self, expr: Expr, ctx: ContextParam) -> TypeContext:
        ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ty
        return ty

    def infer(self):
        # context inference on body
        ctx = self._visit_function(self.func, None)

        # generalize the output context
        fn_ctx, subst = self._generalize(ctx)
        fn_ctx = cast(FunctionTypeContext, fn_ctx)

        # rename unbound context variables
        for t in self.rvars:
            if isinstance(t, NamedId) and t not in subst:
                subst[t] = NamedId(f'r{len(subst) + 1}')

        # resolve definition/expr contexts
        by_defs = {
            d: self._resolve(ctx).subst(subst)
            for d, ctx in self.by_def.items()
        }
        by_expr = {
            e: self._resolve(ctx).subst(subst)
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

    def _default_type(self, ty: TypeAnn) -> TypeContext:
        match ty:
            case None | AnyTypeAnn():
                return VarTypeContext(self.gensym.fresh('t'))
            case BoolTypeAnn():
                return BoolTypeContext()
            case RealTypeAnn():
                return RealTypeContext(self.gensym.fresh('r'))
            case TupleTypeAnn():
                elts = [self._default_type(t) for t in ty.elts]
                return TupleTypeContext(*elts)
            case ListTypeAnn():
                elt = self._default_type(ty.elt)
                return ListTypeContext(elt)
            case _:
                raise RuntimeError(f'unknown type: {ty}')

    # def _arg_ctx(self, ty: TypeAnn, ctx: str | tuple) -> TypeContext:
    #     match ty:
    #         case None | RealTypeAnn() | BoolTypeAnn() | AnyTypeAnn():
    #             if not isinstance(ctx, str):
    #                 raise ValueError(f"expected context variable for argument of type {ty}, got {ctx}")
    #             if ctx not in self.subst:
    #                 self.subst[ctx] = self.gensym.fresh('r')
    #             return self.subst[ctx]
    #         case TupleTypeAnn():
    #             if not isinstance(ctx, tuple):
    #                 raise ValueError(f"expected tuple context for argument of type {ty}, got {ctx}")
    #             if len(ty.elts) != len(ctx):
    #                 raise ValueError(f"tuple context length mismatch: expected {len(ty.elts)}, got {len(ctx)}")
    #             elts = [self._arg_ctx(t, c) for t, c in zip(ty.elts, ctx)]
    #             return TupleTypeContext(elts)
    #         case ListTypeAnn():
    #             elt = self._arg_ctx(ty.elt, ctx)
    #             return ListTypeContext(elt)
    #         case _:
    #             raise RuntimeError(f'unknown type: {ty}')

    # def _cvt_return_ctx(self, ctx: Context | str | tuple) -> TypeContext:
    #     match ctx:
    #         case str():
    #             if ctx not in self.subst:
    #                 raise ContextInferError(f'unbound context variable in ret_ctx: {ctx}')
    #             return self.subst[ctx]
    #         case Context():
    #             return ctx
    #         case tuple():
    #             elts = [self._cvt_return_ctx(c) for c in ctx]
    #             return TupleTypeContext(elts)
    #         case _:
    #             raise ValueError(f"invalid context in ret_ctx: {ctx}")

    # def _default_return_ctx(self, ty: TypeAnn) -> TypeContext:
    #     match ty:
    #         case None | RealTypeAnn() | BoolTypeAnn() | AnyTypeAnn():
    #             return self.gensym.fresh('r')
    #         case TupleTypeAnn():
    #             elts = [self._default_return_ctx(t) for t in ty.elts]
    #             return TupleContext(elts)
    #         case ListTypeAnn():
    #             return self._default_return_ctx(ty.elt)
    #         case _:
    #             raise RuntimeError(f'unknown type: {ty}')

    def infer(self) -> FunctionTypeContext:
        # interpret primitive context
        ctx = self.gensym.fresh('r')
        if self.prim.ctx is not None:
            self.subst[self.prim.ctx] = ctx
        raise NotImplementedError(self.prim)

        # interpret argument contexts
        # arg_types: list[TypeContext] = []
        # if self.prim.arg_ctxs is None:
        #     for _ in self.prim.arg_types:
        #         arg_types.append(self.gensym.fresh('r'))
        # else:
        #     # none specified
        #     assert len(self.prim.arg_ctxs) == len(self.prim.arg_types)
        #     for ty, arg_ctx in zip(self.prim.arg_types, self.prim.arg_ctxs):
        #         arg_types.append(self._arg_ctx(ty, arg_ctx))

        # # interpret return context
        # if self.prim.ret_ctx is None:
        #     ret_ctx = self.gensym.fresh('r')
        # else:
        #     ret_ctx = self._cvt_return_ctx(self.prim.ret_ctx)

        # return FunctionTypeContext(ctx, arg_types, ret_ctx)

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

        type_info = TypeCheck.check(func, def_use)
        inst = ContextTypeInferInstance(func, def_use, type_info)
        return inst.infer()

    @staticmethod
    def primitive(prim: Primitive) -> FunctionTypeContext:
        """
        Infers the context of a primitive.
        """
        if not isinstance(prim, Primitive):
            raise TypeError(f'expected a \'Primitive\', got {prim}')
        return ContextTypeInferPrimitive(prim).infer()
