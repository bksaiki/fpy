"""
Context inference.

TODO: unification for context variables
"""

from dataclasses import dataclass
from typing import cast

from ..ast import *
from ..fpc_context import FPCoreContext
from ..number import Context, INTEGER, REAL
from ..primitive import Primitive
from ..utils import Gensym, NamedId, Unionfind

from ..types import *
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite
from .type_infer import TypeInfer, TypeAnalysis

class ContextInferError(Exception):
    """Context inference error for FPy programs."""
    pass

@dataclass(frozen=True)
class ContextAnalysis:
    fn_type: FunctionType
    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]
    at_expr: dict[Expr, ContextParam]

    @property
    def body_ctx(self):
        return self.fn_type.ctx

    @property
    def arg_types(self):
        return self.fn_type.arg_types

    @property
    def return_type(self):
        return self.fn_type.return_type


class ContextTypeInferInstance(Visitor):
    """
    Context inference instance.

    This visitor traverses the function and infers rounding contexts
    for each definition site.
    """

    func: FuncDef
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis
    unsafe_cast_int: bool

    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]
    at_expr: dict[Expr, ContextParam]
    ret_ty: Type | None
    rvars: Unionfind[ContextParam]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        type_info: TypeAnalysis,
        unsafe_cast_int: bool
    ):
        self.func = func
        self.def_use = def_use
        self.type_info = type_info
        self.unsafe_cast_int = unsafe_cast_int
        self.by_def = {}
        self.by_expr = {}
        self.at_expr = {}
        self.ret_ty = None
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def _lookup_ty(self, e: Expr):
        return self.type_info.by_expr[e]

    def _from_scalar(self, ty: Type, ctx: ContextParam):
        match ty:
            case BoolType():
                return BoolType()
            case RealType():
                return RealType(ctx)
            case ContextType():
                return ContextType()
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _set_context(self, site: Definition, ty: Type):
        self.by_def[site] = ty

    def _fresh_context_var(self) -> NamedId:
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _resolve_context(self, ctx: ContextParam) -> ContextParam:
        return self.rvars.get(ctx, ctx)

    def _resolve(self, ty: Type) -> Type:
        match ty:
            case BoolType() | ContextType() | VarType():
                return ty
            case RealType():
                assert ty.ctx is not None
                ctx = self._resolve_context(ty.ctx)
                return RealType(ctx)
            case TupleType():
                elts = (self._resolve(elt) for elt in ty.elts)
                return TupleType(*elts)
            case ListType():
                elt = self._resolve(ty.elt)
                return ListType(elt)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _instantiate(self, ty: Type) -> Type:
        subst: dict[NamedId, ContextParam] = {}
        for fv in sorted(ty.free_context_vars()):
            subst[fv] = self._fresh_context_var()
        return ty.subst_context(subst)

    def _generalize(self, ty: Type) -> tuple[Type, dict[NamedId, ContextParam]]:
        subst: dict[NamedId, ContextParam] = {}
        for i, fv in enumerate(sorted(ty.free_context_vars())):
            t = self.rvars.find(fv)
            match t: 
                case NamedId():
                    subst[fv] = NamedId(f'r{i + 1}')
                case _:
                    subst[fv] = t
        ty = ty.subst_context(subst)
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

    def _unify(self, a_ty: Type, b_ty: Type) -> Type:
        match a_ty, b_ty:
            case VarType(), VarType():
                if a_ty != b_ty:
                    raise ContextInferError(f'incompatible types: {a_ty} != {b_ty}')
                return a_ty
            case (BoolType(), BoolType()) | (ContextType(), ContextType()):
                return a_ty
            case RealType(), RealType():
                assert a_ty.ctx is not None and b_ty.ctx is not None
                ctx = self._unify_contexts(a_ty.ctx, b_ty.ctx)
                return RealType(ctx)
            case TupleType(), TupleType():
                assert len(a_ty.elts) == len(b_ty.elts)
                elts = [self._unify(a_elt, b_elt) for a_elt, b_elt in zip(a_ty.elts, b_ty.elts)]
                return TupleType(*elts)
            case ListType(), ListType():
                elt = self._unify(a_ty.elt, b_ty.elt)
                return ListType(elt)
            case _:
                raise RuntimeError(f'unreachable: {a_ty}, {b_ty}')

    def _cvt_type(self, ty: Type) -> Type:
        match ty:
            case VarType():
                return VarType(ty.name)
            case BoolType():
                return BoolType()
            case RealType():
                return RealType(self._fresh_context_var())
            case ContextType():
                return ContextType()
            case TupleType():
                elts = [self._cvt_type(elt) for elt in ty.elts]
                return TupleType(*elts)
            case ListType():
                elt = self._cvt_type(ty.elt)
                return ListType(elt)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _cvt_arg_type(self, ty: Type, ann: TypeAnn):
        match ty:
            case VarType():
                return VarType(ty.name)
            case BoolType():
                return BoolType()
            case RealType():
                if isinstance(ann, RealTypeAnn) and ann.ctx is not None:
                    return RealType(ann.ctx)
                else:
                    return RealType(self._fresh_context_var())
            case ContextType():
                return ContextType()
            case TupleType():
                if isinstance(ann, TupleTypeAnn):
                    assert len(ann.elts) == len(ty.elts)
                    elts = [self._cvt_arg_type(elt, ann) for elt, ann in zip(ty.elts, ann.elts)]
                    return TupleType(*elts)
                else:
                    elts = [self._cvt_arg_type(elt, AnyTypeAnn(None)) for elt in ty.elts]
                    return TupleType(*elts)
            case ListType():
                if isinstance(ann, ListTypeAnn):
                    elt = self._cvt_arg_type(ty.elt, ann.elt)
                    return ListType(elt)
                else:
                    elt = self._cvt_arg_type(ty.elt, AnyTypeAnn(None))
                    return ListType(elt)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ty: Type):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self._set_context(d, ty)
            case UnderscoreId():
                pass
            case TupleBinding():
                assert isinstance(ty, TupleType) and len(ty.elts) == len(target.elts)
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
        assert isinstance(ty, BoolType) # type checking should have concluded this
        return ty

    def _visit_foreign(self, e: ForeignVal, ctx: ContextParam):
        return self._cvt_type(self._lookup_ty(e))

    def _visit_decnum(self, e: Decnum, ctx: ContextParam):
        if self.unsafe_cast_int and e.is_integer():
            # unsafely cast to integer
            # C, Γ |- e : real INTEGER
            return RealType(INTEGER)
        else:
            # C, Γ |- e : real REAL
            ty = self._from_scalar(self._lookup_ty(e), REAL)
            assert isinstance(ty, RealType) # type checking should have concluded this
            return ty

    def _visit_hexnum(self, e: Hexnum, ctx: ContextParam):
        if self.unsafe_cast_int and e.is_integer():
            # unsafely cast to integer
            # C, Γ |- e : real INTEGER
            return RealType(INTEGER)
        else:
            # C, Γ |- e : real REAL
            ty = self._from_scalar(self._lookup_ty(e), REAL)
            assert isinstance(ty, RealType) # type checking should have concluded this
            return ty

    def _visit_integer(self, e: Integer, ctx: ContextParam):
        if self.unsafe_cast_int and e.is_integer():
            # unsafely cast to integer
            # C, Γ |- e : real INTEGER
            return RealType(INTEGER)
        else:
            # C, Γ |- e : real REAL
            ty = self._from_scalar(self._lookup_ty(e), REAL)
            assert isinstance(ty, RealType) # type checking should have concluded this
            return ty

    def _visit_rational(self, e: Rational, ctx: ContextParam):
        if self.unsafe_cast_int and e.is_integer():
            # unsafely cast to integer
            # C, Γ |- e : real INTEGER
            return RealType(INTEGER)
        else:
            # C, Γ |- e : real REAL
            ty = self._from_scalar(self._lookup_ty(e), REAL)
            assert isinstance(ty, RealType) # type checking should have concluded this
            return ty

    def _visit_digits(self, e: Digits, ctx: ContextParam):
        if self.unsafe_cast_int and e.is_integer():
            # unsafely cast to integer
            # C, Γ |- e : real INTEGER
            return RealType(INTEGER)
        else:
            # C, Γ |- e : real REAL
            ty = self._from_scalar(self._lookup_ty(e), REAL)
            assert isinstance(ty, RealType) # type checking should have concluded this
            return ty

    def _visit_nullaryop(self, e: NullaryOp, ctx: ContextParam):
        #   Γ |- real : T         Γ |- bool : T
        # ----------------      ------------------
        #  C, Γ |- e : real C    C, Γ |- e : bool
        return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_unaryop(self, e: UnaryOp, ctx: ContextParam):
        arg_ty = self._visit_expr(e.arg, ctx)
        match e:
            case Len() | Dim():
                # length / dimension
                # C, Γ |- len e : real INTEGER
                return RealType(INTEGER)
            case Sum():
                # sum operator
                # C, Γ |- sum e : real C
                return RealType(ctx)
            case Range1():
                # range operator
                # C, Γ |- range e : list (real INTEGER)
                return ListType(RealType(INTEGER))
            case Empty():
                # empty operator
                # C, Γ |- empty e : list T
                return self._cvt_type(self._lookup_ty(e))
            case Enumerate():
                # enumerate operator
                #          C, Γ |- e : list T
                # -----------------------------------------
                #  C, Γ |- enumerate e : list [real INTEGER] x T
                assert isinstance(arg_ty, ListType)
                elt_ty = TupleType(RealType(INTEGER), arg_ty.elt)
                return ListType(elt_ty)
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
                # C, Γ |- size e : real INTEGER
                return RealType(INTEGER)
            case Range2():
                # range operator
                # C, Γ |- range e1 e2 : list (real INTEGER)
                return ListType(RealType(INTEGER))
            case _:
                #   Γ |- real : T         Γ |- bool : T
                # ----------------      ------------------
                #  C, Γ |- e : real C    C, Γ |- e : bool
                return self._from_scalar(self._lookup_ty(e), ctx)

    def _visit_ternaryop(self, e: TernaryOp, ctx: ContextParam):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        match e:
            case Range3():
                # range operator
                # C, Γ |- range e1 e2 e3 : list (real INTEGER)
                return ListType(RealType(INTEGER))
            case _:
                #   Γ |- real : T         Γ |- bool : T
                # ----------------      ------------------
                #  C, Γ |- e : real C    C, Γ |- e : bool
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
                return BoolType()
            case Zip():
                # zip operator
                #  C, Γ |- e_1 : list T_1 ... C, Γ |- e_n : list T_n
                # ---------------------------------------------------
                #        C, Γ |- e : list [T_1 x ... x T_n]
                elt_tys = []
                for arg_ty in arg_tys:
                    assert isinstance(arg_ty, ListType)
                    elt_tys.append(arg_ty.elt)
                return ListType(TupleType(*elt_tys))
            case _:
                raise ValueError(f'unknown n-ary operator: {type(e)}')

    def _visit_compare(self, e: Compare, ctx: ContextParam):
        # C, Γ |- e : bool
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return BoolType()

    def _visit_call(self, e: Call, ctx: ContextParam):
        # get around circular imports
        from ..function import Function

        match e.fn:
            case None:
                # calling None => can't conclude anything
                ty = self._cvt_type(self._lookup_ty(e))
                return ty
            case Primitive():
                # calling a primitive => can't conclude anything
                fn_ty = ContextInfer.infer_primitive(e.fn)
                # instantiate the function context
                fn_ty = cast(FunctionType, self._instantiate(fn_ty))
                # merge caller context
                assert fn_ty.ctx is not None
                self._unify_contexts(ctx, fn_ty.ctx)
                # merge arguments
                if len(fn_ty.arg_types) != len(e.args):
                    raise ContextInferError(
                        f'primitive {e.fn} expects {len(fn_ty.arg_types)} arguments, '
                        f'got {len(e.args)}'
                    )
                for arg, expect_ty in zip(e.args, fn_ty.arg_types):
                    ty = self._visit_expr(arg, ctx)
                    self._unify(ty, expect_ty)

                return fn_ty.return_type
            case Function():
                # calling a function
                # TODO: guard against recursion
                from ..transform import ConstFold, Monomorphize

                # get argument types
                arg_types = [self.type_info.by_expr[arg] for arg in e.args]

                # TODO: is there a better way to do this? bad complexity
                # apply constant folding and monomorphization
                ast = ConstFold.apply(e.fn.ast, enable_op=False)
                ast = Monomorphize.apply_by_arg(ast, ctx if isinstance(ctx, Context) else None, arg_types)
                fn_info = ContextInfer.infer(ast)

                # instantiate the function context
                fn_type = cast(FunctionType, self._instantiate(fn_info.fn_type))

                # merge caller context
                assert fn_type.ctx is not None
                self._unify_contexts(ctx, fn_type.ctx)

                # merge arguments
                for arg, expect_ty in zip(e.args, fn_type.arg_types):
                    ty = self._visit_expr(arg, ctx)
                    self._unify(ty, expect_ty)

                return fn_type.return_type
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
        return TupleType(*arg_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: ContextParam):
        #  C, Γ |- e_1 : T ... C, Γ |- e_n : T
        # -------------------------------------
        #         C, Γ |- e : list T
        if len(e.elts) == 0:
            return self._cvt_type(self._lookup_ty(e))
        else:
            # type checking ensures the base type is the same
            elts = [self._visit_expr(arg, ctx) for arg in e.elts]
            return ListType(elts[0])

    def _visit_list_comp(self, e: ListComp, ctx: ContextParam):
        #       C, Γ |- elt: T
        # --------------------------
        #  C, Γ |- for ... : list T
        for target, iterable in zip(e.targets, e.iterables):
            iterable_ty = self._visit_expr(iterable, ctx)
            assert isinstance(iterable_ty, ListType)
            self._visit_binding(e, target, iterable_ty.elt)
        ty = self._visit_expr(e.elt, ctx)
        return ListType(ty)

    def _visit_list_ref(self, e: ListRef, ctx: ContextParam):
        #      C, Γ |- e: list T       C, Γ |- i : real C
        # -----------------------------------------------
        #            C, Γ |- e[i] : T
        ty = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(ty, ListType)
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

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            ty = self._unify(lhs_ty, rhs_ty)
            self._set_context(phi, ty)

        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: ContextParam):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            ty = self._unify(lhs_ty, rhs_ty)
            self._set_context(phi, ty)

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: ContextParam):
        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self._set_context(phi, lhs_ty)

        # visit condition and body
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # unify phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self._unify(lhs_ty, rhs_ty)

        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: ContextParam):
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, ListType)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self._set_context(phi, lhs_ty)

        self._visit_block(stmt.body, ctx)

        # unify phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self._unify(lhs_ty, rhs_ty)

        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: ContextParam):
        if not isinstance(stmt.ctx, ForeignVal) or not isinstance(stmt.ctx.val, Context):
            raise ContextInferError(f'cannot infer context for `{stmt.ctx.format()}` at `{stmt.format()}`')
        body_ctx = stmt.ctx.val

        # interpreted under a real rounding context
        # REAL, Γ |- ctx : context
        ctx_ty = self._visit_expr(stmt.ctx, REAL)
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self._set_context(d, ctx_ty)
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

    def _visit_pass(self, stmt: PassStmt, ctx: ContextParam):
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
        arg_types: list[Type] = []
        for arg, ty in zip(func.args, self.type_info.arg_types):
            arg_ty = self._cvt_arg_type(ty, arg.type)
            arg_types.append(arg_ty)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_context(d, arg_ty)

        # generate context variables for each free variables
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            ty = self._cvt_type(self.type_info.by_def[d])
            self._set_context(d, ty)

        # visit body
        self._visit_block(func.body, body_ctx)
        assert isinstance(self.ret_ty, Type) # function has no return statement

        # generalize the function context
        arg_types = [self._resolve(ty) for ty in arg_types]
        ret_ty = self._resolve(self.ret_ty)
        return FunctionType(body_ctx, arg_types, ret_ty)

    def _visit_expr(self, expr: Expr, ctx: ContextParam) -> Type:
        ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ty
        self.at_expr[expr] = ctx
        return ty

    def infer(self):
        # context inference on body
        ctx = self._visit_function(self.func, None)

        # generalize the output context
        fn_ctx, subst = self._generalize(ctx)
        fn_ctx = cast(FunctionType, fn_ctx)

        # rename unbound context variables
        for t in self.rvars:
            if isinstance(t, NamedId) and t not in subst:
                subst[t] = NamedId(f'r{len(subst) + 1}')

        # resolve definition/expr contexts
        by_defs = {
            d: self._resolve(ctx).subst_context(subst)
            for d, ctx in self.by_def.items()
        }
        by_expr = {
            e: self._resolve(ctx).subst_context(subst)
            for e, ctx in self.by_expr.items()
        }
        at_expr = {
            e: self._resolve_context(ctx)
            for e, ctx in self.at_expr.items()
        }

        return ContextAnalysis(fn_ctx, by_defs, by_expr, at_expr)


class _ContextInferPrimitive:
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

    def _fresh_context_var(self) -> NamedId:
        return self.gensym.fresh('r')

    def _cvt_arg_type(self, ty: Type, ctx: str | tuple | None) -> Type:
        match ty:
            case VarType():
                return VarType(ty.name)
            case BoolType():
                return BoolType()
            case RealType():
                if ctx is None:
                    return RealType(self._fresh_context_var())
                else:
                    if not isinstance(ctx, str):
                        raise ValueError(f"expected context variable for argument of type {ty}, got {ctx}")
                    if ctx not in self.subst:
                        self.subst[ctx] = self._fresh_context_var()
                    return RealType(self.subst[ctx])
            case ContextType():
                return ContextType()
            case TupleType():
                if ctx is None:
                    elts = [self._cvt_arg_type(t, None) for t in ty.elts]
                    return TupleType(*elts)
                else:
                    if not isinstance(ctx, tuple):
                        raise ValueError(f"expected tuple context for argument of type {ty}, got {ctx}")
                    if len(ty.elts) != len(ctx):
                        raise ValueError(f"tuple context length mismatch: expected {len(ty.elts)}, got {len(ctx)}")
                    elts = [self._cvt_arg_type(t, c) for t, c in zip(ty.elts, ctx)]
                    return TupleType(*elts)
            case ListType():
                if ctx is None:
                    elt = self._cvt_arg_type(ty.elt, None)
                    return ListType(elt)
                else:
                    return ListType(self._cvt_arg_type(ty.elt, ctx))
            case _:
                raise RuntimeError(f'unknown type: {ty}')

    def _cvt_ret_type(self, ty: Type, ctx: Context | str | tuple | None) -> Type:
        match ty:
            case VarType():
                return VarType(ty.name)
            case BoolType():
                return BoolType()
            case RealType():
                if ctx is None:
                    return RealType(self._fresh_context_var())
                elif isinstance(ctx, Context):
                    return RealType(ctx)
                else:
                    if not isinstance(ctx, str):
                        raise ValueError(f"expected context variable for return of type {ty}, got {ctx}")
                    if ctx not in self.subst:
                        raise ValueError(f"unbound context variable '{ctx}' in return type")
                    return RealType(self.subst[ctx])
            case ContextType():
                return ContextType()
            case TupleType():
                if ctx is None:
                    elts = [self._cvt_ret_type(t, None) for t in ty.elts]
                    return TupleType(*elts)
                else:
                    if not isinstance(ctx, tuple):
                        raise ValueError(f"expected tuple context for return of type {ty}, got {ctx}")
                    if len(ty.elts) != len(ctx):
                        raise ValueError(f"tuple context length mismatch: expected {len(ty.elts)}, got {len(ctx)}")
                    elts = [self._cvt_ret_type(t, c) for t, c in zip(ty.elts, ctx)]
                    return TupleType(*elts)
            case ListType():
                if ctx is None:
                    elt = self._cvt_ret_type(ty.elt, None)
                    return ListType(elt)
                else:
                    return ListType(self._cvt_ret_type(ty.elt, ctx)) 
            case _:
                raise RuntimeError(f'unknown type: {ty}')

    def infer(self) -> FunctionType:
        # perform standard type inference
        fn_ty = TypeInfer.infer_primitive(self.prim)

        # interpret primitive context
        ctx = self._fresh_context_var()
        if self.prim.ctx is not None:
            # map specified name to generated name
            self.subst[self.prim.ctx] = ctx

        # interpret argument contexts
        if self.prim.arg_ctxs is None:
            arg_types = [self._cvt_arg_type(ty, None) for ty in fn_ty.arg_types]
        else:
            assert len(self.prim.arg_ctxs) == len(fn_ty.arg_types)
            arg_types = [self._cvt_arg_type(ty, ctx) for ty, ctx in zip(fn_ty.arg_types, self.prim.arg_ctxs)]

        # interpret return context
        ret_type = self._cvt_ret_type(fn_ty.return_type, self.prim.ret_ctx)

        return FunctionType(ctx, arg_types, ret_type)


###########################################################
# Context inference

class ContextInfer:
    """
    Context inference.

    This is just type checking extended with a static analysis
    to infer rounding contexts for every real-valued expression.
    The analysis assigns every statement, definition, and expression
    a rounding context if it can be determined.
    """

    @staticmethod
    def infer(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        unsafe_cast_int: bool = False
    ):
        """
        Performs rounding context inference.
        Produces a map from definition sites to their rounding contexts.

        Raises `ContextInferError` if the context cannot be inferred.

        Optional arguments:
        - `def_use`: pre-computed define-use analysis 
        - `unsafe_cast_int`: allow unrounded integers to be typed
        as an integer rather than real [default: `False`]
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')
        if not isinstance(unsafe_cast_int, bool):
            raise TypeError(f'expected a \'bool\' for unsafe_cast_int, got {unsafe_cast_int}')

        if def_use is None:
            def_use = DefineUse.analyze(func)

        type_info = TypeInfer.check(func, def_use)
        inst = ContextTypeInferInstance(func, def_use, type_info, unsafe_cast_int)
        return inst.infer()

    @staticmethod
    def infer_primitive(prim: Primitive) -> FunctionType:
        """
        Infers the context of a primitive.
        """
        if not isinstance(prim, Primitive):
            raise TypeError(f'expected a \'Primitive\', got {prim}')
        return _ContextInferPrimitive(prim).infer()
