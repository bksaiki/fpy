"""
FPy parsing from FPCore.
"""

from typing import TypeAlias

import titanfp.fpbench.fpcast as fpc
from titanfp.fpbench.fpcparser import data_as_expr

from .codegen import IRCodegen
from .definition import DefinitionAnalysis
from .fpyast import *
from .live_vars import LiveVarAnalysis
from .syntax_check import SyntaxCheck

from ..passes import SSA, VerifyIR
from ..utils import Gensym, pythonize_id

DataElt: TypeAlias = tuple['DataElt'] | fpc.ValueExpr

_unary_table = {
    'neg': UnaryOpKind.NEG,
    'not': UnaryOpKind.NOT,
    'fabs': UnaryOpKind.FABS,
    'sqrt': UnaryOpKind.SQRT,
    'cbrt': UnaryOpKind.CBRT,
    'ceil': UnaryOpKind.CEIL,
    'floor': UnaryOpKind.FLOOR,
    'nearbyint': UnaryOpKind.NEARBYINT,
    'round': UnaryOpKind.ROUND,
    'trunc': UnaryOpKind.TRUNC,
    'acos': UnaryOpKind.ACOS,
    'asin': UnaryOpKind.ASIN,
    'atan': UnaryOpKind.ATAN,
    'cos': UnaryOpKind.COS,
    'sin': UnaryOpKind.SIN,
    'tan': UnaryOpKind.TAN,
    'acosh': UnaryOpKind.ACOSH,
    'asinh': UnaryOpKind.ASINH,
    'atanh': UnaryOpKind.ATANH,
    'cosh': UnaryOpKind.COSH,
    'sinh': UnaryOpKind.SINH,
    'tanh': UnaryOpKind.TANH,
    'exp': UnaryOpKind.EXP,
    'exp2': UnaryOpKind.EXP2,
    'expm1': UnaryOpKind.EXPM1,
    'log': UnaryOpKind.LOG,
    'log10': UnaryOpKind.LOG10,
    'log1p': UnaryOpKind.LOG1P,
    'log2': UnaryOpKind.LOG2,
    'erf': UnaryOpKind.ERF,
    'erfc': UnaryOpKind.ERFC,
    'lgamma': UnaryOpKind.LGAMMA,
    'tgamma': UnaryOpKind.TGAMMA,
    'isfinite': UnaryOpKind.ISFINITE,
    'isinf': UnaryOpKind.ISINF,
    'isnan': UnaryOpKind.ISNAN,
    'isnormal': UnaryOpKind.ISNORMAL,
    'signbit': UnaryOpKind.SIGNBIT,
    'cast': UnaryOpKind.CAST,
    'range': UnaryOpKind.RANGE,
    'dim': UnaryOpKind.DIM,
}

_binary_table = {
    '+': BinaryOpKind.ADD,
    '-': BinaryOpKind.SUB,
    '*': BinaryOpKind.MUL,
    '/': BinaryOpKind.DIV,
    'copysign': BinaryOpKind.COPYSIGN,
    'fdim': BinaryOpKind.FDIM,
    'fmax': BinaryOpKind.FMAX,
    'fmin': BinaryOpKind.FMIN,
    'fmod': BinaryOpKind.FMOD,
    'remainder': BinaryOpKind.REMAINDER,
    'hypot': BinaryOpKind.HYPOT,
    'atan2': BinaryOpKind.ATAN2,
    'pow': BinaryOpKind.POW,
}

_ternary_table = {
    'fma': TernaryOpKind.FMA
}

def _zeros(ns: list[Expr]) -> Expr:
    vars = [UnderscoreId() for _ in ns]
    args = [UnaryOp(UnaryOpKind.RANGE, n, None) for n in ns]
    return CompExpr(vars, args, Integer(0, None), None)

class _Ctx:
    env: dict[str, NamedId]
    stmts: list[Stmt]

    def __init__(
        self,
        env: Optional[dict[str, NamedId]] = None,
        stmts: Optional[list[Stmt]] = None
    ):
        if env is None:
            self.env = {}
        else:
            self.env = env

        if stmts is None:
            self.stmts = []
        else:
            self.stmts = stmts


    def without_stmts(self):
        ctx = _Ctx()
        ctx.env = dict(self.env)
        return ctx


class _FPCore2FPy:
    """Compiler from FPCore to the FPy AST."""
    core: fpc.FPCore
    gensym: Gensym
    default_name: str

    def __init__(self, core: fpc.FPCore, default_name: str):
        self.core = core
        self.gensym = Gensym(rename_hook=pythonize_id)
        self.default_name = default_name

    def _visit_var(self, e: fpc.Var, ctx: _Ctx) -> Expr:
        if e.value not in ctx.env:
            raise ValueError(f'variable {e.value} not in scope')
        return Var(ctx.env[e.value], None)

    def _visit_decnum(self, e: fpc.Decnum, ctx: _Ctx) -> Expr:
        return Decnum(str(e.value), None)

    def _visit_hexnum(self, e: fpc.Hexnum, ctx: _Ctx) -> Expr:
        return Hexnum(str(e.value), None)

    def _visit_integer(self, e: fpc.Integer, ctx: _Ctx) -> Expr:
        return Integer(e.value, None)

    def _visit_rational(self, e: fpc.Rational, ctx: _Ctx) -> Expr:
        return Rational(e.p, e.q, None)

    def _visit_digits(self, e: fpc.Digits, ctx: _Ctx) -> Expr:
        return Digits(e.m, e.e, e.b, None)

    def _visit_constant(self, e: fpc.Constant, ctx: _Ctx) -> Expr:
        match e.value:
            case 'TRUE':
                return Bool(True, None)
            case 'FALSE':
                return Bool(False, None)
            case _:
                return Constant(str(e.value), None)

    def _visit_unary(self, e: fpc.UnaryExpr, ctx: _Ctx) -> Expr:
        if e.name == '-':
            kind = _unary_table['neg']
            arg = self._visit(e.children[0], ctx)
            return UnaryOp(kind, arg, None)
        elif e.name in _unary_table:
            kind = _unary_table[e.name]
            arg = self._visit(e.children[0], ctx)
            return UnaryOp(kind, arg, None)
        else:
            raise NotImplementedError(f'unsupported unary operation {e.name}')

    def _visit_binary(self, e: fpc.BinaryExpr, ctx: _Ctx) -> Expr:
        if e.name in _binary_table:
            kind = _binary_table[e.name]
            left = self._visit(e.children[0], ctx)
            right = self._visit(e.children[1], ctx)
            return BinaryOp(kind, left, right, None)
        else:
            raise NotImplementedError(f'unsupported binary operation {e.name}')

    def _visit_ternary(self, e: fpc.TernaryExpr, ctx: _Ctx) -> Expr:
        if e.name in _ternary_table:
            kind = _ternary_table[e.name]
            arg0 = self._visit(e.children[0], ctx)
            arg1 = self._visit(e.children[1], ctx)
            arg2 = self._visit(e.children[2], ctx)
            return TernaryOp(kind, arg0, arg1, arg2, None)
        else:
            raise NotImplementedError(f'unsupported ternary operation {e.name}')

    def _visit_nary(self, e: fpc.NaryExpr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.And():
                exprs = [self._visit(e, ctx) for e in e.children]
                return NaryOp(NaryOpKind.AND, exprs, None)
            case fpc.Or():
                exprs = [self._visit(e, ctx) for e in e.children]
                return NaryOp(NaryOpKind.OR, exprs, None)
            case fpc.LT():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.LT for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.GT():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.GT for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.LEQ():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.LE for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.GEQ():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.GE for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.EQ():
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.EQ for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.NEQ():
                # TODO: need to check if semantics are the same
                assert len(e.children) >= 2, "not enough children"
                ops = [CompareOp.NE for _ in e.children[1:]]
                exprs = [self._visit(e, ctx) for e in e.children]
                return Compare(ops, exprs, None)
            case fpc.Size():
                # BUG: titanfp package says `fpc.Size` is n-ary
                if len(e.children) != 2:
                    raise ValueError('size operator expects 2 arguments')
                arg0 = self._visit(e.children[0], ctx)
                arg1 = self._visit(e.children[1], ctx)
                return BinaryOp(BinaryOpKind.SIZE, arg0, arg1, None)
            case fpc.UnknownOperator():
                name = pythonize_id(e.name)
                exprs = [self._visit(e, ctx) for e in e.children]
                return Call(name, exprs, None)
            case _:
                raise NotImplementedError('unexpected FPCore expression', e)

    def _visit_array(self, e: fpc.Array, ctx: _Ctx) -> Expr:
        exprs = [self._visit(e, ctx) for e in e.children]
        return TupleExpr(exprs, None)

    def _visit_ref(self, e: fpc.Ref, ctx: _Ctx) -> Expr:
        value = self._visit(e.children[0], ctx)
        slices = [self._visit(e, ctx) for e in e.children[1:]]
        return RefExpr(value, slices, None)

    def _visit_if(self, e: fpc.If, ctx: _Ctx) -> Expr:
        # create new blocks
        ift_ctx = ctx.without_stmts()
        iff_ctx = ctx.without_stmts()

        # compile children
        cond_expr = self._visit(e.cond, ctx)
        ift_expr = self._visit(e.then_body, ift_ctx)
        iff_expr = self._visit(e.else_body, iff_ctx)

        # emit temporary to bind result of branches
        t = self.gensym.fresh('t')
        ift_ctx.stmts.append(VarAssign(t, ift_expr, None, None))
        iff_ctx.stmts.append(VarAssign(t, iff_expr, None, None))

        # create if statement and bind it
        if_stmt = IfStmt(cond_expr, Block(ift_ctx.stmts), Block(iff_ctx.stmts), None)
        ctx.stmts.append(if_stmt)

        return Var(t, None)

    def _visit_let(self, e: fpc.Let, ctx: _Ctx) -> Expr:
        env = ctx.env
        is_star = isinstance(e, fpc.LetStar)

        for var, val in e.let_bindings:
            # compile value
            val_ctx = _Ctx(env=env, stmts=ctx.stmts) if is_star else ctx
            v_e = self._visit(val, val_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            env = { **env, var: t }
            stmt = VarAssign(t, v_e, None, None)
            ctx.stmts.append(stmt)

        return self._visit(e.body, _Ctx(env=env, stmts=ctx.stmts))

    def _visit_whilestar(self, e: fpc.WhileStar, ctx: _Ctx) -> Expr:
        env = ctx.env
        for var, init, _ in e.while_bindings:
            # compile value
            init_ctx = _Ctx(env=env, stmts=ctx.stmts)
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            env = { **env, var: t }
            stmt = VarAssign(t, init_e, None, None)
            ctx.stmts.append(stmt)

        # compile condition
        cond_ctx = _Ctx(env=env, stmts=ctx.stmts)
        cond_e = self._visit(e.cond, cond_ctx)

        # create loop body
        stmts: list[Stmt] = []
        update_ctx = _Ctx(env=env, stmts=stmts)
        for var, _, update in e.while_bindings:
            # compile value and update loop variable
            update_e = self._visit(update, update_ctx)
            stmt = VarAssign(env[var], update_e, None, None)
            stmts.append(stmt)

        # append while statement
        while_stmt = WhileStmt(cond_e, Block(stmts), None)
        ctx.stmts.append(while_stmt)

        # compile body
        body_ctx = _Ctx(env=env, stmts=ctx.stmts)
        return self._visit(e.body, body_ctx)

    def _visit_while(self, e: fpc.While, ctx: _Ctx) -> Expr:
        # initialize loop variables
        env = ctx.env
        for var, init, _ in e.while_bindings:
            # compile value
            init_e = self._visit(init, ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            env = { **env, var: t }
            stmt = VarAssign(t, init_e, None, None)
            ctx.stmts.append(stmt)

        # compile condition
        cond_ctx = _Ctx(env=env, stmts=ctx.stmts)
        cond_e = self._visit(e.cond, cond_ctx)

        # create loop body
        loop_env = dict(env)
        stmts: list[Stmt] = []
        update_ctx = _Ctx(env=env, stmts=stmts)
        for var, _, update in e.while_bindings:
            # compile value
            update_e = self._visit(update, update_ctx)
            # bind value to temporary
            t = self.gensym.fresh('t')
            loop_env = { **loop_env, var: t }
            stmt = VarAssign(t, update_e, None, None)
            stmts.append(stmt)

        # rebind temporaries
        for var, _, _ in e.while_bindings:
            v = env[var]
            t = loop_env[var]
            stmt = VarAssign(v, Var(t, None), None, None)
            stmts.append(stmt)

        # append while statement
        while_stmt = WhileStmt(cond_e, Block(stmts), None)
        ctx.stmts.append(while_stmt)

        # compile body
        body_ctx = _Ctx(env=env, stmts=ctx.stmts)
        return self._visit(e.body, body_ctx)

    def _make_tensor_body(
        self,
        iter_vars: list[NamedId],
        range_vars: list[NamedId],
        stmts: list[Stmt],
    ) -> list[Stmt]:
        if len(iter_vars) == 0:
            return stmts
        else:
            t = iter_vars[0]
            n = range_vars[0]
            inner_stmts: list[Stmt] = []
            e = UnaryOp(UnaryOpKind.RANGE, Var(n, None), None)
            stmt = ForStmt(t, e, Block(inner_stmts), None)
            stmts.append(stmt)
            return self._make_tensor_body(iter_vars[1:], range_vars[1:], inner_stmts)

    def _visit_tensorstar(self, e: fpc.TensorStar, ctx: _Ctx) -> Expr:
        # (tensor ([<i0> <e0>] ...) ([<x0> <init0> <update0>] ...) <body>)
        #
        # <n0> = <e0>
        # ...
        # <x0> = <init0>
        # ...
        # <t> = zeros(<n0>, ...)
        # for <i0> in range(<n0>):
        #    ...
        #      t[<i0>, ...] = <body>
        #      <x0> = <update0>
        #      ...
        #

        # bind iteration bounds to temporaries
        bound_vars: list[NamedId] = []
        for _, val in e.dim_bindings:
            t = self.gensym.fresh('t')
            stmt: Stmt = VarAssign(t, self._visit(val, ctx), None, None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize loop variables
        init_env = ctx.env.copy()
        init_ctx = _Ctx(env=ctx.env, stmts=ctx.stmts)
        for var, init, _ in e.while_bindings:
            # compile value
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            stmt = VarAssign(t, init_e, None, None)
            ctx.stmts.append(stmt)
            init_env[var] = t

        # initialize tensor
        tuple_id = self.gensym.fresh('t')
        zeroed = _zeros([Var(var, None) for var in bound_vars])
        stmt = VarAssign(tuple_id, zeroed, None, None)
        ctx.stmts.append(stmt)

        # initial iteration variables
        iter_vars: list[NamedId] = []
        loop_env = init_env.copy()
        for var, _ in e.dim_bindings:
            iter_id = self.gensym.fresh(var)
            iter_vars.append(iter_id)
            loop_env[var] = iter_id

        # generate for loops
        loop_stmts = self._make_tensor_body(iter_vars, bound_vars, ctx.stmts)
        loop_ctx = _Ctx(env=loop_env, stmts=loop_stmts)

        # set tensor element
        body_e = self._visit(e.body, loop_ctx)
        stmt = RefAssign(tuple_id, [Var(v, None) for v in iter_vars], body_e, None)
        loop_stmts.append(stmt)

        # compile loop updates
        for var, _, update in e.while_bindings:
            # compile value
            update_e = self._visit(update, loop_ctx)
            # bind value to temporary
            t = loop_ctx.env[var]
            stmt = VarAssign(t, update_e, None, None)
            loop_stmts.append(stmt)

        return Var(tuple_id, None)

    def _visit_tensor(self, e: fpc.Tensor, ctx: _Ctx) -> Expr:
        # (tensor ([<i0> <e0>] ...) <body>)
        #
        # <n0> = <e0>
        # ...
        # <t> = zeros(<n0>, ...)
        # for <i0> in range(<n0>):
        #    ...
        #      t[<i0>, ...] = <body>
        #

        # bind iteration bounds to temporaries
        bound_vars: list[NamedId] = []
        for _, val in e.dim_bindings:
            t = self.gensym.fresh('t')
            stmt: Stmt = VarAssign(t, self._visit(val, ctx), None, None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize tensor
        tuple_id = self.gensym.fresh('t')
        zeroed = _zeros([Var(var, None) for var in bound_vars])
        stmt = VarAssign(tuple_id, zeroed, None, None)
        ctx.stmts.append(stmt)

        # initial iteration variables
        iter_vars: list[NamedId] = []
        loop_env = ctx.env.copy()
        for var, _ in e.dim_bindings:
            iter_id = self.gensym.fresh(var)
            iter_vars.append(iter_id)
            loop_env[var] = iter_id

        # generate for loops
        loop_stmts = self._make_tensor_body(iter_vars, bound_vars, ctx.stmts)
        loop_ctx = _Ctx(env=loop_env, stmts=loop_stmts)

        # set tensor element
        body_e = self._visit(e.body, loop_ctx)
        stmt = RefAssign(tuple_id, [Var(v, None) for v in iter_vars], body_e, None)
        loop_stmts.append(stmt)

        return Var(tuple_id, None)

    def _visit_for(self, e: fpc.For, ctx: _Ctx) -> Expr:
        # (for ([<i0> <e0>] ...) ([<x0> <init0> <update0>] ...) <body>)
        #
        # <n0> = <e0>
        # ...
        # <x0> = <init0>
        # ...
        # for <i0> in range(<n0>):
        #   ...
        #     <t0> = <update0>
        #     ...
        #     <x0> = <t0>
        #     ...
        # <body>
        #

        is_star = isinstance(e, fpc.ForStar)

        # bind iteration bounds to temporaries
        bound_vars: list[NamedId] = []
        for _, val in e.dim_bindings:
            t = self.gensym.fresh('t')
            stmt: Stmt = VarAssign(t, self._visit(val, ctx), None, None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize loop variables
        init_env = ctx.env.copy()
        for var, init, _ in e.while_bindings:
            # compile value
            init_ctx = _Ctx(init_env if is_star else ctx.env, ctx.stmts)
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            stmt = VarAssign(t, init_e, None, None)
            ctx.stmts.append(stmt)
            init_env[var] = t

        # initial iteration variables
        iter_vars: list[NamedId] = []
        for var, _ in e.dim_bindings:
            iter_id = self.gensym.fresh(var)
            iter_vars.append(iter_id)
            init_env[var] = iter_id

        # generate for loops
        loop_env = init_env.copy()
        loop_stmts = self._make_tensor_body(iter_vars, bound_vars, ctx.stmts)
        loop_ctx = _Ctx(env=loop_env, stmts=loop_stmts)

        if is_star:
            # update loop variables
            for var, _, update in e.while_bindings:
                t = loop_env[var]
                update_e = self._visit(update, loop_ctx)
                stmt = VarAssign(t, update_e, None, None)
                loop_stmts.append(stmt)
        else:
            # temporary update variables
            update_env = loop_env.copy()
            for var, _, update in e.while_bindings:
                update_e = self._visit(update, loop_ctx)
                update_var = self.gensym.fresh(var)
                stmt = VarAssign(update_var, update_e, None, None)
                loop_stmts.append(stmt)
                update_env[var] = update_var

            # bind temporaries to loop variables
            for var, _, _ in e.while_bindings:
                x = init_env[var]
                t = update_env[var]
                stmt = VarAssign(x, Var(t, None), None, None)
                loop_stmts.append(stmt)

        body_ctx = _Ctx(env=init_env, stmts=ctx.stmts)
        return self._visit(e.body, body_ctx)

    def _visit_ctx(self, e: fpc.Ctx, ctx: _Ctx) -> Expr:
        # compile body
        val_ctx = ctx.without_stmts()
        val = self._visit(e.body, val_ctx)

        # compile properties
        props = self._visit_props(e.props, _Ctx(env=ctx.env))

        # bind value to temporary
        t = self.gensym.fresh('t')
        block = Block(val_ctx.stmts + [VarAssign(t, val, None, None)])
        stmt = ContextStmt(None, props, block, None)
        ctx.stmts.append(stmt)
        return Var(t, None)

    def _visit(self, e: fpc.Expr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.Var():
                return self._visit_var(e, ctx)
            case fpc.Decnum():
                return self._visit_decnum(e, ctx)
            case fpc.Hexnum():
                return self._visit_hexnum(e, ctx)
            case fpc.Integer():
                return self._visit_integer(e, ctx)
            case fpc.Rational():
                return self._visit_rational(e, ctx)
            case fpc.Digits():
                return self._visit_digits(e, ctx)
            case fpc.Constant():
                return self._visit_constant(e, ctx)
            case fpc.UnaryExpr():
                return self._visit_unary(e, ctx)
            case fpc.BinaryExpr():
                return self._visit_binary(e, ctx)
            case fpc.TernaryExpr():
                return self._visit_ternary(e, ctx)
            case fpc.Array():
                return self._visit_array(e, ctx)
            case fpc.Ref():
                return self._visit_ref(e, ctx)
            case fpc.NaryExpr():
                return self._visit_nary(e, ctx)
            case fpc.If():
                return self._visit_if(e, ctx)
            case fpc.Let():
                return self._visit_let(e, ctx)
            case fpc.WhileStar():
                return self._visit_whilestar(e, ctx)
            case fpc.While():
                return self._visit_while(e, ctx)
            case fpc.TensorStar():
                return self._visit_tensorstar(e, ctx)
            case fpc.Tensor():
                return self._visit_tensor(e, ctx)
            case fpc.For():
                return self._visit_for(e, ctx)
            case fpc.Ctx():
                return self._visit_ctx(e, ctx)
            case _:
                raise NotImplementedError(f'cannot convert to FPy {e}')

    def _visit_data(self, data: DataElt):
        match data:
            case fpc.String():
                return data.value
            case fpc.ValueExpr():
                return data.value
            case tuple():
                return [self._visit_data(d) for d in data]
            case _:
                raise NotImplementedError(repr(data))

    def _visit_props(self, props: dict[str, fpc.Data], ctx: _Ctx):
        new_props: dict[str, Any] = {}
        for k, v in props.items():
            match k:
                case 'pre' | 'spec' | 'alt':
                    e = data_as_expr(v, strict=True)
                    new_props[k] = self._visit(e, _Ctx(env=ctx.env))
                case _:
                    new_props[pythonize_id(k)] = self._visit_data(v.value)
        return new_props

    def _visit_function(self, f: fpc.FPCore):
        # setup context
        ctx = _Ctx()

        # compile arguments
        args: list[Argument] = []
        for name, _, shape in f.inputs:
            # TODO: argument properties and shape
            t = self.gensym.fresh(name)
            arg = Argument(t, None, None)
            args.append(arg)
            ctx.env[name] = t

            if shape is not None and any(isinstance(dim, str) for dim in shape):
                dim_ids: list[Id] = []
                for dim in shape:
                    if isinstance(dim, str):
                        d = self.gensym.fresh(dim)
                        dim_ids.append(d)
                    else:
                        dim_ids.append(UnderscoreId())

                shape_e = UnaryOp(UnaryOpKind.SHAPE, Var(t, None), None)
                stmt = TupleAssign(TupleBinding(dim_ids, None), shape_e, None)
                ctx.stmts.append(stmt)
                for dim, dim_id in zip(shape, dim_ids):
                    if isinstance(dim, str):
                        assert isinstance(dim_id, NamedId), "must be a NamedId"
                        ctx.env[dim] = dim_id

        # compile 
        props = self._visit_props(f.props, _Ctx(env=ctx.env))

        # compile function body
        e = self._visit(f.e, ctx)
        block = Block(ctx.stmts + [Return(e, None)])

        name = self.default_name if f.ident is None else pythonize_id(f.ident)
        ast = FunctionDef(name, args, block, None)
        ast.ctx = props
        return ast

    def convert(self) -> FunctionDef:
        ast = self._visit_function(self.core)
        if not isinstance(ast, FunctionDef):
            raise TypeError(f'should have produced a FunctionDef {ast}')
        return ast

def fpcore_to_fpy(core: fpc.FPCore, default_name: str = 'f'):
    ast = _FPCore2FPy(core, default_name).convert()

    # analyze and lower to the IR
    SyntaxCheck.analyze(ast)
    DefinitionAnalysis.analyze(ast)
    LiveVarAnalysis.analyze(ast)
    ir = IRCodegen.lower(ast)
    ir = SSA.apply(ir)
    VerifyIR.check(ir)

    return ir
