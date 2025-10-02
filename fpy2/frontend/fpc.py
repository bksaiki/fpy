"""
FPy parsing from FPCore.
"""

from typing import Any, TypeAlias

import titanfp.fpbench.fpcast as fpc
from titanfp.fpbench.fpcparser import data_as_expr

from ..ast.fpyast import *
from ..analysis import SyntaxCheck
from ..env import ForeignEnv
from ..fpc_context import FPCoreContext, NoSuchContextError
from ..utils import Gensym, pythonize_id


DataElt: TypeAlias = tuple['DataElt'] | fpc.ValueExpr

# Lazy lookup tables to avoid expensive initialization at import time
_constants_cache = None
_unary_table_cache = None
_binary_table_cache = None
_ternary_table_cache = None

def _get_constants():
    global _constants_cache
    if _constants_cache is None:
        _constants_cache = {
            'TRUE': BoolVal(True, None),
            'FALSE': BoolVal(False, None),
            'NAN': ConstNan(_func_symbol('nan'), None),
            'INFINITY': ConstInf(_func_symbol('inf'), None),
            'PI': ConstPi(_func_symbol('const_pi'), None),
            'E': ConstE(_func_symbol('const_e'), None),
            'LOG2E': ConstLog2E(_func_symbol('const_log2e'), None),
            'LOG10E': ConstLog10E(_func_symbol('const_log10e'), None),
            'LN2': ConstLn2(_func_symbol('const_ln2'), None),
            'PI_2': ConstPi_2(_func_symbol('const_pi_2'), None),
            'PI_4': ConstPi_4(_func_symbol('const_pi_4'), None),
            'M_1_PI': Const1_Pi(_func_symbol('const_1_pi'), None),
            'M_2_PI': Const2_Pi(_func_symbol('const_2_pi'), None),
            'M_2_SQRTPI': Const2_SqrtPi(_func_symbol('const_2_sqrtpi'), None),
            'SQRT2': ConstSqrt2(_func_symbol('const_sqrt2'), None),
            'SQRT1_2': ConstSqrt1_2(_func_symbol('const_sqrt1_2'), None),
        }
    return _constants_cache

def _get_unary_table():
    global _unary_table_cache
    if _unary_table_cache is None:
        _unary_table_cache = {
            'neg': Neg,
            'not': Not,
            'fabs': Fabs,
            'sqrt': Sqrt,
            'cbrt': Cbrt,
            'ceil': Ceil,
            'floor': Floor,
            'nearbyint': NearbyInt,
            'round': RoundInt,
            'trunc': Trunc,
            'acos': Acos,
            'asin': Asin,
            'atan': Atan,
            'cos': Cos,
            'sin': Sin,
            'tan': Tan,
            'acosh': Acosh,
            'asinh': Asinh,
            'atanh': Atanh,
            'cosh': Cosh,
            'sinh': Sinh,
            'tanh': Tanh,
            'exp': Exp,
            'exp2': Exp2,
            'expm1': Expm1,
            'log': Log,
            'log10': Log10,
            'log1p': Log1p,
            'log2': Log2,
            'erf': Erf,
            'erfc': Erfc,
            'lgamma': Lgamma,
            'tgamma': Tgamma,
            'isfinite': IsFinite,
            'isinf': IsInf,
            'isnan': IsNan,
            'isnormal': IsNormal,
            'signbit': Signbit,
            'range': Range,
            'dim': Dim,
        }
    return _unary_table_cache

def _get_binary_table():
    global _binary_table_cache
    if _binary_table_cache is None:
        _binary_table_cache = {
            '+': Add,
            '-': Sub,
            '*': Mul,
            '/': Div,
            'copysign': Copysign,
            'fdim': Fdim,
            'fmod': Fmod,
            'remainder': Remainder,
            'hypot': Hypot,
            'atan2': Atan2,
            'pow': Pow,
        }
    return _binary_table_cache

def _get_ternary_table():
    global _ternary_table_cache
    if _ternary_table_cache is None:
        _ternary_table_cache = {
            'fma': Fma
        }
    return _ternary_table_cache

def _func_symbol(name: str):
    return Var(NamedId(name), None)

def _empty(ns: list[Expr]) -> Expr:
    if len(ns) == 1:
        return Empty(_func_symbol('empty'), ns[0], None)
    elif len(ns) > 1:
        elt = _empty(ns[1:])
        return ListComp([UnderscoreId()], [Range(_func_symbol('range'), ns[0], None)], elt, None)
    else:
        raise ValueError(f'ns={ns} cannot be empty')

def _round(x: Expr) -> Expr:
    return Round(_func_symbol('round'), x, None)

# TODO: clean this up
class _Ctx:
    env: dict[str, NamedId]
    props: dict[str, Any]
    stmts: list[Stmt]

    def __init__(
        self,
        env: dict[str, NamedId] | None = None,
        props: dict[str, Any] | None = None,
        stmts: list[Stmt] | None = None
    ):
        if env is None:
            self.env = {}
        else:
            self.env = env

        if props is None:
            self.props = {}
        else:
            self.props = props

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
    env: ForeignEnv | None

    def __init__(self, core: fpc.FPCore, default_name: str, env: ForeignEnv | None = None):
        self.core = core
        self.gensym = Gensym(rename_hook=pythonize_id)
        self.default_name = default_name
        self.env = env

    def _visit_var(self, e: fpc.Var, ctx: _Ctx) -> Expr:
        if e.value not in ctx.env:
            raise ValueError(f'variable {e.value} not in scope')
        return Var(ctx.env[e.value], None)

    def _visit_constant(self, e: fpc.Constant, ctx: _Ctx) -> Expr:
        constants = _get_constants()
        if e.value not in constants:
            raise ValueError(f'unknown constant {e.name}')
        return constants[e.value]

    def _visit_decnum(self, e: fpc.Decnum, ctx: _Ctx) -> Expr:
        return _round(Decnum(str(e.value), None))

    def _visit_hexnum(self, e: fpc.Hexnum, ctx: _Ctx) -> Expr:
        return _round(Hexnum(_func_symbol('hexnum'), str(e.value), None))

    def _visit_integer(self, e: fpc.Integer, ctx: _Ctx) -> Expr:
        return _round(Integer(int(e.value), None))

    def _visit_rational(self, e: fpc.Rational, ctx: _Ctx) -> Expr:
        return _round(Rational(_func_symbol('rational'), e.p, e.q, None))

    def _visit_digits(self, e: fpc.Digits, ctx: _Ctx) -> Expr:
        return _round(Digits(_func_symbol('digits'), e.m, e.e, e.b, None))

    def _visit_unary(self, e: fpc.UnaryExpr, ctx: _Ctx) -> Expr:
        unary_table = _get_unary_table()
        if e.name == '-':
            arg = self._visit(e.children[0], ctx)
            return Neg(arg, None)
        elif e.name == 'cast':
            arg = self._visit(e.children[0], ctx)
            return _round(arg)
        elif e.name in unary_table:
            cls = unary_table[e.name]
            arg = self._visit(e.children[0], ctx)
            if issubclass(cls, NamedUnaryOp):
                return cls(_func_symbol(e.name), arg, None)
            else:
                return cls(arg, None)
        else:
            raise NotImplementedError(f'unsupported unary operation {e.name}')

    def _visit_binary(self, e: fpc.BinaryExpr, ctx: _Ctx) -> Expr:
        binary_table = _get_binary_table()
        if e.name in binary_table:
            cls = binary_table[e.name]
            left = self._visit(e.children[0], ctx)
            right = self._visit(e.children[1], ctx)
            if issubclass(cls, NamedBinaryOp):
                return cls(_func_symbol(e.name), left, right, None)
            else:
                return cls(left, right, None)
        else:
            match e.name:
                case 'fmin':
                    left = self._visit(e.children[0], ctx)
                    right = self._visit(e.children[1], ctx)
                    return Min(_func_symbol('min'), (left, right), None)
                case 'fmax':
                    left = self._visit(e.children[0], ctx)
                    right = self._visit(e.children[1], ctx)
                    return Max(_func_symbol('max'), (left, right), None)
                case _:
                    raise NotImplementedError(f'unsupported binary operation {e.name}')

    def _visit_ternary(self, e: fpc.TernaryExpr, ctx: _Ctx) -> Expr:
        ternary_table = _get_ternary_table()
        if e.name in ternary_table:
            cls = ternary_table[e.name]
            arg0 = self._visit(e.children[0], ctx)
            arg1 = self._visit(e.children[1], ctx)
            arg2 = self._visit(e.children[2], ctx)
            if issubclass(cls, NamedTernaryOp):
                return cls(_func_symbol(e.name), arg0, arg1, arg2, None)
            else:
                return cls(arg0, arg1, arg2, None)
        else:
            raise NotImplementedError(f'unsupported ternary operation {e.name}')

    def _visit_nary(self, e: fpc.NaryExpr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.And():
                exprs = [self._visit(e, ctx) for e in e.children]
                return And(exprs, None)
            case fpc.Or():
                exprs = [self._visit(e, ctx) for e in e.children]
                return Or(exprs, None)
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
                return Size(_func_symbol('size'), arg0, arg1, None)
            case fpc.UnknownOperator():
                ident = pythonize_id(e.name)
                exprs = [self._visit(e, ctx) for e in e.children]
                return Call(_func_symbol(ident), None, exprs, {}, None)
            case _:
                raise NotImplementedError('unexpected FPCore expression', e)

    def _visit_array(self, e: fpc.Array, ctx: _Ctx) -> Expr:
        exprs = [self._visit(e, ctx) for e in e.children]
        return ListExpr(exprs, None)

    def _visit_ref(self, e: fpc.Ref, ctx: _Ctx) -> Expr:
        # (ref <array> <index> ...)
        # => <array>[<index>][...]
        arr = self._visit(e.children[0], ctx)
        indices = [self._visit(e, ctx) for e in e.children[1:]]

        val: fpc.Expr = arr
        for index in indices:
            val = ListRef(val, index, None)
        return val

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
        ift_ctx.stmts.append(Assign(t, None, ift_expr, None))
        iff_ctx.stmts.append(Assign(t, None, iff_expr, None))

        # create if statement and bind it
        if_stmt = IfStmt(cond_expr, StmtBlock(ift_ctx.stmts), StmtBlock(iff_ctx.stmts), None)
        ctx.stmts.append(if_stmt)

        return Var(t, None)

    def _visit_let(self, e: fpc.Let, ctx: _Ctx) -> Expr:
        env = ctx.env
        is_star = isinstance(e, fpc.LetStar)

        for var, val in e.let_bindings:
            # compile value
            val_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts) if is_star else ctx
            v_e = self._visit(val, val_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            env = { **env, var: t }
            stmt = Assign(t, None, v_e, None)
            ctx.stmts.append(stmt)

        return self._visit(e.body, _Ctx(env=env, props=ctx.props, stmts=ctx.stmts))

    def _visit_whilestar(self, e: fpc.WhileStar, ctx: _Ctx) -> Expr:
        env = ctx.env
        for var, init, _ in e.while_bindings:
            # compile value
            init_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts)
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            env = { **env, var: t }
            stmt = Assign(t, None, init_e, None)
            ctx.stmts.append(stmt)

        # compile condition
        cond_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts)
        cond_e = self._visit(e.cond, cond_ctx)

        # create loop body
        stmts: list[Stmt] = []
        update_ctx = _Ctx(env=env, props=ctx.props, stmts=stmts)
        for var, _, update in e.while_bindings:
            # compile value and update loop variable
            update_e = self._visit(update, update_ctx)
            stmt = Assign(env[var], None, update_e, None)
            stmts.append(stmt)

        # append while statement
        while_stmt = WhileStmt(cond_e, StmtBlock(stmts), None)
        ctx.stmts.append(while_stmt)

        # compile body
        body_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts)
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
            stmt = Assign(t, None, init_e, None)
            ctx.stmts.append(stmt)

        # compile condition
        cond_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts)
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
            stmt = Assign(t, None, update_e, None)
            stmts.append(stmt)

        # rebind temporaries
        for var, _, _ in e.while_bindings:
            v = env[var]
            t = loop_env[var]
            stmt = Assign(v, None, Var(t, None), None)
            stmts.append(stmt)

        # append while statement
        while_stmt = WhileStmt(cond_e, StmtBlock(stmts), None)
        ctx.stmts.append(while_stmt)

        # compile body
        body_ctx = _Ctx(env=env, props=ctx.props, stmts=ctx.stmts)
        return self._visit(e.body, body_ctx)

    def _make_tensor_body(
        self,
        iter_vars: list[NamedId],
        range_vars: list[NamedId],
        stmts: list[Stmt]
    ) -> list[Stmt]:
        if len(iter_vars) == 0:
            return stmts
        else:
            t = iter_vars[0]
            n = range_vars[0]
            inner_stmts: list[Stmt] = []
            e = Range(_func_symbol('range'), Var(n, None), None)
            stmt = ForStmt(t, e, StmtBlock(inner_stmts), None)
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
        #      <x0> = <update0>
        #      ...
        #      t[<i0>, ...] = <body>
        #

        # bind iteration bounds to temporaries
        bound_vars: list[NamedId] = []
        for _, val in e.dim_bindings:
            t = self.gensym.fresh('t')
            stmt: Stmt = Assign(t, None, self._visit(val, ctx), None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize loop variables
        init_env = ctx.env.copy()
        init_ctx = _Ctx(env=ctx.env, props=ctx.props, stmts=ctx.stmts)
        for var, init, _ in e.while_bindings:
            # compile value
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            stmt = Assign(t, None, init_e, None)
            ctx.stmts.append(stmt)
            init_env[var] = t

        # initialize tensor
        tuple_id = self.gensym.fresh('t')
        zeroed = _empty([Var(var, None) for var in bound_vars])
        stmt = Assign(tuple_id, None, zeroed, None)
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
        loop_ctx = _Ctx(env=loop_env, props=ctx.props, stmts=loop_stmts)

        # compile loop updates
        for var, _, update in e.while_bindings:
            # compile value
            update_e = self._visit(update, loop_ctx)
            # bind value to temporary
            t = loop_ctx.env[var]
            stmt = Assign(t, None, update_e, None)
            loop_stmts.append(stmt)

        # set tensor element
        body_e = self._visit(e.body, loop_ctx)
        stmt = IndexedAssign(tuple_id, [Var(v, None) for v in iter_vars], body_e, None)
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
            stmt: Stmt = Assign(t, None, self._visit(val, ctx), None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize tensor
        tuple_id = self.gensym.fresh('t')
        zeroed = _empty([Var(var, None) for var in bound_vars])
        stmt = Assign(tuple_id, None, zeroed, None)
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
        loop_ctx = _Ctx(env=loop_env, props=ctx.props, stmts=loop_stmts)

        # set tensor element
        body_e = self._visit(e.body, loop_ctx)
        stmt = IndexedAssign(tuple_id, [Var(v, None) for v in iter_vars], body_e, None)
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
            stmt: Stmt = Assign(t, None, self._visit(val, ctx), None)
            ctx.stmts.append(stmt)
            bound_vars.append(t)

        # initialize loop variables
        init_env = ctx.env.copy()
        for var, init, _ in e.while_bindings:
            # compile value
            init_ctx = _Ctx(init_env if is_star else ctx.env, props=ctx.props, stmts=ctx.stmts)
            init_e = self._visit(init, init_ctx)
            # bind value to variable
            t = self.gensym.fresh(var)
            stmt = Assign(t, None, init_e, None)
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
        loop_ctx = _Ctx(env=loop_env, props=ctx.props, stmts=loop_stmts)

        if is_star:
            # update loop variables
            for var, _, update in e.while_bindings:
                t = loop_env[var]
                update_e = self._visit(update, loop_ctx)
                stmt = Assign(t, None, update_e, None)
                loop_stmts.append(stmt)
        else:
            # temporary update variables
            update_env = loop_env.copy()
            for var, _, update in e.while_bindings:
                update_e = self._visit(update, loop_ctx)
                update_var = self.gensym.fresh(var)
                stmt = Assign(update_var, None, update_e, None)
                loop_stmts.append(stmt)
                update_env[var] = update_var

            # bind temporaries to loop variables
            for var, _, _ in e.while_bindings:
                x = init_env[var]
                t = update_env[var]
                stmt = Assign(x, None, Var(t, None), None)
                loop_stmts.append(stmt)

        body_ctx = _Ctx(env=init_env, props=ctx.props, stmts=ctx.stmts)
        return self._visit(e.body, body_ctx)

    def _visit_ctx(self, e: fpc.Ctx, ctx: _Ctx) -> Expr:
        # compile body
        val_ctx = ctx.without_stmts()
        val = self._visit(e.body, val_ctx)

        # compile properties to a context
        props = self._visit_props(e.props, ctx)
        fpc_ctx = FPCoreContext(**props)

        # try to convert to a native FPy context
        try:
            ctx_val = ForeignVal(fpc_ctx.to_context(), None)
        except NoSuchContextError:
            ctx_val = ForeignVal(fpc_ctx, None)

        # bind value to temporary
        t = self.gensym.fresh('t')
        block = StmtBlock(val_ctx.stmts + [Assign(t, None, val, None)])
        stmt = ContextStmt(UnderscoreId(), ctx_val, block, None)
        ctx.stmts.append(stmt)
        return Var(t, None)

    def _visit(self, e: fpc.Expr, ctx: _Ctx) -> Expr:
        match e:
            case fpc.Var():
                return self._visit_var(e, ctx)
            case fpc.Constant():
                return self._visit_constant(e, ctx)
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
        new_props = dict(ctx.props)
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
            # argument id
            t = self.gensym.fresh(name)
            match shape:
                case tuple() | list():
                    # tensor argument
                    # bind named tensor dimensions
                    dims: list[int | NamedId] = []
                    for i, dim in enumerate(shape):
                        match dim:
                            case int():
                                # definite size dimension
                                dims.append(dim)
                            case str():
                                # named dimension
                                dim_id = self.gensym.fresh(dim)
                                size_e = Size(_func_symbol('size'), Var(t, None), Integer(i, None), None)
                                stmt = Assign(dim_id, None, size_e, None)
                                ctx.stmts.append(stmt)
                                # TODO: duplicate dimension names means a runtime check
                                # How should this be expressed in FPy?
                                ctx.env[dim] = dim_id
                                dims.append(dim_id)
                            case _:
                                raise RuntimeError(f'unsupported shape {shape} for argument {name}')

                    arg = Argument(t, SizedTensorTypeAnn(dims, AnyTypeAnn(None), None), None)
                    args.append(arg)
                    ctx.env[name] = t
                case None:
                    # scalar argument
                    arg = Argument(t, None, None)
                    args.append(arg)
                    ctx.env[name] = t
                case _:
                    raise RuntimeError(f'unsupported shape {shape} for argument {name}')

        # compile 
        props = self._visit_props(f.props, ctx)
        ctx.props = props

        # possibly generate context
        if 'precision' in props:
            try:
                ctx_val: None | Context | FPCoreContext = FPCoreContext(**props).to_context()
            except NoSuchContextError:
                ctx_val = FPCoreContext(**props)
            del props['precision']
        else:
            ctx_val = None

        # compile function body
        e = self._visit(f.e, ctx)
        block = StmtBlock(ctx.stmts + [ReturnStmt(e, None)])

        name = self.default_name if f.ident is None else pythonize_id(f.ident)
        env = ForeignEnv.default() if self.env is None else self.env
        return FuncDef(name, args, set(), ctx_val, block, None, props, env)

    def convert(self) -> FuncDef:
        return self._visit_function(self.core)


def fpcore_to_fpy(
    core: fpc.FPCore,
    *,
    env: ForeignEnv | None = None,
    default_name: str = 'f',
    ignore_unknown: bool = False
):
    # TODO: support `prefix` argument to list how
    # FPy builtins are printed
    ast = _FPCore2FPy(core, default_name, env).convert()
    SyntaxCheck.check(ast, ignore_unknown=ignore_unknown)
    return ast
