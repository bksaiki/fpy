"""
Interpreter backed by Python bytecode.
"""

import ast as pyast
import inspect

from fractions import Fraction
from typing import Any, TypeAlias

from .. import ops
from ..ast.fpyast import *
from ..ast.visitor import Visitor
from ..env import ForeignEnv
from ..function import Function
from ..number import Float, RealFloat, FP64, REAL
from ..primitive import Primitive
from ..utils import is_dyadic
from .interpreter import Interpreter

###########################################################
# Runtime

RealValue: TypeAlias = Float | Fraction
"""Type of real values in FPy programs."""
ScalarValue: TypeAlias = bool | Context | RealValue
"""Type of scalar values in FPy programs."""
TensorValue: TypeAlias = list['Value'] | tuple['Value', ...]
"""Type of list values in FPy programs."""
Value: TypeAlias = ScalarValue | TensorValue
"""Type of values in FPy programs."""


def _cvt_float(x: Value):
    match x:
        case Float():
            return x
        case Fraction():
            if not is_dyadic(x):
                raise TypeError(f'expected a dyadic rational, got `{x}`')
            return Float.from_rational(x, ctx=REAL)
        case _:
            raise TypeError(f'expected a real number, got `{x}`')

def _cvt_context_arg(cls: type[Context], name: str, arg: Any, ty: type):
    if ty is int:
        # convert to int
        val = _cvt_float(arg)
        if not val.is_integer():
            raise ValueError(f'expected an integer argument for `{name}={arg}`')
        return int(val)
    elif ty is float:
        # convert to float
        val = _cvt_float(arg)
        if not FP64.representable_under(val):
            raise ValueError(f'argument for `{name}={arg}` is not representable as a float')
        return float(val)
    elif ty is RealFloat:
        # convert to RealFloat
        val = _cvt_float(arg)
        if val.is_nar():
            raise ValueError(f'argument for `{name}={arg}` cannot be Inf/NaN')
        return val.as_real()
    else:
        # don't apply a conversion
        return arg

def _construct_context(cls: type[Context], args: list, kwargs: dict[str, object]):
    sig = inspect.signature(cls.__init__)
    _, *params = list(sig.parameters)

    ctor_args = []
    for arg, name in zip(args, params):
        param = sig.parameters[name]
        ctor_arg = _cvt_context_arg(cls, name, arg, param.annotation)
        ctor_args.append(ctor_arg)

    ctor_kwargs = {}
    for name, val in kwargs.items():
        if name not in sig.parameters:
            raise TypeError(f'unknown parameter {name} for constructor {cls}')
        param = sig.parameters[name]
        ctor_kwargs[name] = _cvt_context_arg(cls, name, val, param.annotation)

    return cls(*ctor_args, **ctor_kwargs)


def _eval_sum(*args, ctx: Context | None = None):
    raise NotImplementedError

def _eval_empty(*args, ctx: Context | None = None):
    raise NotImplementedError

def _eval_zip(*args, ctx: Context | None = None):
    raise NotImplementedError

def _eval_call(fn, ctx, *args, **kwargs):
    match fn:
        case Function():
            # calling FPy function
            if kwargs:
                raise RuntimeError('FPy functions do not support keyword arguments')
            return fn(*args, ctx=ctx)
        case Primitive():
            # calling FPy primitive
            if kwargs:
                raise RuntimeError('FPy primitives do not support keyword arguments')
            return fn(*args, ctx=ctx)
        case type() if issubclass(fn, Context):
            # calling context constructor
            return _construct_context(fn, args, kwargs)
        case _:
            # calling foreign function: only `print` is allowed
            if kwargs:
                raise RuntimeError('foreign functions do not support keyword arguments')
            if fn == print:
                print(*args)
                # TODO: should we allow `None` to return
                return None
            else:
                raise RuntimeError(f'attempting to call a Python function: `{fn}`')

###########################################################
# Operator tables

_NULLARY_TABLE: dict[type[NullaryOp], object] = {
    ConstNan: ops.nan,
    ConstInf: ops.inf,
    ConstPi: ops.const_pi,
    ConstE: ops.const_e,
    ConstLog2E: ops.const_log2e,
    ConstLog10E: ops.const_log10e,
    ConstLn2: ops.const_ln2,
    ConstPi_2: ops.const_pi_2,
    ConstPi_4: ops.const_pi_4,
    Const1_Pi: ops.const_1_pi,
    Const2_Pi: ops.const_2_pi,
    Const2_SqrtPi: ops.const_2_sqrt_pi,
    ConstSqrt2: ops.const_sqrt2,
    ConstSqrt1_2: ops.const_sqrt1_2,
}

_UNARY_TABLE: dict[type[UnaryOp], object] = {
    Abs: ops.fabs,
    Sqrt: ops.sqrt,
    Neg: ops.neg,
    Cbrt: ops.cbrt,
    Ceil: ops.ceil,
    Floor: ops.floor,
    NearbyInt: ops.nearbyint,
    RoundInt: ops.roundint,
    Trunc: ops.trunc,
    Acos: ops.acos,
    Asin: ops.asin,
    Atan: ops.atan,
    Cos: ops.cos,
    Sin: ops.sin,
    Tan: ops.tan,
    Acosh: ops.acosh,
    Asinh: ops.asinh,
    Atanh: ops.atanh,
    Cosh: ops.cosh,
    Sinh: ops.sinh,
    Tanh: ops.tanh,
    Exp: ops.exp,
    Exp2: ops.exp2,
    Expm1: ops.expm1,
    Log: ops.log,
    Log10: ops.log10,
    Log1p: ops.log1p,
    Log2: ops.log2,
    Erf: ops.erf,
    Erfc: ops.erfc,
    Lgamma: ops.lgamma,
    Tgamma: ops.tgamma,
    IsFinite: ops.isfinite,
    IsInf: ops.isinf,
    IsNan: ops.isnan,
    IsNormal: ops.isnormal,
    Signbit: ops.signbit,
    Round: ops.round,
    RoundExact: ops.round_exact,
    Cast: ops.cast,
    Logb: ops.logb,
    DeclContext: ops.declcontext
}

_BINARY_TABLE: dict[type[BinaryOp], object] = {
    Add: ops.add,
    Sub: ops.sub,
    Mul: ops.mul,
    Div: ops.div,
    Copysign: ops.copysign,
    Fdim: ops.fdim,
    Mod: ops.mod,
    Fmod: ops.fmod,
    Remainder: ops.remainder,
    Hypot: ops.hypot,
    Atan2: ops.atan2,
    Pow: ops.pow,
    RoundAt: ops.round_at
}

_TERNARY_TABLE: dict[type[TernaryOp], object] = {
    Fma: ops.fma,
}

_NARY_TABLE: dict[type[NaryOp], object] = {
    Sum: _eval_sum,
    Empty: _eval_empty,
    Zip: _eval_zip,
}

###########################################################
# Eval namespace

def inject_namespace(namespace: ForeignEnv) -> dict[str, object]:
    # add special symbols to namespace
    namespace['__fpy_Fraction'] = Fraction
    namespace['__fpy_call'] = _eval_call

    # add operations to the namespace
    for op_type, fn in _NULLARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _UNARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _BINARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _TERNARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _NARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn

###########################################################
# Bytecode compiler

CTX_NAME = '__ctx__'

class BytecodeCompiler(Visitor):
    """
    Compiler that compiles FPy AST to Python bytecode.
    """

    func: FuncDef
    env: ForeignEnv

    def __init__(self, func: FuncDef, env: ForeignEnv):
        self.func = func
        self.env = env

    def compile(self):
        # compile the function to a Python AST
        ast = self._visit_function(self.func, None)
        print(pyast.dump(ast))
        # compile the Python AST to bytecode
        source_name = self._location_to_name(self.func.loc)
        code = compile(pyast.Module(body=[ast], type_ignores=[]), filename=source_name, mode='exec')
        # inject runtime symbols
        namespace = {}
        inject_namespace(namespace)
        # add free variables to the namespace
        for var in self.func.free_vars:
            name = str(var)
            namespace[name] = self.env[name]
        # return the function object
        exec(code, namespace)
        return namespace[self.func.name]

    def _location_to_name(self, loc: Location | None) -> str:
        return '<unknown>' if loc is None else loc.source

    def _location_to_attributes(self, loc: Location | None) -> dict[str, int]:
        if loc is None:
            # dummy value to signal missing location information
            return {
                'lineno': -1,
                'col_offset': -1,
                'end_lineno': -1,
                'end_col_offset': -1
            }
        else:
            return {
                'lineno': loc.start_line,
                'col_offset': loc.start_column,
                'end_lineno': loc.end_line,
                'end_col_offset': loc.end_column
            }

    def _rational_to_ast(self, e: Rational) -> pyast.Call:
        val = e.as_rational()
        attrs = self._location_to_attributes(e.loc)
        return pyast.Call(
            func=pyast.Name(id='__fpy_Fraction', ctx=pyast.Load(), **attrs),
            args=[
                pyast.Constant(value=val.numerator, **attrs),
                pyast.Constant(value=val.denominator, **attrs)
            ],
            keywords=[],
            **attrs
        )

    def _visit_var(self, e: Var, ctx: None):
        attrs = self._location_to_attributes(e.loc)
        return pyast.Name(id=str(e.name), ctx=pyast.Load(), **attrs)

    def _visit_bool(self, e: BoolVal, ctx: None):
        attrs = self._location_to_attributes(e.loc)
        return pyast.Constant(value=e.val, **attrs)

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: None):
        return self._rational_to_ast(e)

    def _visit_hexnum(self, e: Hexnum, ctx: None):
        return self._rational_to_ast(e)

    def _visit_integer(self, e: Integer, ctx: None):
        return self._rational_to_ast(e)

    def _visit_rational(self, e: Rational, ctx: None):
        return self._rational_to_ast(e)

    def _visit_digits(self, e: Digits, ctx: None):
        return self._rational_to_ast(e)

    def _visit_nullaryop(self, e: NullaryOp, ctx: None):
        if type(e) in _NULLARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            attrs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **attrs)
            ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)
            return pyast.Call(func=func, args=[ctx_arg], keywords=[], **attrs)
        else:
            raise NotImplementedError(f'unsupported nullary operation: {type(e).__name__}')

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if type(e) in _UNARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg = self._visit_expr(e.arg, ctx)
            attrs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **attrs)
            ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)
            return pyast.Call(func=func, args=[arg, ctx_arg], keywords=[], **attrs)
        else:
            raise NotImplementedError(f'unsupported unary operation: {type(e).__name__}')

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        if type(e) in _BINARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            attrs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **attrs)
            ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)
            return pyast.Call(func=func, args=[arg1, arg2, ctx_arg], keywords=[], **attrs)
        else:
            raise NotImplementedError(f'unsupported binary operation: {type(e).__name__}')

    def _visit_ternaryop(self, e: TernaryOp, ctx: None):
        if type(e) in _TERNARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            arg3 = self._visit_expr(e.third, ctx)
            attrs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **attrs)
            ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)
            return pyast.Call(func=func, args=[arg1, arg2, arg3, ctx_arg], keywords=[], **attrs)
        else:
            raise NotImplementedError(f'unsupported ternary operation: {type(e).__name__}')

    def _visit_naryop(self, e: NaryOp, ctx: None):
        attrs = self._location_to_attributes(e.loc)
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        if type(e) in _NARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            func = pyast.Name(id=name, ctx=pyast.Load(), **attrs)
            ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)
            return pyast.Call(func=func, args=args + [ctx_arg], keywords=[], **attrs)

        match e:
            case Min():
                func = pyast.Name(id='min', ctx=pyast.Load(), **attrs)
                return pyast.Call(func=func, args=args, keywords=[], **attrs)
            case Max():
                func = pyast.Name(id='max', ctx=pyast.Load(), **attrs)
                return pyast.Call(func=func, args=args, keywords=[], **attrs)
            case _:
                raise NotImplementedError(f'unsupported n-ary operation: {type(e).__name__}')

    def _visit_call(self, e: Call, ctx: None):
        func = self._visit_expr(e.func, ctx)
        attrs = self._location_to_attributes(e.loc)
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        ctx_arg = pyast.Name(id=CTX_NAME, ctx=pyast.Load(), **attrs)

        kwargs: list[pyast.keyword] = []
        for kw, val in e.kwargs:
            value = self._visit_expr(val, ctx)
            kwarg = pyast.keyword(arg=kw, value=value, **attrs)
            kwargs.append(kwarg)

        name = pyast.Name(id='__fpy_call', ctx=pyast.Load(), **attrs)
        return pyast.Call(func=name, args=[func, ctx_arg] + args, keywords=kwargs, **attrs)

    def _visit_compare_op(self, op: CompareOp) -> pyast.cmpop:
        match op:
            case CompareOp.EQ:
                return pyast.Eq()
            case CompareOp.NE:
                return pyast.NotEq()
            case CompareOp.LT:
                return pyast.Lt()
            case CompareOp.LE:
                return pyast.LtE()
            case CompareOp.GT:
                return pyast.Gt()
            case CompareOp.GE:
                return pyast.GtE()
            case _:
                raise NotImplementedError(f'unsupported comparison operator: {type(op).__name__}')

    def _visit_compare(self, e: Compare, ctx: None):
        ops = [self._visit_compare_op(op) for op in e.ops]
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        attrs = self._location_to_attributes(e.loc)
        return pyast.Compare(args[0], ops, args[1:], **attrs)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        args = [self._visit_expr(elt, ctx) for elt in e.elts]
        attrs = self._location_to_attributes(e.loc)
        return pyast.Tuple(elts=args, ctx=pyast.Load(), **attrs)

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        args = [self._visit_expr(elt, ctx) for elt in e.elts]
        attrs = self._location_to_attributes(e.loc)
        return pyast.List(elts=args, ctx=pyast.Load(), **attrs)

    def _visit_target_unshaped(self, target: Id | TupleBinding) -> pyast.expr | list[pyast.expr]:
        match target:
            case SourceId():
                attrs = self._location_to_attributes(target.loc)
                return pyast.Name(id=str(target), ctx=pyast.Store(), **attrs)
            case NamedId():
                attrs = self._location_to_attributes(None)
                return pyast.Name(id=str(target), ctx=pyast.Store(), **attrs)
            case UnderscoreId():
                attrs = self._location_to_attributes(None)
                return pyast.Name(id='_', ctx=pyast.Store(), **attrs)
            case TupleBinding():
                elts = [self._visit_target(elt) for elt in target.elts]
                attrs = self._location_to_attributes(target.loc)
                return pyast.Tuple(elts=elts, ctx=pyast.Store(), **attrs)
            case _:
                raise NotImplementedError(f'unsupported target type: {type(target).__name__}')

    def _visit_list_comp(self, e: ListComp, ctx: None):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: None):
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: None):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        attrs = self._location_to_attributes(e.loc)
        return pyast.IfExp(test=cond, body=ift, orelse=iff, **attrs)

    def _visit_attribute(self, e: Attribute, ctx: None):
        value = self._visit_expr(e.value, ctx)
        attrs = self._location_to_attributes(e.loc)
        return pyast.Attribute(value=value, attr=e.attr, ctx=pyast.Load(), **attrs)

    def _visit_target(self, target: Id | TupleBinding) -> list[pyast.expr]:
        res = self._visit_target_unshaped(target)
        return res if isinstance(res, list) else [res]

    def _visit_assign(self, stmt: Assign, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        targets = self._visit_target(stmt.target)
        attrs = self._location_to_attributes(stmt.loc)
        return pyast.Assign(targets=targets, value=expr, **attrs)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, ctx: None):
        raise NotImplementedError

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx: None):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        attrs = self._location_to_attributes(stmt.loc)
        return pyast.Expr(value=expr, **attrs)

    def _visit_return(self, stmt: ReturnStmt, ctx: None) -> pyast.Return:
        expr = self._visit_expr(stmt.expr, ctx)
        attrs = self._location_to_attributes(stmt.loc)
        return pyast.Return(value=expr, **attrs)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        attrs = self._location_to_attributes(stmt.loc)
        return pyast.Pass(**attrs)

    def _visit_block(self, block: StmtBlock, ctx: None) -> list[pyast.stmt]:
        return [self._visit_statement(stmt, ctx) for stmt in block.stmts]

    def _visit_arguments(self, arg: Argument) -> pyast.arg:
        attrs = self._location_to_attributes(arg.loc)
        return pyast.arg(arg=str(arg.name), **attrs)

    def _visit_function(self, func: FuncDef, ctx: None):
        # TODO: convert value to FPy value
        args = [self._visit_arguments(arg) for arg in func.args]
        body = self._visit_block(func.body, None)
        attrs = self._location_to_attributes(func.loc)
        ctx_arg = pyast.arg(arg=CTX_NAME, **attrs)
        args = pyast.arguments(posonlyargs=args, args=[ctx_arg])
        return pyast.FunctionDef(name=func.name, args=args, body=body, **attrs)

###########################################################
# Interpreter

class BytecodeInterpreter(Interpreter):
    """
    Interpreter that compiles to Python bytecode and executes it.
    """

    def eval(self, func: Function, args, ctx: Context | None = None):
        if not isinstance(func, Function):
            raise TypeError(f'Expected Function, got `{func}`')
        # compile the function to bytecode
        compiler = BytecodeCompiler(func.ast, func.env)
        fn = compiler.compile()
        # compute the context to use during evaluation
        ctx = self._func_ctx(func, ctx)
        # call the function with the given arguments
        return fn(*args, __ctx__=ctx)

    def eval_expr(self, expr, env, ctx):
        raise NotImplementedError

