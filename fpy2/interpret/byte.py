"""
Interpreter backed by Python bytecode.
"""

import ast as pyast

from fractions import Fraction

from .. import ops
from ..ast.fpyast import *
from ..ast.visitor import Visitor
from ..function import Function
from .interpreter import Interpreter

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

###########################################################
# Eval namespace

def get_default_namespace() -> dict[str, object]:
    # namespace object with constructors
    namespace = {
        '__fpy_Fraction': Fraction,
    }

    # add operations to the namespace
    for op_type, fn in _NULLARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _UNARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _BINARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn
    for op_type, fn in _TERNARY_TABLE.items():
        namespace[f'__fpy_{op_type.__name__}'] = fn

    return namespace

###########################################################
# Bytecode compiler

class BytecodeCompiler(Visitor):
    """
    Compiler that compiles FPy AST to Python bytecode.
    """

    func: FuncDef

    def __init__(self, func: FuncDef):
        self.func = func

    def compile(self):
        # compile the function to a Python AST
        ast = self._visit_function(self.func, None)
        print(pyast.dump(ast))
        # compile the Python AST to bytecode
        source_name = self._location_to_name(self.func.loc)
        code = compile(pyast.Module(body=[ast], type_ignores=[]), filename=source_name, mode='exec')
        # return the function object
        namespace = get_default_namespace()
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
        kwargs = self._location_to_attributes(e.loc)
        return pyast.Call(
            func=pyast.Name(id='__fpy_Fraction', ctx=pyast.Load(), **kwargs),
            args=[
                pyast.Constant(value=val.numerator, **kwargs),
                pyast.Constant(value=val.denominator, **kwargs)
            ],
            keywords=[],
            **kwargs
        )

    def _visit_var(self, e: Var, ctx: None):
        kwargs = self._location_to_attributes(e.loc)
        return pyast.Name(id=str(e.name), ctx=pyast.Load(), **kwargs)

    def _visit_bool(self, e: BoolVal, ctx: None):
        kwargs = self._location_to_attributes(e.loc)
        return pyast.Constant(value=e.val, **kwargs)

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
            kwargs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **kwargs)
            ctx_arg = pyast.Name(id='__ctx__', ctx=pyast.Load(), **kwargs)
            return pyast.Call(func=func, args=[ctx_arg], keywords=[], **kwargs)
        else:
            raise NotImplementedError(f'unsupported nullary operation: {type(e).__name__}')

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if type(e) in _UNARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg = self._visit_expr(e.arg, ctx)
            kwargs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **kwargs)
            ctx_arg = pyast.Name(id='__ctx__', ctx=pyast.Load(), **kwargs)
            return pyast.Call(func=func, args=[arg, ctx_arg], keywords=[], **kwargs)
        else:
            raise NotImplementedError(f'unsupported unary operation: {type(e).__name__}')

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        if type(e) in _BINARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            kwargs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **kwargs)
            ctx_arg = pyast.Name(id='__ctx__', ctx=pyast.Load(), **kwargs)
            return pyast.Call(func=func, args=[arg1, arg2, ctx_arg], keywords=[], **kwargs)
        else:
            raise NotImplementedError(f'unsupported binary operation: {type(e).__name__}')

    def _visit_ternaryop(self, e: TernaryOp, ctx: None):
        if type(e) in _TERNARY_TABLE:
            name = f'__fpy_{type(e).__name__}'
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            arg3 = self._visit_expr(e.third, ctx)
            kwargs = self._location_to_attributes(e.loc)
            func = pyast.Name(id=name, ctx=pyast.Load(), **kwargs)
            ctx_arg = pyast.Name(id='__ctx__', ctx=pyast.Load(), **kwargs)
            return pyast.Call(func=func, args=[arg1, arg2, arg3, ctx_arg], keywords=[], **kwargs)
        else:
            raise NotImplementedError(f'unsupported ternary operation: {type(e).__name__}')

    def _visit_naryop(self, e: NaryOp, ctx: None):
        raise NotImplementedError

    def _visit_call(self, e: Call, ctx: None):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, ctx: None):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        raise NotImplementedError

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        raise NotImplementedError

    def _visit_list_comp(self, e: ListComp, ctx: None):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: None):
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: None):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        raise NotImplementedError

    def _visit_attribute(self, e: Attribute, ctx: None):
        raise NotImplementedError

    def _visit_target_unshaped(self, target: Id | TupleBinding) -> pyast.expr | list[pyast.expr]:
        match target:
            case SourceId():
                kwargs = self._location_to_attributes(target.loc)
                return pyast.Name(id=str(target), ctx=pyast.Store(), **kwargs)
            case UnderscoreId():
                kwargs = self._location_to_attributes(target.loc)
                return pyast.Name(id='_', ctx=pyast.Store(), **kwargs)
            case TupleBinding(elements):
                return [self._visit_target(elt, False) for elt in elements]

    def _visit_target(self, target: Id | TupleBinding) -> list[pyast.expr]:
        res = self._visit_target_unshaped(target)
        return res if isinstance(res, list) else [res]

    def _visit_assign(self, stmt: Assign, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        targets = self._visit_target(stmt.target)
        kwargs = self._location_to_attributes(stmt.loc)
        return pyast.Assign(targets=targets, value=expr, **kwargs)

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
        raise NotImplementedError

    def _visit_return(self, stmt: ReturnStmt, ctx: None) -> pyast.Return:
        expr = self._visit_expr(stmt.expr, ctx)
        kwargs = self._location_to_attributes(stmt.loc)
        return pyast.Return(value=expr, **kwargs)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        kwargs = self._location_to_attributes(stmt.loc)
        return pyast.Pass(**kwargs)

    def _visit_block(self, block: StmtBlock, ctx: None) -> list[pyast.stmt]:
        return [self._visit_statement(stmt, ctx) for stmt in block.stmts]

    def _visit_arguments(self, arg: Argument) -> pyast.arg:
        kwargs = self._location_to_attributes(arg.loc)
        return pyast.arg(arg=arg.name, **kwargs)

    def _visit_function(self, func: FuncDef, ctx: None):
        args = [self._visit_arguments(arg) for arg in func.args]
        body = self._visit_block(func.body, None)
        kwargs = self._location_to_attributes(func.loc)
        ctx_arg = pyast.arg(arg='__ctx__', **kwargs)
        return pyast.FunctionDef(
            name=func.name,
            args=pyast.arguments(posonlyargs=args, args=[ctx_arg]),
            body=body,
            **kwargs
        )

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
        compiler = BytecodeCompiler(func.ast)
        fn = compiler.compile()
        # compute the context to use during evaluation
        ctx = self._func_ctx(func, ctx)
        # call the function with the given arguments
        return fn(*args, __ctx__=ctx)

    def eval_expr(self, expr, env, ctx):
        raise NotImplementedError

