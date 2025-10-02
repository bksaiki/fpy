"""
This module contains the parser for the FPy language.
"""

import ast

from typing import Any, Callable, Mapping
from types import FunctionType

from ..ast.fpyast import *
from ..env import ForeignEnv
from ..number import Float, Real
from ..utils import NamedId, UnderscoreId, SourceId
from ..ops import *

_nullary_table: dict[Callable, type[NullaryOp]] = {
    nan: ConstNan,
    inf: ConstInf,
    const_pi: ConstPi,
    const_e: ConstE,
    const_log2e: ConstLog2E,
    const_log10e: ConstLog10E,
    const_ln2: ConstLn2,
    const_pi_2: ConstPi_2,
    const_pi_4: ConstPi_4,
    const_1_pi: Const1_Pi,
    const_2_pi: Const2_Pi,
    const_2_sqrt_pi: Const2_SqrtPi,
    const_sqrt2: ConstSqrt2,
    const_sqrt1_2: ConstSqrt1_2,
}

_unary_table: dict[Callable, type[UnaryOp] | type[NamedUnaryOp]] = {
    abs: Fabs,
    fabs: Fabs,
    sqrt: Sqrt,
    cbrt: Cbrt,
    ceil: Ceil,
    floor: Floor,
    nearbyint: NearbyInt,
    roundint: RoundInt,
    trunc: Trunc,
    acos: Acos,
    asin: Asin,
    atan: Atan,
    cos: Cos,
    sin: Sin,
    tan: Tan,
    acosh: Acosh,
    asinh: Asinh,
    atanh: Atanh,
    cosh: Cosh,
    sinh: Sinh,
    tanh: Tanh,
    exp: Exp,
    exp2: Exp2,
    expm1: Expm1,
    log: Log,
    log10: Log10,
    log1p: Log1p,
    log2: Log2,
    erf: Erf,
    erfc: Erfc,
    lgamma: Lgamma,
    tgamma: Tgamma,
    isfinite: IsFinite,
    isinf: IsInf,
    isnan: IsNan,
    isnormal: IsNormal,
    signbit: Signbit,
    round: Round,
    round_exact: RoundExact,
    len: Len,
    range: Range,
    empty: Empty,
    dim: Dim,
    enumerate: Enumerate,
    sum: Sum
}

_binary_table: dict[Callable, type[BinaryOp] | type[NamedBinaryOp]] = {
    add: Add,
    sub: Sub,
    mul: Mul,
    div: Div,
    copysign: Copysign,
    fdim: Fdim,
    fmod: Fmod,
    remainder: Remainder,
    hypot: Hypot,
    atan2: Atan2,
    pow: Pow,
    size: Size,
    round_at: RoundAt,
}

_ternary_table: dict[Callable, type[TernaryOp] | type[NamedTernaryOp]] = {
    fma: Fma
}

_nary_table: dict[Callable, type[NaryOp] | type[NamedNaryOp]] = {
    zip: Zip,
    max: Max,
    min: Min,
    fmin: Min,
    fmax: Max,
}


class FPyParserError(Exception):
    """Parser error for FPy"""
    loc: Location
    why: str
    where: ast.AST
    ctx: ast.AST | None

    def __init__(
        self,
        loc: Location,
        why: str,
        where: ast.AST,
        ctx: ast.AST | None = None
    ):
        msg_lines = [why]
        match where:
            case ast.expr() | ast.stmt():
                start_line = loc.start_line
                start_col = loc.start_column
                msg_lines.append(f' at: {loc.source}:{start_line}:{start_col}')
            case _:
                pass

        msg_lines.append(f' where: {ast.unparse(where)}')
        if ctx is not None:
            msg_lines.append(f' in: {ast.unparse(ctx)}')

        super().__init__('\n'.join(msg_lines))
        self.loc = loc
        self.why = why
        self.where = where
        self.ctx = ctx


def _ipow(expr: Expr, n: int, loc: Location):
    assert n >= 0, "must be a non-negative integer"
    if n == 0:
        return Integer(1, loc)
    elif n == 1:
        return expr
    else:
        e = Mul(expr, expr, loc)
        for _ in range(2, n):
            e = Mul(e, expr, loc)
        return e

class Parser:
    """
    FPy parser.

    Converts a Python AST (from the `ast` module) to a FPy AST.
    """

    name: str
    source: str
    env: ForeignEnv
    lines: list[str]
    start_line: int

    def __init__(
        self,
        name: str, 
        source: str,
        env: ForeignEnv,
        start_line: int = 1
    ):
        self.name = name
        self.source = source
        self.env = env
        self.lines = source.splitlines()
        self.start_line = start_line

    def _parse_location(self, e: ast.expr | ast.stmt | ast.arg) -> Location:
        """Extracts the parse location of a  Python ST node."""
        assert e.end_lineno is not None, "missing end line number"
        assert e.end_col_offset is not None, "missing end column offset"
        return Location(
            self.name,
            e.lineno + self.start_line - 1,
            e.col_offset,
            e.end_lineno + self.start_line - 1,
            e.end_col_offset
        )

    def _convert_type(self, ty, loc: Location):
        if ty == Real:
            return RealTypeAnn(loc)
        elif isinstance(ty, type):
            if issubclass(ty, bool):
                return BoolTypeAnn(loc)
            elif issubclass(ty, int) or issubclass(ty, float):
                # TODO: more specific type
                return RealTypeAnn(loc)
            elif issubclass(ty, Float):
                return RealTypeAnn(loc)
            else:
                # TODO: implement
                return AnyTypeAnn(loc)
        elif isinstance(ty, tuple):
            elts = [self._convert_type(elt, loc) for elt in ty]
            return TupleTypeAnn(elts, loc)
        elif isinstance(ty, list):
            elt = self._convert_type(ty[0], loc)
            return ListTypeAnn(elt, loc)
        else:
            # TODO: implement
            return AnyTypeAnn(loc)

    def _eval_type_annotation(self, ann: ast.expr):
        loc = self._parse_location(ann)
        match ann:
            case ast.Attribute():
                attr = self._parse_attribute(ann)
                return self._eval_attribute(attr, ann)
            case ast.Name():
                ident = self._parse_id(ann)
                if isinstance(ident, UnderscoreId):
                    raise FPyParserError(loc, 'FPy function call must begin with a named identifier', ann)
                if ident.base not in self.env:
                    raise FPyParserError(loc, f'name \'{ident.base}\' not defined:', ann)
                return self.env[ident.base]
            case ast.Subscript():
                ctor = self._eval_type_annotation(ann.value)
                if ctor is tuple:
                    # tuple[t1, ...]
                    arg = self._eval_type_annotation(ann.slice)
                    match arg:
                        case tuple():
                            return arg
                        case _:
                            return (arg,)
                elif ctor is list:
                    # list[t]
                    arg = self._eval_type_annotation(ann.slice)
                    return [arg]
                else:
                    return None
            case ast.Tuple():
                return tuple(self._eval_type_annotation(elt) for elt in ann.elts)
            case _:
                # TODO: implement
                return None

    def _parse_type_annotation(self, ann: ast.expr) -> TypeAnn:
        loc = self._parse_location(ann)
        ty = self._eval_type_annotation(ann)
        if ty is None:
            return AnyTypeAnn(loc)
        return self._convert_type(ty, loc)

    def _parse_id(self, e: ast.Name):
        if e.id == '_':
            return UnderscoreId()
        else:
            loc = self._parse_location(e)
            return SourceId(e.id, loc)

    def _parse_constant(self, e: ast.Constant):
        # TODO: reparse all constants to get exact value
        loc = self._parse_location(e)
        match e.value:
            case bool():
                return BoolVal(e.value, loc)
            case int():
                return Integer(e.value, loc)
            case float():
                if e.value.is_integer():
                    return Integer(int(e.value), loc)
                else:
                    return Decnum(str(e.value), loc)
            case str():
                return ForeignVal(e.value, loc)
            case None:
                return ForeignVal(e.value, loc)
            case _:
                raise FPyParserError(loc, 'Unsupported constant', e)

    def _parse_hexfloat(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 1:
            raise FPyParserError(loc, 'FPy `hexfloat` expects one argument', e)
        arg = self._parse_expr(e.args[0])
        if not isinstance(arg, ForeignVal):
            raise FPyParserError(loc, 'FPy `hexfloat` expects a string', e)
        return Hexnum(func, arg.val, loc)

    def _parse_rational(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 2:
            raise FPyParserError(loc, 'FPy `rational` expects two arguments', e)
        p = self._parse_expr(e.args[0])
        if not isinstance(p, Integer):
            raise FPyParserError(loc, 'FPy `rational` expects an integer as first argument', e)
        q = self._parse_expr(e.args[1])
        if not isinstance(q, Integer):
            raise FPyParserError(loc, 'FPy `rational` expects an integer as second argument', e)
        return Rational(func, p.val, q.val, loc)

    def _parse_digits(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 3:
            raise FPyParserError(loc, 'FPy `digits` expects three arguments', e)
        m_e = self._parse_expr(e.args[0])
        if not isinstance(m_e, Integer):
            raise FPyParserError(loc, 'FPy `digits` expects an integer as first argument', e)
        e_e = self._parse_expr(e.args[1])
        if not isinstance(e_e, Integer):
            raise FPyParserError(loc, 'FPy `digits` expects an integer as second argument', e)
        b_e = self._parse_expr(e.args[2])
        if not isinstance(b_e, Integer):
            raise FPyParserError(loc, 'FPy `digits` expects an integer as third argument', e)
        return Digits(func, m_e.val, e_e.val, b_e.val, loc)

    def _parse_boolop(self, e: ast.BoolOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.And():
                args = [self._parse_expr(e) for e in e.values]
                return And(args, loc)
            case ast.Or():
                args = [self._parse_expr(e) for e in e.values]
                return Or(args, loc)
            case _:
                raise FPyParserError(loc, 'Not a valid FPy operator', e.op, e)

    def _parse_unaryop(self, e: ast.UnaryOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.UAdd():
                return self._parse_expr(e.operand)
            case ast.USub():
                arg = self._parse_expr(e.operand)
                if isinstance(arg, Integer):
                    return Integer(-arg.val, loc)
                else:
                    return Neg(arg, loc)
            case ast.Not():
                arg = self._parse_expr(e.operand)
                return Not(arg, loc)
            case _:
                raise FPyParserError(loc, 'Not a valid FPy operator', e.op, e)

    def _parse_binop(self, e: ast.BinOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.Add():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return Add(lhs, rhs, loc)
            case ast.Sub():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return Sub(lhs, rhs, loc)
            case ast.Mult():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return Mul(lhs, rhs, loc)
            case ast.Div():
                lhs = self._parse_expr(e.left)
                rhs = self._parse_expr(e.right)
                return Div(lhs, rhs, loc)
            case ast.Pow():
                base = self._parse_expr(e.left)
                exp = self._parse_expr(e.right)
                if not isinstance(exp, Integer) or exp.val < 0:
                    raise FPyParserError(loc, 'FPy only supports `**` for small integer exponent, use `pow()` instead', e.op, e)
                return _ipow(base, exp.val, loc)
            case _:
                raise FPyParserError(loc, 'Not a valid FPy operator', e.op, e)

    def _parse_cmpop(self, op: ast.cmpop, e: ast.Compare):
        loc = self._parse_location(e)
        match op:
            case ast.Lt():
                return CompareOp.LT
            case ast.LtE():
                return CompareOp.LE
            case ast.GtE():
                return CompareOp.GE
            case ast.Gt():
                return CompareOp.GT
            case ast.Eq():
                return CompareOp.EQ
            case ast.NotEq():
                return CompareOp.NE
            case _:
                raise FPyParserError(loc, 'Not a valid FPy comparator', op, e)

    def _parse_compare(self, e: ast.Compare):
        loc = self._parse_location(e)
        ops = [self._parse_cmpop(op, e) for op in e.ops]
        args = [self._parse_expr(e) for e in [e.left, *e.comparators]]
        return Compare(ops, args, loc)

    def _eval_var(self, v: Var, e: ast.expr):
        if v.name.base not in self.env:
            loc = self._parse_location(e)
            raise FPyParserError(loc, f'name \'{v.name}\' not defined:', e)
        return self.env[v.name.base]

    def _eval_attribute(self, a: Attribute, e: ast.expr):
        match a.value:
            case Var():
                # evaluating `x.y`
                base = self._eval_var(a.value, e)
            case Attribute():
                # evaluating `x.y.z` where `x.y` is `a`
                base = self._eval_attribute(a.value, e)

        # lookup the attribute
        if not hasattr(base, a.attr):
            loc = self._parse_location(e)
            raise FPyParserError(loc, f'unknown attribute \'{a.attr}\' for {base}', e)

        return getattr(base, a.attr)

    def _parse_call(self, e: ast.Call):
        """Parse a Python call function."""
        # parse function expression
        loc = self._parse_location(e.func)
        match e.func:
            case ast.Attribute():
                func = self._parse_attribute(e.func)
                fn = self._eval_attribute(func, e.func)
            case ast.Name():
                name = self._parse_id(e.func)
                func = Var(name, None)
                if isinstance(func, UnderscoreId):
                    raise FPyParserError(loc, 'FPy function call must begin with a named identifier', e)
                if name.base not in self.env:
                    raise FPyParserError(loc, f'name \'{name.base}\' not defined:', e)
                fn = self.env[name.base]
            case _:
                raise RuntimeError('unreachable')

        # parse arguments
        args = [self._parse_expr(arg) for arg in e.args]

        # parse keyword arguments
        kwargs: list[tuple[str, Expr]] = []
        for kwarg in e.keywords:
            if kwarg.arg is None:
                raise FPyParserError(loc, 'FPy does not support **kwargs', e)
            kwarg_val = self._parse_expr(kwarg.value)
            kwargs.append((kwarg.arg, kwarg_val))

        # lookup builtin symbols
        if fn in _nullary_table:
            cls0 = _nullary_table[fn]
            if len(e.args) != 0:
                raise FPyParserError(loc, f'FPy expects 0 arguments for `{fn}`, got {len(e.args)}', e)
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            return cls0(func, loc)
        elif fn in _unary_table:
            cls1 = _unary_table[fn]
            if len(args) != 1:
                raise FPyParserError(loc, f'FPy expects 1 argument for `{fn}`, got {len(e.args)}', e)
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            if issubclass(cls1, NamedUnaryOp):
                return cls1(func, args[0], loc)
            else:
                return cls1(args[0], loc)
        elif fn in _binary_table:
            cls2 = _binary_table[fn]
            if len(args) != 2:
                raise FPyParserError(loc, f'FPy expects 2 arguments for `{fn}`, got {len(e.args)}', e)
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            if issubclass(cls2, NamedBinaryOp):
                return cls2(func, args[0], args[1], loc)
            else:
                return cls2(args[0], args[1], loc)
        elif fn in _ternary_table:
            cls3 = _ternary_table[fn]
            if len(args) != 3:
                raise FPyParserError(loc, f'FPy expects 3 arguments for `{fn}`, got {len(e.args)}', e)
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            if issubclass(cls3, NamedTernaryOp):
                return cls3(func, args[0], args[1], args[2], loc)
            else:
                return cls3(args[0], args[1], args[2], loc)
        elif fn in _nary_table:
            cls = _nary_table[fn]
            if (cls is Min or cls is Max) and len(args) < 2:
                raise FPyParserError(loc, f'FPy expects at least 2 arguments for `{fn}`', e)
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            if issubclass(cls, NamedNaryOp):
                return cls(func, args, loc)
            else:
                return cls(args, loc)
        elif fn == rational:
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            return self._parse_rational(e, func)
        elif fn == hexfloat:
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            return self._parse_hexfloat(e, func)
        elif fn == digits:
            if kwargs:
                raise FPyParserError(loc, f'FPy does not support keyword arguments for `{fn}`', e)
            return self._parse_digits(e, func)
        elif fn == len:
            if len(args) != 1:
                raise FPyParserError(loc, 'FPy expects 1 argument for `len`', e)
            if kwargs:
                raise FPyParserError(loc, 'FPy does not support keyword arguments for `len`', e)
            return Size(func, args[0], Integer(0, None), loc)
        else:
            return Call(func, fn, args, kwargs, loc)

    def _parse_slice(self, e: ast.Slice):
        """Parse a Python slice expression."""
        if e.lower is None:
            lower = None
        else:
            lower = self._parse_expr(e.lower)

        if e.upper is None:
            upper = None
        else:
            upper = self._parse_expr(e.upper)

        if e.step is not None:
            loc = self._parse_location(e.step)
            raise FPyParserError(loc, 'FPy does not support slice step', e.step)

        return lower, upper

    def _parse_subscript(self, e: ast.Subscript):
        """Parsing a subscript slice that is an expression"""
        loc = self._parse_location(e)
        value = self._parse_expr(e.value)
        match e.slice:
            case ast.Slice():
                lower, upper = self._parse_slice(e.slice)
                return ListSlice(value, lower, upper, loc)
            case _:
                slice =  self._parse_expr(e.slice)
                return ListRef(value, slice, loc)

    def _parse_subscript_target(self, e: ast.Subscript):
        """Parsing a subscript slice that is the LHS of an assignment."""
        t: ast.expr = e
        slices: list[Expr] = []
        while isinstance(t, ast.Subscript):
            slices.append(self._parse_expr(t.slice))
            t = t.value

        target = self._parse_expr(t)
        slices.reverse()

        return target, slices

    def _is_foreign_val(self, e: ast.expr):
        match e:
            case ast.Name() | ast.Constant():
                return True
            case ast.Attribute():
                return self._is_foreign_val(e.value)
            case _:
                return False

    def _parse_attribute(self, e: ast.Attribute):
        loc = self._parse_location(e)
        value = self._parse_expr(e.value)
        return Attribute(value, e.attr, loc)

    def _parse_expr(self, e: ast.expr) -> Expr:
        """Parse a Python expression."""
        loc = self._parse_location(e)
        match e:
            case ast.Name():
                ident = self._parse_id(e)
                return Var(ident, loc)
            case ast.Constant():
                return self._parse_constant(e)
            case ast.BoolOp():
                return self._parse_boolop(e)
            case ast.UnaryOp():
                return self._parse_unaryop(e)
            case ast.BinOp():
                return self._parse_binop(e)
            case ast.Compare():
                return self._parse_compare(e)
            case ast.Call():
                return self._parse_call(e)
            case ast.Attribute():
                return self._parse_attribute(e)
            case ast.Tuple():
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.List():
                return ListExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.ListComp():
                targets: list[Id | TupleBinding] = []
                iterables: list[Expr] = []
                for gen in e.generators:
                    target, iterable = self._parse_comprehension(gen, loc)
                    targets.append(target)
                    iterables.append(iterable)
                elt = self._parse_expr(e.elt)
                return ListComp(targets, iterables, elt, loc)
            case ast.Subscript():
                return self._parse_subscript(e)
            case ast.IfExp():
                cond = self._parse_expr(e.test)
                ift = self._parse_expr(e.body)
                iff = self._parse_expr(e.orelse)
                return IfExpr(cond, ift, iff, loc)
            case _:
                raise FPyParserError(loc, 'expression is unsupported in FPy', e)

    def _parse_tuple_target(self, target: ast.expr, e: ast.AST):
        loc = self._parse_location(target)
        match target:
            case ast.Name():
                return self._parse_id(target)
            case ast.Tuple():
                elts = [self._parse_tuple_target(elt, e) for elt in target.elts]
                return TupleBinding(elts, loc)
            case _:
                raise FPyParserError(loc, 'FPy expects an identifier', target, e)       

    def _parse_comprehension(self, gen: ast.comprehension, loc: Location):
        if gen.is_async:
            raise FPyParserError(loc, 'FPy does not support async comprehensions', gen)
        if gen.ifs != []:
            raise FPyParserError(loc, 'FPy does not support if conditions in comprehensions', gen)
        target = self._parse_tuple_target(gen.target, gen)
        iterable = self._parse_expr(gen.iter)
        return target, iterable

    def _parse_contextdata(self, e: ast.expr):
        loc = self._parse_location(e)
        match e:
            case ast.Constant():
                if isinstance(e.value, str):
                    return e.value
                else:
                    return self._parse_constant(e)
            case ast.List() | ast.Tuple():
                return [self._parse_contextdata(elt) for elt in e.elts]
            case ast.Name():
                return self._parse_id(e)
            case _:
                raise FPyParserError(loc, 'unexpected FPy context data', e)

    def _parse_contextname(self, item: ast.withitem):
        var = item.optional_vars
        match var:
            case None:
                return UnderscoreId()
            case ast.Name():
                return NamedId(var.id)
            case _:
                loc = self._parse_location(var)
                raise FPyParserError(loc, '`Context` can only be optionally bound to an identifier`', var, item)

    def _parse_augassign(self, stmt: ast.AugAssign):
        loc = self._parse_location(stmt)
        if not isinstance(stmt.target, ast.Name):
            raise FPyParserError(loc, 'Unsupported target in FPy', stmt)

        ident = self._parse_id(stmt.target)
        if not isinstance(ident, NamedId):
            raise FPyParserError(loc, 'Not a valid FPy identifier', stmt)

        match stmt.op:
            case ast.Add():
                value = self._parse_expr(stmt.value)
                e: Expr = Add(Var(ident, loc), value, loc)
            case ast.Sub():
                value = self._parse_expr(stmt.value)
                e = Sub(Var(ident, loc), value, loc)
            case ast.Mult():
                value = self._parse_expr(stmt.value)
                e = Mul(Var(ident, loc), value, loc)
            case ast.Div():
                value = self._parse_expr(stmt.value)
                e = Div(Var(ident, loc), value, loc)
            case _:
                raise FPyParserError(loc, 'Unsupported operator-assignment in FPy', stmt)

        return Assign(ident, None, e, loc)

    def _parse_statement(self, stmt: ast.stmt) -> Stmt:
        """Parse a Python statement."""
        loc = self._parse_location(stmt)
        match stmt:
            case ast.AugAssign():
                return self._parse_augassign(stmt)
            case ast.AnnAssign():
                if not isinstance(stmt.target, ast.Name):
                    raise FPyParserError(loc, 'Unsupported target in FPy', stmt)
                if stmt.annotation is None:
                    raise FPyParserError(loc, 'FPy requires a type annotation', stmt)
                if stmt.value is None:
                    raise FPyParserError(loc, 'FPy requires a value', stmt)

                ident = self._parse_id(stmt.target)
                ty = self._parse_type_annotation(stmt.annotation)
                value = self._parse_expr(stmt.value)
                return Assign(ident, ty, value, loc)
            case ast.Assign():
                if len(stmt.targets) != 1:
                    raise FPyParserError(loc, 'FPy only supports single assignment', stmt)
                target = stmt.targets[0]
                match target:
                    case ast.Name():
                        ident = self._parse_id(target)
                        value = self._parse_expr(stmt.value)
                        return Assign(ident, None, value, loc)
                    case ast.Tuple():
                        binding = self._parse_tuple_target(target, stmt)
                        value = self._parse_expr(stmt.value)
                        return Assign(binding, None, value, loc)
                    case ast.Subscript():
                        var, slices = self._parse_subscript_target(target)
                        if not isinstance(var, Var):
                            raise FPyParserError(loc, 'FPy expects a variable', target, stmt)
                        value = self._parse_expr(stmt.value)
                        return IndexedAssign(var.name, slices, value, loc)
                    case _:
                        raise FPyParserError(loc, 'Unexpected binding type', stmt)
            case ast.If():
                cond = self._parse_expr(stmt.test)
                ift = self._parse_statements(stmt.body)
                if stmt.orelse == []:
                    return If1Stmt(cond, ift, loc)
                else:
                    iff = self._parse_statements(stmt.orelse)
                    return IfStmt(cond, ift, iff, loc)
            case ast.While():
                if stmt.orelse != []:
                    raise FPyParserError(loc, 'FPy does not support else clause in while statement', stmt)
                cond = self._parse_expr(stmt.test)
                block = self._parse_statements(stmt.body)
                return WhileStmt(cond, block, loc)
            case ast.For():
                if stmt.orelse != []:
                    raise FPyParserError(loc, 'FPy does not support else clause in for statement', stmt)
                for_target = self._parse_tuple_target(stmt.target, stmt)
                iterable = self._parse_expr(stmt.iter)
                block = self._parse_statements(stmt.body)
                return ForStmt(for_target, iterable, block, loc)
            case ast.Return():
                if stmt.value is None:
                    raise FPyParserError(loc, 'Return statement must have value', stmt)
                e = self._parse_expr(stmt.value)
                return ReturnStmt(e, loc)
            case ast.With():
                if len(stmt.items) != 1:
                    raise FPyParserError(loc, 'FPy only supports with statements with a single item', stmt)
                item = stmt.items[0]
                name = self._parse_contextname(item)
                ctx = self._parse_expr(item.context_expr)
                block = self._parse_statements(stmt.body)
                return ContextStmt(name, ctx, block, loc)
            case ast.Assert():
                test = self._parse_expr(stmt.test)
                if stmt.msg is None:
                    return AssertStmt(test, None, loc)
                else:
                    msg = self._parse_expr(stmt.msg)
                    return AssertStmt(test, msg, loc)
            case ast.Expr():
                e = self._parse_expr(stmt.value)
                return EffectStmt(e, loc)
            case ast.Pass():
                return PassStmt(loc)
            case _:
                raise FPyParserError(loc, 'statement is unsupported in FPy', stmt)

    def _parse_statements(self, stmts: list[ast.stmt]):
        """Parse a list of Python statements."""
        return StmtBlock([self._parse_statement(s) for s in stmts])

    def _parse_arguments(self, pos_args: list[ast.arg]):
        args: list[Argument] = []
        for arg in pos_args:
            if arg.arg == '_':
                ident: Id = UnderscoreId()
            else:
                loc = self._parse_location(arg)
                ident = SourceId(arg.arg, loc)

            if arg.annotation is None:
                args.append(Argument(ident, AnyTypeAnn(loc), loc))
            else:
                ty = self._parse_type_annotation(arg.annotation)
                args.append(Argument(ident, ty, loc))

        return args

    def _parse_returns(self, e: ast.expr):
        return self._parse_type_annotation(e)

    def _parse_lambda(self, f: ast.Lambda):
        """Parse a Python lambda expression."""
        loc = self._parse_location(f)
        args = self._parse_arguments(f.args.args)
        expr = self._parse_expr(f.body)
        block = StmtBlock([ReturnStmt(expr, expr.loc)])
        return FuncDef('pre', args, set(), None, block, None, {}, ForeignEnv.default(), loc=loc)

    def _parse_function(self, f: ast.FunctionDef, env: ForeignEnv):
        """Parse a Python function definition."""
        loc = self._parse_location(f)

        # check arguments are only positional
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise FPyParserError(loc, 'FPy does not support variadic arguments', f, f.args.vararg)
        if f.args.kwarg:
            raise FPyParserError(loc, 'FPy does not support keyword arguments', f, f.args.kwarg)

        # description
        docstring = ast.get_docstring(f)
        if docstring is not None:
            body = f.body[1:]
        else:
            body = f.body

        # parse arguments and body
        args = self._parse_arguments(pos_args)
        block = self._parse_statements(body)

        # return AST and decorator list
        func = FuncDef(f.name, args, set(), None, block, None, {}, env, loc=loc)
        return func, f.decorator_list

    def _eval(
        self,
        e: ast.expr,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, object] | None = None
    ):
        globals = None if globals is None else dict(globals)
        return eval(ast.unparse(e), globals, locals)

    def _start_parse(self):
        start_loc = Location(self.name, self.start_line, 0, self.start_line, 0)

        mod = ast.parse(self.source, self.name)
        if len(mod.body) > 1:
            raise FPyParserError(start_loc, 'FPy only supports single function definitions', mod)

        ptree = mod.body[0]
        if not isinstance(ptree, ast.FunctionDef):
            raise FPyParserError(start_loc, 'FPy only supports single function definitions', mod)

        return ptree


    def parse_function(self, env: ForeignEnv):
        """Parses `self.source` as an FPy `FunctionDef`."""
        ptree = self._start_parse()
        return self._parse_function(ptree, env)

    def parse_signature(self, ignore_ctx: bool = False):
        """Parses `self.source` to extract the arguments."""
        f = self._start_parse()
        loc = self._parse_location(f)

        # check arguments are only positional
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise FPyParserError(loc, 'FPy does not support variadic arguments', f, f.args.vararg)
        if f.args.kwarg:
            raise FPyParserError(loc, 'FPy does not support keyword arguments', f, f.args.kwarg)

        # check that there's a return annotation
        if f.returns is None:
            raise FPyParserError(loc, 'FPy requires a return annotation', f, f.returns)

        # prune context argument
        if ignore_ctx and len(pos_args) >= 1:
            if pos_args[-1].arg == 'ctx':
                pos_args = pos_args[:-1]

        # parse arguments and returns
        args = self._parse_arguments(pos_args)
        returns = self._parse_returns(f.returns)

        arg_types = [arg.type for arg in args]
        return arg_types, returns

    def find_decorator(
        self,
        decorator_list: list[ast.expr],
        decorator: Any,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, object] | None = None
    ):
        """Returns the decorator AST for a particular decorator"""
        for dec in reversed(decorator_list):
            match dec:
                case ast.Call():
                    f = self._eval(dec.func, globals=globals, locals=locals)
                    if isinstance(f, FunctionType) and f == decorator:
                        return dec
                case ast.Name() | ast.Attribute():
                    f = self._eval(dec, globals=globals, locals=locals)
                    if isinstance(f, FunctionType) and f == decorator:
                        return dec

        raise RuntimeError('unreachable')

    # reparse
    def parse_decorator(self, decorator: ast.expr) -> dict[str, Any]:
        """
        (Re)-parses the `@fpy` decorator.

        Returns a `dict` where each key is only the set of keywords that must be parsed.
        Supported keywords include:

        - `pre`: a precondition expression
        """
        match decorator:
            case ast.Name() | ast.Attribute():
                return {}
            case ast.Call():
                if decorator.args != []:
                    loc = self._parse_location(decorator)
                    raise FPyParserError(loc, 'FPy decorators do not accept arguments', decorator)

                props: dict[str, Any] = {}
                for kwd in decorator.keywords:
                    match kwd.arg:
                        case 'pre':
                            # TODO: check arguments are a strict subset?
                            if not isinstance(kwd.value, ast.Lambda):
                                loc = self._parse_location(kwd.value)
                                raise FPyParserError(loc, 'FPy `pre` expects a lambda expression', kwd.value)
                            props['pre'] = self._parse_lambda(kwd.value)
                return props
            case _:
                raise NotImplementedError('unsupported decorator', decorator)
