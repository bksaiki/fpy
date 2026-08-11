"""
This module contains the parser for the FPy language.
"""

import ast
from collections.abc import Callable, Mapping
from typing import Any

from ..ast.fpyast import *
from ..env import ForeignEnv
from ..number import Float, Real
from ..ops import *
from ..utils import NamedId, SourceId, UnderscoreId

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
    abs: Abs,
    fabs: Abs,
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
    round_exact: Cast,
    cast: Cast,
    len: Len,
    dim: Dim,
    fst: Fst,
    snd: Snd,
    enumerate: Enumerate,
    sum: Sum,
    any: AnyOf,
    all: AllOf,
    logb: Logb,
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
    empty: Empty,
}

# Dispatch for fixed-arity builtin operators: (table, arg count, `func`-carrying base).
# `NullaryOp` always takes `func`, so it is its own "named" base.
_fixed_arity_tables: list[tuple[Mapping[Callable, type], int, type]] = [
    (_nullary_table, 0, NullaryOp),
    (_unary_table, 1, NamedUnaryOp),
    (_binary_table, 2, NamedBinaryOp),
    (_ternary_table, 3, NamedTernaryOp),
]

# Python binary operators, shared by `_parse_binop` and `_parse_augassign`.
_binop_table: dict[type[ast.operator], type[BinaryOp]] = {
    ast.Add: Add,
    ast.Sub: Sub,
    ast.Mult: Mul,
    ast.Div: Div,
    ast.Mod: Mod,
    ast.Pow: Pow,
}

def _make_binop(cls: type[BinaryOp], lhs: Expr, rhs: Expr, loc: Location) -> Expr:
    """Construct a `_binop_table` node; operator syntax carries no `func`."""
    if issubclass(cls, NamedBinaryOp):
        return cls(None, lhs, rhs, loc)
    else:
        return cls(lhs, rhs, loc)

# Python comparison operators.
_cmpop_table: dict[type[ast.cmpop], CompareOp] = {
    ast.Lt: CompareOp.LT,
    ast.LtE: CompareOp.LE,
    ast.GtE: CompareOp.GE,
    ast.Gt: CompareOp.GT,
    ast.Eq: CompareOp.EQ,
    ast.NotEq: CompareOp.NE,
}


class FPyParserError(Exception):
    """Parser error for FPy"""


class Parser:
    """
    FPy parser.

    Converts a Python AST (from the `ast` module) to a FPy AST.
    """

    name: str
    env: ForeignEnv
    lines: list[str]
    start_line: int
    col_offset: int

    def __init__(
        self,
        name: str,
        lines: list[str],
        env: ForeignEnv,
        start_line: int = 1,
        col_offset: int = 0
    ):
        self.name = name
        self.env = env
        self.lines = lines
        self.start_line = start_line
        self.col_offset = col_offset

    def _parse_location(self, e: ast.expr | ast.stmt | ast.arg) -> Location:
        """Extracts the parse location of a Python AST node."""
        assert e.end_lineno is not None, "missing end line number"
        assert e.end_col_offset is not None, "missing end column offset"

        return Location(
            self.name,
            e.lineno + self.start_line,
            e.col_offset + self.col_offset,
            e.end_lineno + self.start_line,
            e.end_col_offset + self.col_offset + 1
        )

    def _parse_error(self, why: str, where: ast.AST, ctx: ast.AST | None = None):
        msg_lines = [why]
        # Operator nodes (`ast.operator`, `ast.cmpop`, `ast.boolop`, ...) carry
        # no source position, so fall back to the containing expression `ctx`
        # rather than reporting an unhelpful `?:?`.
        loc_node = where if isinstance(where, ast.expr | ast.stmt | ast.arg) else ctx
        if isinstance(loc_node, ast.expr | ast.stmt | ast.arg):
            loc = self._parse_location(loc_node)
            msg_lines.append(f' at: {self.name}:{loc.start_line}:{loc.start_column}')
        else:
            msg_lines.append(f' at: {self.name}:?:?')

        # `ast.unparse` of a bare operator node is empty; omit the noise.
        where_src = ast.unparse(where)
        if where_src:
            msg_lines.append(f' where: {where_src}')
        if ctx is not None:
            msg_lines.append(f' in: {ast.unparse(ctx)}')

        return FPyParserError('\n'.join(msg_lines))

    def _convert_type(self, ty, loc: Location):
        if ty == Real:
            return RealTypeAnn(None, loc)
        elif isinstance(ty, type):
            if issubclass(ty, bool):
                return BoolTypeAnn(loc)
            elif issubclass(ty, int) or issubclass(ty, float):
                # TODO: more specific type
                return RealTypeAnn(None, loc)
            elif issubclass(ty, Float):
                return RealTypeAnn(None, loc)
            else:
                # TODO: implement
                return AnyTypeAnn(loc)
        elif isinstance(ty, tuple):
            elts = [self._convert_type(elt, loc) for elt in ty]
            return TupleTypeAnn(elts, loc)
        elif isinstance(ty, list):
            elt = self._convert_type(ty[0], loc)
            return ListTypeAnn(elt, None, loc)
        else:
            # TODO: implement
            return AnyTypeAnn(loc)

    def _eval_type_annotation(self, ann: ast.expr):
        match ann:
            case ast.Attribute():
                attr = self._parse_attribute(ann)
                return self._eval_attribute(attr, ann)
            case ast.Name():
                ident = self._parse_id(ann)
                if isinstance(ident, UnderscoreId):
                    raise self._parse_error('FPy function call must begin with a named identifier', ann)
                ident_str = str(ident)
                if ident_str not in self.env:
                    raise self._parse_error(f'name \'{ident_str}\' not defined', ann)
                return self.env[ident_str]
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
                raise self._parse_error('Unsupported constant', e)

    def _parse_hexfloat(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 1:
            raise self._parse_error('FPy `hexfloat` expects one argument', e)
        arg = self._parse_expr(e.args[0])
        if not isinstance(arg, ForeignVal):
            raise self._parse_error('FPy `hexfloat` expects a string', e)
        return Hexnum(func, arg.val, loc)

    def _parse_rational(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 2:
            raise self._parse_error('FPy `rational` expects two arguments', e)
        p = self._parse_expr(e.args[0])
        if not isinstance(p, Integer):
            raise self._parse_error('FPy `rational` expects an integer as first argument', e)
        q = self._parse_expr(e.args[1])
        if not isinstance(q, Integer):
            raise self._parse_error('FPy `rational` expects an integer as second argument', e)
        return Rational(func, p.val, q.val, loc)

    def _parse_digits(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        if len(e.args) != 3:
            raise self._parse_error('FPy `digits` expects three arguments', e)
        m_e = self._parse_expr(e.args[0])
        if not isinstance(m_e, Integer):
            raise self._parse_error('FPy `digits` expects an integer as first argument', e)
        e_e = self._parse_expr(e.args[1])
        if not isinstance(e_e, Integer):
            raise self._parse_error('FPy `digits` expects an integer as second argument', e)
        b_e = self._parse_expr(e.args[2])
        if not isinstance(b_e, Integer):
            raise self._parse_error('FPy `digits` expects an integer as third argument', e)
        return Digits(func, m_e.val, e_e.val, b_e.val, loc)

    def _parse_range(self, e: ast.Call, func: FuncSymbol):
        loc = self._parse_location(e)
        match len(e.args):
            case 1:
                # range(stop)
                stop = self._parse_expr(e.args[0])
                return Range1(func, stop, loc)
            case 2:
                # range(start, stop)
                start = self._parse_expr(e.args[0])
                stop = self._parse_expr(e.args[1])
                return Range2(func, start, stop, loc)
            case 3:
                # range(start, stop, step)
                start = self._parse_expr(e.args[0])
                stop = self._parse_expr(e.args[1])
                step = self._parse_expr(e.args[2])
                return Range3(func, start, stop, step, loc)
            case _:
                raise self._parse_error('FPy `range` expects 1, 2, or 3 arguments', e)

    def _parse_boolop(self, e: ast.BoolOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.And():
                args = [self._parse_expr(v) for v in e.values]
                return And(args, loc)
            case ast.Or():
                args = [self._parse_expr(v) for v in e.values]
                return Or(args, loc)
            case _:
                raise self._parse_error('Not a valid FPy operator', e.op, e)

    def _parse_unaryop(self, e: ast.UnaryOp):
        loc = self._parse_location(e)
        match e.op:
            case ast.UAdd():
                return self._parse_expr(e.operand)
            case ast.USub():
                arg = self._parse_expr(e.operand)
                if isinstance(arg, RationalVal) and arg.as_rational() == 0:
                    # Negating a zero literal yields negative zero, a signed
                    # literal — fold it here so the sign survives regardless of
                    # context (a `Neg` under REAL loses it). See `as_real`.
                    return Decnum('-0.0', loc)
                elif isinstance(arg, Integer):
                    return Integer(-arg.val, loc)
                else:
                    return Neg(arg, loc)
            case ast.Not():
                arg = self._parse_expr(e.operand)
                return Not(arg, loc)
            case _:
                raise self._parse_error('Not a valid FPy operator', e.op, e)

    def _parse_binop(self, e: ast.BinOp):
        loc = self._parse_location(e)
        cls = _binop_table.get(type(e.op))
        if cls is None:
            raise self._parse_error('Not a valid FPy operator', e.op, e)
        lhs = self._parse_expr(e.left)
        rhs = self._parse_expr(e.right)
        return _make_binop(cls, lhs, rhs, loc)

    def _parse_cmpop(self, op: ast.cmpop, e: ast.Compare):
        result = _cmpop_table.get(type(op))
        if result is None:
            raise self._parse_error('Not a valid FPy comparator', op, e)
        return result

    def _parse_compare(self, e: ast.Compare):
        loc = self._parse_location(e)
        ops = [self._parse_cmpop(op, e) for op in e.ops]
        args = [self._parse_expr(operand) for operand in [e.left, *e.comparators]]
        return Compare(ops, args, loc)

    def _eval_var(self, v: Var, e: ast.expr):
        ident = str(v.name)
        if ident not in self.env:
            raise self._parse_error(f'name \'{ident}\' not defined', e)
        return self.env[ident]

    def _eval_attribute(self, a: Attribute, e: ast.expr):
        match a.value:
            case Var():
                # evaluating `x.y`
                base = self._eval_var(a.value, e)
            case Attribute():
                # evaluating `x.y.z` where `x.y` is `a`
                base = self._eval_attribute(a.value, e)
            case _:
                raise self._parse_error('FPy only supports attribute access on variables', e)

        # lookup the attribute
        if not hasattr(base, a.attr):
            raise self._parse_error(f'unknown attribute \'{a.attr}\' for {base}', e)

        return getattr(base, a.attr)

    def _check_no_kwargs(self, fn, kwargs: list, e: ast.Call):
        if kwargs:
            raise self._parse_error(f'FPy does not support keyword arguments for `{fn}`', e)

    def _check_arity(self, fn, args: list, arity: int, e: ast.Call):
        if len(args) != arity:
            noun = 'argument' if arity == 1 else 'arguments'
            raise self._parse_error(f'FPy expects {arity} {noun} for `{fn}`, got {len(args)}', e)

    def _parse_nary(self, fn, func: Var, args: list[Expr], kwargs: list, loc: Location, e: ast.Call):
        cls = _nary_table[fn]
        # min/max/fmin/fmax split at parse time by arity: 1 arg →
        # reduce-form (AMin/AMax over a list), ≥ 2 args → variadic
        # scalar form.  See AMin/AMax docstrings for the rationale.
        if cls is Min and len(args) == 1:
            return AMin(func, args[0], loc)
        if cls is Max and len(args) == 1:
            return AMax(func, args[0], loc)
        if (cls is Min or cls is Max) and len(args) == 0:
            raise self._parse_error(f'FPy expects at least 1 argument for `{fn}`', e)
        if cls is Empty and len(args) < 1:
            raise self._parse_error(f'FPy expects at least 1 argument for `{fn}`', e)
        self._check_no_kwargs(fn, kwargs, e)
        if issubclass(cls, NamedNaryOp):
            return cls(func, args, loc)
        else:
            return cls(args, loc)

    def _parse_call(self, e: ast.Call):
        """Parse a Python call function."""
        # parse function expression
        match e.func:
            case ast.Attribute():
                func = self._parse_attribute(e.func)
                fn = self._eval_attribute(func, e.func)
            case ast.Name():
                name = self._parse_id(e.func)
                if isinstance(name, UnderscoreId):
                    raise self._parse_error('FPy function call must begin with a named identifier', e)
                func = Var(name, None)
                ident = str(name)
                if ident not in self.env:
                    raise self._parse_error(f'name \'{ident}\' not defined', e)
                fn = self.env[ident]
            case _:
                raise self._parse_error('unsupported call target in FPy', e.func, e)

        # parse arguments
        args = [self._parse_expr(arg) for arg in e.args]

        # parse keyword arguments
        kwargs: list[tuple[str, Expr]] = []
        for kwarg in e.keywords:
            if kwarg.arg is None:
                raise self._parse_error('FPy does not support **kwargs', e)
            kwarg_val = self._parse_expr(kwarg.value)
            kwargs.append((kwarg.arg, kwarg_val))

        loc = self._parse_location(e.func)

        # fixed-arity builtin operators
        for table, arity, named_base in _fixed_arity_tables:
            if fn in table:
                cls: Any = table[fn]
                self._check_arity(fn, args, arity, e)
                self._check_no_kwargs(fn, kwargs, e)
                if issubclass(cls, named_base):
                    return cls(func, *args, loc)
                else:
                    return cls(*args, loc)

        # variadic builtin operators (min/max/empty/zip/...) with bespoke arity rules
        if fn in _nary_table:
            return self._parse_nary(fn, func, args, kwargs, loc, e)

        # builtins with bespoke argument parsing
        bespoke = {
            rational: self._parse_rational,
            hexfloat: self._parse_hexfloat,
            digits: self._parse_digits,
            range: self._parse_range,
        }
        if fn in bespoke:
            self._check_no_kwargs(fn, kwargs, e)
            return bespoke[fn](e, func)

        # user-defined / foreign call
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
            raise self._parse_error('FPy does not support slice step', e.step)

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

    def _parse_attribute(self, e: ast.Attribute):
        loc = self._parse_location(e)
        value = self._parse_expr(e.value)
        return Attribute(value, e.attr, loc)

    def _parse_expr(self, e: ast.expr) -> Expr:
        """Parse a Python expression."""
        match e:
            case ast.Name():
                ident = self._parse_id(e)
                loc = self._parse_location(e)
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
                loc = self._parse_location(e)
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.List():
                loc = self._parse_location(e)
                return ListExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.ListComp():
                targets: list[Id | TupleBinding] = []
                iterables: list[Expr] = []
                for gen in e.generators:
                    target, iterable = self._parse_comprehension(gen)
                    targets.append(target)
                    iterables.append(iterable)
                elt = self._parse_expr(e.elt)
                loc = self._parse_location(e)
                return ListComp(targets, iterables, elt, loc)
            case ast.Subscript():
                return self._parse_subscript(e)
            case ast.IfExp():
                cond = self._parse_expr(e.test)
                ift = self._parse_expr(e.body)
                iff = self._parse_expr(e.orelse)
                loc = self._parse_location(e)
                return IfExpr(cond, ift, iff, loc)
            case _:
                raise self._parse_error('expression is unsupported in FPy', e)

    def _parse_tuple_target(self, target: ast.expr, e: ast.AST):
        loc = self._parse_location(target)
        match target:
            case ast.Name():
                return self._parse_id(target)
            case ast.Tuple():
                elts = [self._parse_tuple_target(elt, e) for elt in target.elts]
                return TupleBinding(elts, loc)
            case _:
                raise self._parse_error('FPy expects an identifier', target, e)

    def _parse_comprehension(self, gen: ast.comprehension):
        if gen.is_async:
            raise self._parse_error('FPy does not support async comprehensions', gen)
        if gen.ifs != []:
            raise self._parse_error('FPy does not support if conditions in comprehensions', gen)
        target = self._parse_tuple_target(gen.target, gen)
        iterable = self._parse_expr(gen.iter)
        return target, iterable

    def _parse_contextdata(self, e: ast.expr):
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
                raise self._parse_error('unexpected FPy context data', e)

    def _parse_contextname(self, item: ast.withitem):
        var = item.optional_vars
        match var:
            case None:
                return UnderscoreId()
            case ast.Name():
                return NamedId(var.id)
            case _:
                raise self._parse_error('`Context` can only be optionally bound to an identifier`', var, item)

    def _parse_augassign(self, stmt: ast.AugAssign):
        loc = self._parse_location(stmt)
        if not isinstance(stmt.target, ast.Name):
            raise self._parse_error('Unsupported target in FPy', stmt)

        ident = self._parse_id(stmt.target)
        if not isinstance(ident, NamedId):
            raise self._parse_error('Not a valid FPy identifier', stmt)

        cls = _binop_table.get(type(stmt.op))
        if cls is None:
            raise self._parse_error('Unsupported operator-assignment in FPy', stmt)

        value = self._parse_expr(stmt.value)
        e = _make_binop(cls, Var(ident, loc), value, loc)
        return Assign(ident, None, e, loc)

    def _parse_statement(self, stmt: ast.stmt) -> Stmt:
        """Parse a Python statement."""
        loc = self._parse_location(stmt)
        match stmt:
            case ast.AugAssign():
                return self._parse_augassign(stmt)
            case ast.AnnAssign():
                if not isinstance(stmt.target, ast.Name):
                    raise self._parse_error('Unsupported target in FPy', stmt)
                if stmt.annotation is None:
                    raise self._parse_error('FPy requires a type annotation', stmt)
                if stmt.value is None:
                    raise self._parse_error('FPy requires a value', stmt)

                ident = self._parse_id(stmt.target)
                ty = self._parse_type_annotation(stmt.annotation)
                value = self._parse_expr(stmt.value)
                return Assign(ident, ty, value, loc)
            case ast.Assign():
                if len(stmt.targets) != 1:
                    raise self._parse_error('FPy only supports single assignment', stmt)
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
                            raise self._parse_error('FPy expects a variable', target, stmt)
                        value = self._parse_expr(stmt.value)
                        return IndexedAssign(var.name, slices, value, loc)
                    case _:
                        raise self._parse_error('Unexpected binding type', stmt)
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
                    raise self._parse_error('FPy does not support else clause in while statement', stmt)
                cond = self._parse_expr(stmt.test)
                block = self._parse_statements(stmt.body)
                return WhileStmt(cond, block, loc)
            case ast.For():
                if stmt.orelse != []:
                    raise self._parse_error('FPy does not support else clause in for statement', stmt)
                for_target = self._parse_tuple_target(stmt.target, stmt)
                iterable = self._parse_expr(stmt.iter)
                block = self._parse_statements(stmt.body)
                return ForStmt(for_target, iterable, block, loc)
            case ast.Return():
                if stmt.value is None:
                    raise self._parse_error('Return statement must have value', stmt)
                e = self._parse_expr(stmt.value)
                return ReturnStmt(e, loc)
            case ast.With():
                if len(stmt.items) != 1:
                    raise self._parse_error('FPy only supports with statements with a single item', stmt)
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
                raise self._parse_error('statement is unsupported in FPy', stmt)

    def _parse_statements(self, stmts: list[ast.stmt]):
        """Parse a list of Python statements."""
        return StmtBlock([self._parse_statement(s) for s in stmts])

    def _parse_arguments(self, pos_args: list[ast.arg]):
        args: list[Argument] = []
        for arg in pos_args:
            loc = self._parse_location(arg)
            if arg.arg == '_':
                ident: Id = UnderscoreId()
            else:
                ident = SourceId(arg.arg, loc)

            if arg.annotation is None:
                args.append(Argument(ident, AnyTypeAnn(loc), loc))
            else:
                ty = self._parse_type_annotation(arg.annotation)
                args.append(Argument(ident, ty, loc))

        return args

    def _parse_returns(self, e: ast.expr):
        return self._parse_type_annotation(e)

    def _parse_function(self, f: ast.FunctionDef, env: ForeignEnv):
        """Parse a Python function definition."""
        loc = self._parse_location(f)

        # check arguments are only positional
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise self._parse_error('FPy does not support variadic arguments', f, f.args.vararg)
        if f.args.kwarg:
            raise self._parse_error('FPy does not support keyword arguments', f, f.args.kwarg)

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
        meta = FuncMeta(set(), None, None, {}, env)
        func = FuncDef(f.name, args, block, meta, loc=loc)
        return func, f.decorator_list

    def _start_parse(self):
        src = ''.join(self.lines)
        mod = ast.parse(src, self.name)
        if len(mod.body) != 1:
            raise self._parse_error('FPy only supports single function definitions', mod)

        ptree = mod.body[0]
        if not isinstance(ptree, ast.FunctionDef):
            raise self._parse_error('FPy only supports single function definitions', mod)

        return ptree

    def parse_function(self, env: ForeignEnv):
        """Parses `self.source` as an FPy `FunctionDef`."""
        ptree = self._start_parse()
        return self._parse_function(ptree, env)

    def parse_signature(self, ignore_ctx: bool = False):
        """Parses `self.source` to extract the arguments."""
        f = self._start_parse()

        # check arguments are only positional
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise self._parse_error('FPy does not support variadic arguments', f, f.args.vararg)
        if f.args.kwarg:
            raise self._parse_error('FPy does not support keyword arguments', f, f.args.kwarg)

        # check that there's a return annotation
        if f.returns is None:
            raise self._parse_error('FPy requires a return annotation', f, f.returns)

        # prune context argument
        if ignore_ctx and len(pos_args) >= 1 and pos_args[-1].arg == 'ctx':
            pos_args = pos_args[:-1]

        # parse arguments and returns
        args = self._parse_arguments(pos_args)
        returns = self._parse_returns(f.returns)

        arg_types = [arg.type for arg in args]
        return arg_types, returns
