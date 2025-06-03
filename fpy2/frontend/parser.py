"""
This module contains the parser for the FPy language.
"""

import ast

from typing import Mapping
from types import FunctionType

from ..ast.fpyast import *
from ..utils import NamedId, UnderscoreId, SourceId

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

_constants: set[str] = {
    'PI',
    'E',
    'NAN',
    'INFINITY',
    'LOG2E',
    'LOG10E',
    'LN2',
    'PI_2',
    'PI_4',
    'M_1_PI',
    'M_2_PI',
    'M_2_SQRTPI',
    'SQRT2',
    'SQRT1_2'
}

_unary_table: dict[str, type[UnaryOp]] = {
    'abs': Fabs,
    'fabs': Fabs,
    'sqrt': Sqrt,
    'cbrt': Cbrt,
    'ceil': Ceil,
    'floor': Floor,
    'nearbyint': NearbyInt,
    'round': Round,
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
    'not': Not,
    'cast': Cast,
    'range': Range,
    'shape': Shape,
    'dim': Dim,
}

_binary_table: dict[str, type[BinaryOp]] = {
    'add': Add,
    'sub': Sub,
    'mul': Mul,
    'div': Div,
    'copysign': Copysign,
    'fdim': Fdim,
    'fmax': Fmax,
    'min': Fmin,
    'max': Fmax,
    'fmin': Fmin,
    'fmod': Fmod,
    'remainder': Remainder,
    'hypot': Hypot,
    'atan2': Atan2,
    'pow': Pow,
    'size': Size
}

_ternary_table: dict[str, type[TernaryOp]] = {
    'fma': Fma
}

_nary_table: dict[str, type[NaryOp]] = {
    'and': And,
    'or': Or,
    'zip': Zip,
}

_special_functions = {
    'digits',
    'hexfloat',
    'rational'
}

class FPyParserError(Exception):
    """Parser error for FPy"""
    loc: Location
    why: str
    where: ast.AST
    ctx: Optional[ast.AST]

    def __init__(
        self,
        loc: Location,
        why: str,
        where: ast.AST,
        ctx: Optional[ast.AST] = None
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


class Parser:
    """
    FPy parser.

    Converts a Python AST (from the `ast` module) to a FPy AST.
    """

    name: str
    source: str
    lines: list[str]
    start_line: int

    def __init__(
        self,
        name: str, 
        source: str,
        start_line: int = 1
    ):
        self.name = name
        self.source = source
        self.lines = source.splitlines()
        self.start_line = start_line

    def parse_function(self):
        """Parses `self.source` as an FPy `FunctionDef`."""
        start_loc = Location(self.name, self.start_line, 0, self.start_line, 0)

        mod = ast.parse(self.source, self.name)
        if len(mod.body) > 1:
            raise FPyParserError(start_loc, 'FPy only supports single function definitions', mod)

        ptree = mod.body[0]
        if not isinstance(ptree, ast.FunctionDef):
            raise FPyParserError(start_loc, 'FPy only supports single function definitions', mod)

        return self._parse_function(ptree)

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


    def _parse_type_annotation(self, ann: ast.expr) -> TypeAnn:
        loc = self._parse_location(ann)
        match ann:
            case ast.Name('Real'):
                return RealTypeAnn(loc)
            case ast.Name('int'):
                # TODO: more specific type
                return RealTypeAnn(loc)
            case ast.Name('float'):
                return RealTypeAnn(loc)
            case ast.Name('bool'):
                return RealTypeAnn(loc)
            case _:
                # TODO: implement
                return AnyTypeAnn(loc)

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
            case _:
                raise FPyParserError(loc, 'Unsupported constant', e)

    def _parse_hexfloat(self, e: ast.Call):
        loc = self._parse_location(e)
        if len(e.args) != 1:
            raise FPyParserError(loc, 'FPy `hexfloat` expects one argument', e)
        arg = self._parse_expr(e.args[0])
        if not isinstance(arg, ForeignVal):
            raise FPyParserError(loc, 'FPy `hexfloat` expects a string', e)
        return Hexnum(arg.val, loc)

    def _parse_rational(self, e: ast.Call):
        loc = self._parse_location(e)
        if len(e.args) != 2:
            raise FPyParserError(loc, 'FPy `rational` expects two arguments', e)
        p = self._parse_expr(e.args[0])
        if not isinstance(p, Integer):
            raise FPyParserError(loc, 'FPy `rational` expects an integer as first argument', e)
        q = self._parse_expr(e.args[1])
        if not isinstance(q, Integer):
            raise FPyParserError(loc, 'FPy `rational` expects an integer as second argument', e)
        return Rational(p.val, q.val, loc)

    def _parse_digits(self, e: ast.Call):
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
        return Digits(m_e.val, e_e.val, b_e.val, loc)

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

    def _parse_call(self, e: ast.Call) -> str:
        """Parse a Python call expression."""
        loc = self._parse_location(e)
        if not isinstance(e.func, ast.Name):
            raise FPyParserError(loc, 'Unsupported call expression', e)
        return e.func.id

    def _parse_slice(self, slice: ast.expr, e: ast.expr):
        match slice:
            case ast.slice():
                loc = self._parse_location(e)
                raise  FPyParserError(loc, 'Slices unsupported', e, slice)
            case ast.Tuple():
                return [self._parse_expr(s) for s in slice.elts]
            case _:
                return [self._parse_expr(slice)]

    def _parse_subscript(self, e: ast.Subscript):
        value = self._parse_expr(e.value)
        slices = self._parse_slice(e.slice, e)
        while isinstance(value, TupleRef):
            v_value, v_slices = value.value, value.slices
            value = v_value
            slices = v_slices + slices
        return (value, slices)

    def _is_foreign_val(self, e: ast.expr):
        match e:
            case ast.Name() | ast.Constant():
                return True
            case ast.Attribute():
                return self._is_foreign_val(e.value)
            case _:
                return False

    def _parse_foreign_attribute(self, e: ast.expr):
        loc = self._parse_location(e)
        match e:
            case ast.Attribute():
                match e.value:
                    case ast.Attribute():
                        val = self._parse_foreign_attribute(e.value)
                        return ForeignAttribute(val.name, [NamedId(e.attr)] + val.attrs, loc)
                    case ast.Name():
                        name = self._parse_id(e.value)
                        if isinstance(name, UnderscoreId):
                            raise FPyParserError(loc, 'FPy foreign attribute must begin with a named identifier', e)
                        return ForeignAttribute(name, [NamedId(e.attr)], loc)
                    case _:
                        raise FPyParserError(loc, 'FPy foreign attribute must begin with an identifier', e)
            case _:
                raise FPyParserError(loc, 'Not a valid foreign attribute', e)

    def _parse_expr(self, e: ast.expr, *, allow_foreign: bool = False) -> Expr:
        """Parse a Python expression."""
        loc = self._parse_location(e)
        match e:
            case ast.Name():
                if e.id in _constants:
                    return Constant(e.id, loc)
                else:
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
                name = self._parse_call(e)
                if name in _unary_table:
                    if len(e.args) != 1:
                        raise FPyParserError(loc, 'FPy unary operator expects one argument', e)
                    arg = self._parse_expr(e.args[0])
                    cls1 = _unary_table[name]
                    return cls1(arg, loc)
                elif name in _binary_table:
                    if len(e.args) != 2:
                        raise FPyParserError(loc, 'FPy binary operator expects two arguments', e)
                    lhs = self._parse_expr(e.args[0])
                    rhs = self._parse_expr(e.args[1])
                    cls2 = _binary_table[name]
                    return cls2(lhs, rhs, loc)
                elif name in _ternary_table:
                    if len(e.args) != 3:
                        raise FPyParserError(loc, 'FPy ternary operator expects three arguments', e)
                    arg0 = self._parse_expr(e.args[0])
                    arg1 = self._parse_expr(e.args[1])
                    arg2 = self._parse_expr(e.args[2])
                    cls3 = _ternary_table[name]
                    return cls3(arg0, arg1, arg2, loc)
                elif name in _nary_table:
                    args = [self._parse_expr(arg) for arg in e.args]
                    clsN = _nary_table[name]
                    return clsN(args, loc)
                elif name == 'len':
                    arg = self._parse_expr(e.args[0])
                    return Size(arg, Integer(0, loc), loc)
                elif name == 'rational':
                    return self._parse_rational(e)
                elif name == 'hexfloat':
                    return self._parse_hexfloat(e)
                elif name == 'digits':
                    return self._parse_digits(e)
                else:
                    return Call(name, [self._parse_expr(arg) for arg in e.args], loc)
            case ast.Tuple():
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.List():
                return TupleExpr([self._parse_expr(e) for e in e.elts], loc)
            case ast.ListComp():
                targets: list[Id | TupleBinding] = []
                iterables: list[Expr] = []
                for gen in e.generators:
                    target, iterable = self._parse_comprehension(gen, loc)
                    targets.append(target)
                    iterables.append(iterable)
                elt = self._parse_expr(e.elt)
                return CompExpr(targets, iterables, elt, loc)
            case ast.Subscript():
                value, slices = self._parse_subscript(e)
                return TupleRef(value, slices, loc)
            case ast.IfExp():
                cond = self._parse_expr(e.test)
                ift = self._parse_expr(e.body)
                iff = self._parse_expr(e.orelse)
                return IfExpr(cond, ift, iff, loc)
            case _:
                if not allow_foreign:
                    raise FPyParserError(loc, 'expression is unsupported in FPy', e)
                return self._parse_foreign_attribute(e)

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
                return var.id
            case _:
                loc = self._parse_location(var)
                raise FPyParserError(loc, '`Context` can only be optionally bound to an identifier`', var, item)

    def _parse_contextexpr(self, item: ast.withitem):
        e = item.context_expr
        loc = self._parse_location(e)
        match e:
            case ast.Name():
                name = self._parse_id(e)
                return Var(name, loc)
            case ast.Call():
                # parse constructor
                match e.func:
                    case ast.Name():
                        name = self._parse_id(e.func)
                        if isinstance(name, UnderscoreId):
                            raise FPyParserError(loc, 'Context constructor must be a named identifier or foreign attribute', e)
                        ctor = Var(name, loc)
                    case ast.Attribute():
                        ctor = self._parse_foreign_attribute(e.func)
                    case _:
                        raise FPyParserError(loc, 'Context constructor must be a named identifier or foreign attribute', e)
                # parse positional arguments
                pos_args: list[Expr] = []
                for arg in e.args:
                    match arg:
                        case ast.Starred():
                            raise FPyParserError(loc, 'FPy does not support starred arguments', arg, e)
                        case _:
                            pos_args.append(self._parse_expr(arg, allow_foreign=True))
                # parse keyword arguments
                kw_args: list[tuple[str, Expr]] = []
                for kwd in e.keywords:
                    if kwd.arg is None:
                        raise FPyParserError(loc, 'FPy does not support unnamed keyword arguments', kwd, e)
                    v = self._parse_expr(kwd.value, allow_foreign=True)
                    kw_args.append((kwd.arg, v))
                return ContextExpr(ctor, pos_args, kw_args, loc)
            case _:
                raise FPyParserError(loc, 'FPy expects a valid context expression', e, item)

    def _parse_augassign(self, stmt: ast.AugAssign):
        loc = self._parse_location(stmt)
        if not isinstance(stmt.target, ast.Name):
            raise FPyParserError(loc, 'Unsupported target in FPy', stmt)

        ident = self._parse_id(stmt.target)
        if not isinstance(ident, NamedId):
            raise FPyParserError(loc, 'Not a valid FPy identifier', stmt)

        match stmt.op:
            case ast.Add():
                cls: type[BinaryOp] = Add
            case ast.Sub():
                cls = Sub
            case ast.Mult():
                cls = Mul
            case ast.Div():
                cls = Div
            case ast.Mod():
                cls = Fmod
            case _:
                raise FPyParserError(loc, 'Unsupported operator-assignment in FPy', stmt)

        value = self._parse_expr(stmt.value)
        e = cls(Var(ident, loc), value, loc)
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
                        var, slices = self._parse_subscript(target)
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
                ctx = self._parse_contextexpr(item)
                block = self._parse_statements(stmt.body)
                return ContextStmt(name, ctx, block, loc)
            case ast.Assert():
                test = self._parse_expr(stmt.test)
                if stmt.msg is not None:
                    raise FPyParserError(loc, 'FPy does not support assert messages', stmt)
                return AssertStmt(test, None, loc)
            case ast.Expr():
                e = self._parse_expr(stmt.value)
                return EffectStmt(e, loc)
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

    def _parse_lambda(self, f: ast.Lambda):
        """Parse a Python lambda expression."""
        loc = self._parse_location(f)
        args = self._parse_arguments(f.args.args)
        expr = self._parse_expr(f.body)
        block = StmtBlock([ReturnStmt(expr, expr.loc)])
        return FuncDef('pre', args, block, loc=loc)

    def _parse_function(self, f: ast.FunctionDef):
        """Parse a Python function definition."""
        loc = self._parse_location(f)

        # check arguments are only positional
        pos_args = f.args.posonlyargs + f.args.args
        if f.args.vararg:
            raise FPyParserError(loc, 'FPy does not support variary arguments', f, f.args.vararg)
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
        func = FuncDef(f.name, args, block, loc=loc)
        return func, f.decorator_list

    def _eval(
        self,
        e: ast.expr,
        globals: Optional[Mapping[str, Any]] = None,
        locals: Optional[Mapping[str, object]] = None
    ):
        globals = None if globals is None else dict(globals)
        return eval(ast.unparse(e), globals, locals)

    def find_decorator(
        self,
        decorator_list: list[ast.expr],
        decorator: Any,
        globals: Optional[Mapping[str, Any]] = None,
        locals: Optional[Mapping[str, object]] = None
    ):
        """Returns the decorator AST for a particular decorator"""
        for dec in reversed(decorator_list):
            match dec:
                case ast.Call():
                    f = self._eval(dec.func, globals=globals, locals=locals)
                    if isinstance(f, FunctionType) and f == decorator:
                        return dec
                case ast.Name():
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
            case ast.Name():
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
