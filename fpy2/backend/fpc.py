"""Compilation from FPy to FPCore."""

import titanfp.fpbench.fpcast as fpc

from ..analysis import DefineUse, DefineUseAnalysis
from ..ast import *
from ..fpc_context import FPCoreContext
from ..function import Function
from ..number import Context
from ..transform import ConstFold, ForBundling, ForUnpack, FuncUpdate, IfBundling, WhileBundling
from ..utils import Gensym

from .backend import Backend, CompileError

# Cached table storage
_nullary_table_cache: dict[type[NullaryOp], fpc.Expr] | None = None
_unary_table_cache: dict[type[UnaryOp], type[fpc.Expr]] | None = None
_binary_table_cache: dict[type[BinaryOp], type[fpc.Expr]] | None = None
_ternary_table_cache: dict[type[TernaryOp], type[fpc.Expr]] | None = None
_nary_table_cache: dict[type[NaryOp], type[fpc.Expr]] | None = None

def _get_nullary_table() -> dict[type[NullaryOp], fpc.Expr]:
    """Get the cached nullary operations table."""
    global _nullary_table_cache
    if _nullary_table_cache is None:
        _nullary_table_cache = {
            ConstNan: fpc.Constant('NAN'),
            ConstInf: fpc.Constant('INFINITY'),
            ConstPi: fpc.Constant('PI'),
            ConstE: fpc.Constant('E'),
            ConstLog2E: fpc.Constant('LOG2E'),
            ConstLog10E: fpc.Constant('LOG10E'),
            ConstLn2: fpc.Constant('LN2'),
            ConstPi_2: fpc.Constant('PI_2'),
            ConstPi_4: fpc.Constant('PI_4'),
            Const1_Pi: fpc.Constant('M_1_PI'),
            Const2_Pi: fpc.Constant('M_2_PI'),
            Const2_SqrtPi: fpc.Constant('M_2_SQRTPI'),
            ConstSqrt2: fpc.Constant('SQRT2'),
            ConstSqrt1_2: fpc.Constant('SQRT1_2'),
        }
    return _nullary_table_cache

def _get_unary_table() -> dict[type[UnaryOp], type[fpc.Expr]]:
    """Get the cached unary operations table."""
    global _unary_table_cache
    if _unary_table_cache is None:
        _unary_table_cache = {
            Fabs: fpc.Fabs,
            Sqrt: fpc.Sqrt,
            Neg: fpc.Neg,
            Cbrt: fpc.Cbrt,
            Ceil: fpc.Ceil,
            Floor: fpc.Floor,
            NearbyInt: fpc.Nearbyint,
            RoundInt: fpc.Round,
            Trunc: fpc.Trunc,
            Acos: fpc.Acos,
            Asin: fpc.Asin,
            Atan: fpc.Atan,
            Cos: fpc.Cos,
            Sin: fpc.Sin,
            Tan: fpc.Tan,
            Acosh: fpc.Acosh,
            Asinh: fpc.Asinh,
            Atanh: fpc.Atanh,
            Cosh: fpc.Cosh,
            Sinh: fpc.Sinh,
            Tanh: fpc.Tanh,
            Exp: fpc.Exp,
            Exp2: fpc.Exp2,
            Expm1: fpc.Expm1,
            Log: fpc.Log,
            Log10: fpc.Log10,
            Log1p: fpc.Log1p,
            Log2: fpc.Log2,
            Erf: fpc.Erf,
            Erfc: fpc.Erfc,
            Lgamma: fpc.Lgamma,
            Tgamma: fpc.Tgamma,
            IsFinite: fpc.Isfinite,
            IsInf: fpc.Isinf,
            IsNan: fpc.Isnan,
            IsNormal: fpc.Isnormal,
            Signbit: fpc.Signbit,
            Not: fpc.Not,
        }
    return _unary_table_cache

def _get_binary_table() -> dict[type[BinaryOp], type[fpc.Expr]]:
    """Get the cached binary operations table."""
    global _binary_table_cache
    if _binary_table_cache is None:
        _binary_table_cache = {
            Add: fpc.Add,
            Sub: fpc.Sub,
            Mul: fpc.Mul,
            Div: fpc.Div,
            Copysign: fpc.Copysign,
            Fdim: fpc.Fdim,
            Fmod: fpc.Fmod,
            Remainder: fpc.Remainder,
            Hypot: fpc.Hypot,
            Atan2: fpc.Atan2,
            Pow: fpc.Pow,
        }
    return _binary_table_cache

def _get_ternary_table() -> dict[type[TernaryOp], type[fpc.Expr]]:
    """Get the cached ternary operations table."""
    global _ternary_table_cache
    if _ternary_table_cache is None:
        _ternary_table_cache = {
            Fma: fpc.Fma,
        }
    return _ternary_table_cache

def _get_nary_table() -> dict[type[NaryOp], type[fpc.Expr]]:
    """Get the cached n-ary operations table."""
    global _nary_table_cache
    if _nary_table_cache is None:
        _nary_table_cache = {
            Or: fpc.Or,
            And: fpc.And,
        }
    return _nary_table_cache

class FPCoreCompileError(CompileError):
    """Any FPCore compilation error"""
    pass

def _nary_mul(args: list[fpc.Expr]):
    assert args != [], 'must be at least 1 argument'
    if len(args) == 1:
        return args[0]
    else:
        e = fpc.Mul(args[0], args[1])
        for arg in args[2:]:
            e = fpc.Mul(e, arg)
        return e

def _size0_expr(x: str):
    return fpc.Size(fpc.Var(x), fpc.Integer(0))


class _FPCoreCompileInstance(Visitor):
    """Compilation instance from FPy to FPCore"""

    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    unsafe_int_cast: bool

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, unsafe_int_cast: bool = True):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=def_use.names())
        self.unsafe_int_cast = unsafe_int_cast

    def compile(self) -> fpc.FPCore:
        f = self._visit_function(self.func, None)
        assert isinstance(f, fpc.FPCore), 'unexpected result type'
        return f

    def _compile_arg(self, arg: Argument) -> tuple[str, dict, list[int | str] | None]:
        match arg.type:
            case AnyTypeAnn() | None:
                return str(arg.name), {}, None
            case RealTypeAnn():
                return str(arg.name), {}, None
            case SizedTensorTypeAnn():
                dims: list[int | str] = []
                for dim in arg.type.dims:
                    if isinstance(dim, int):
                        dims.append(dim)
                    elif isinstance(dim, NamedId):
                        dims.append(str(dim))
                    else:
                        raise FPCoreCompileError('unexpected dimension type', dim)
                return str(arg.name), {}, dims
            case _:
                raise FPCoreCompileError('unsupported argument type', arg)

    def _compile_tuple_binding(self, tuple_id: str, binding: TupleBinding, pos: list[fpc.Expr]):
        tuple_binds: list[tuple[str, fpc.Expr]] = []
        for i, elt in enumerate(binding):
            match elt:
                case Id():
                    idxs = [fpc.Integer(i), *pos]
                    tuple_bind = (str(elt), fpc.Ref(fpc.Var(tuple_id), *idxs))
                    tuple_binds.append(tuple_bind)
                case TupleBinding():
                    idxs = [fpc.Integer(i), *pos]
                    tuple_binds += self._compile_tuple_binding(tuple_id, elt, idxs)
                case _:
                    raise FPCoreCompileError('unexpected tensor element', elt)
        return tuple_binds

    def _compile_compareop(self, op: CompareOp):
        match op:
            case CompareOp.LT:
                return fpc.LT
            case CompareOp.LE:
                return fpc.LEQ
            case CompareOp.GE:
                return fpc.GEQ
            case CompareOp.GT:
                return fpc.GT
            case CompareOp.EQ:
                return fpc.EQ
            case CompareOp.NE:
                return fpc.NEQ
            case _:
                raise NotImplementedError('unreachable', op)

    def _visit_var(self, e: Var, ctx: None) -> fpc.Expr:
        return fpc.Var(str(e.name))

    def _visit_bool(self, e: BoolVal, ctx: None):
        return fpc.Constant('TRUE' if e.val else 'FALSE')

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> fpc.Expr:
        raise FPCoreCompileError('unsupported value', e.val)

    def _visit_decnum(self, e: Decnum, ctx: None) -> fpc.Expr:
        if e.is_integer() and self.unsafe_int_cast:
            # unsafe integer cast: compile under integer context
            v = int(e.as_rational())
            return fpc.Ctx({ 'precision': 'integer' }, fpc.Integer(v))
        raise FPCoreCompileError('cannot compile unrounded constant', e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: None):
        if e.is_integer() and self.unsafe_int_cast:
            # unsafe integer cast: compile under integer context
            v = int(e.as_rational())
            return fpc.Ctx({ 'precision': 'integer' }, fpc.Integer(v))
        raise FPCoreCompileError('cannot compile unrounded constant', e.val)

    def _visit_integer(self, e: Integer, ctx: None) -> fpc.Expr:
        if self.unsafe_int_cast:
            # unsafe integer cast: compile under integer context
            return fpc.Ctx({ 'precision': 'integer' }, fpc.Integer(e.val))
        else:
            raise FPCoreCompileError('cannot compile unrounded constant', e.val)

    def _visit_rational(self, e: Rational, ctx: None):
        if self.unsafe_int_cast and e.is_integer():
            # unsafe integer cast: compile under integer context
            v = int(e.as_rational())
            return fpc.Ctx({ 'precision': 'integer' }, fpc.Integer(v))
        raise FPCoreCompileError('cannot compile unrounded constant', f'{e.p}/{e.q}')

    def _visit_digits(self, e: Digits, ctx: None) -> fpc.Expr:
        if self.unsafe_int_cast and e.e == 0 and e.b == 2:
            # unsafe integer cast: compile under integer context
            v = int(e.as_rational())
            return fpc.Ctx({ 'precision': 'integer' }, fpc.Integer(v))
        raise FPCoreCompileError('cannot compile unrounded constant', f'digits({e.m}, {e.e}, {e.b})')

    def _visit_call(self, e: Call, ctx: None) -> fpc.Expr:
        if e.kwargs:
            raise FPCoreCompileError('cannot compile keyword arguments to FPCore', e)
        match e.func:
            case Var():
                name = str(e.func.name)
            case Attribute():
                raise FPCoreCompileError('cannot compile method call to FPCore', e.func)
            case _:
                raise RuntimeError('unreachable', e.func)
        args = [self._visit_expr(c, ctx) for c in e.args]
        return fpc.UnknownOperator(*args, name=name)

    def _visit_round(self, e: Round | RoundExact, ctx: None) -> fpc.Expr:
        # round expression
        match e.arg:
            case Decnum():
                # round(n) => n
                return fpc.Decnum(e.arg.val)
            case Hexnum():
                # round(n) => n
                return fpc.Hexnum(e.arg.val)
            case Integer():
                # round(n) => n
                return fpc.Integer(e.arg.val)
            case Rational():
                # round(p/q) => p/q
                return fpc.Rational(e.arg.p, e.arg.q)
            case Digits():
                # round(digits(m, e, b)) => digits(m, e, b)
                return fpc.Digits(e.arg.m, e.arg.e, e.arg.b)
            case _:
                # round(e) => cast(e)
                arg = self._visit_expr(e.arg, ctx)
                return fpc.Cast(arg)

    def _visit_round_at(self, e: RoundAt, ctx: None) -> fpc.Expr:
        raise FPCoreCompileError('cannot compile `round_at` expression to FPCore', e)

    def _visit_len(self, arg: Expr, ctx: None) -> fpc.Expr:
        # length expression
        arr = self._visit_expr(arg, ctx)
        return fpc.Size(arr, fpc.Integer(0))

    def _visit_range(self, arg: Expr, ctx: None) -> fpc.Expr:
        # expand range expression
        tuple_id = str(self.gensym.fresh('i'))
        size = self._visit_expr(arg, ctx)
        return fpc.Tensor([(tuple_id, size)], fpc.Var(tuple_id))

    def _visit_empty(self, arg: Expr, ctx: None) -> fpc.Expr:
        # tensor with uninitialized values
        tuple_id = str(self.gensym.fresh('i'))
        size = self._visit_expr(arg, ctx)
        return fpc.Tensor([(tuple_id, size)], fpc.Integer(0))

    def _visit_size(self, arr: Expr, dim: Expr, ctx) -> fpc.Expr:
        tup = self._visit_expr(arr, ctx)
        idx = self._visit_expr(dim, ctx)
        return fpc.Size(tup, idx)

    def _visit_dim(self, arr: Expr, ctx) -> fpc.Expr:
        tup = self._visit_expr(arr, ctx)
        return fpc.Dim(tup)

    def _visit_enumerate(self, arr: Expr, ctx: None) -> fpc.Expr:
        # (let ([t <tuple>])
        #  (tensor ([i (size t 0)])
        #    (array i (ref t i))))
        tuple_id = str(self.gensym.fresh('i'))
        iter_id = str(self.gensym.fresh('i'))
        tup = self._visit_expr(arr, ctx)
        return fpc.Let(
            [(tuple_id, tup)],
            fpc.Tensor(
                [(iter_id, _size0_expr(tuple_id))],
                fpc.Array(fpc.Var(iter_id), fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))
            )
        )

    def _visit_zip(self, args: tuple[Expr, ...], ctx: None) -> fpc.Expr:
        # expand zip expression (for N=2)
        #  (let ([t0 <tuple0>] [t1 <tuple1>])
        #    (tensor ([i (size t0 0)])
        #      (array (ref t0 i) (ref t1 i)))))
        if len(args) == 0:
            # no children => empty zip
            return fpc.Array()
        else:
            tuples = [self._visit_expr(t, ctx) for t in args]
            tuple_ids = [str(self.gensym.fresh('t')) for _ in args]
            iter_id = str(self.gensym.fresh('i'))
            return fpc.Let(
                list(zip(tuple_ids, tuples)),
                fpc.Tensor([(iter_id, fpc.Size(fpc.Var(tuple_ids[0]), fpc.Integer(0)))],
                    fpc.Array(*[fpc.Ref(fpc.Var(tid), fpc.Var(iter_id)) for tid in tuple_ids])
                )
            )

    def _visit_min(self, args: tuple[Expr, ...], ctx: None) -> fpc.Expr:
        # expand min expression by left associativity
        # at least two arguments
        # (fmin (fmin (fmin t1 t2) t3) ... tn)
        vals = [self._visit_expr(arg, ctx) for arg in args]
        min_expr = vals[0]
        for val in vals[1:]:
            min_expr = fpc.Fmin(min_expr, val)
        return min_expr

    def _visit_max(self, args: tuple[Expr, ...], ctx: None) -> fpc.Expr:
        # expand max expression by left associativity
        # at least two arguments
        # (fmax (fmax (fmax t1 t2) t3) ... tn)
        vals = [self._visit_expr(arg, ctx) for arg in args]
        min_expr = vals[0]
        for val in vals[1:]:
            min_expr = fpc.Fmax(min_expr, val)
        return min_expr

    def _visit_sum(self, arg: Expr, ctx: None) -> fpc.Expr:
        # expand sum expression by left associativity
        # the sum expression has at most one argument
        # (let ([t <tuple>])
        #   (for ([i (! :precision integer (- (size t 0) 1))]
        #         [accum (ref t i) (+ accum (ref t (! :precision integer (+ i 1))))])
        #     accum))
        tuple_id = str(self.gensym.fresh('t'))
        iter_id = str(self.gensym.fresh('i'))
        accum_id = str(self.gensym.fresh('accum'))
        idx_ctx = { 'precision': 'integer' }

        tup = self._visit_expr(arg, ctx)
        return fpc.Let(
            [(tuple_id, tup)],
            fpc.For(
                [(iter_id, fpc.Ctx(idx_ctx, fpc.Sub(fpc.Size(fpc.Var(tuple_id), fpc.Integer(0)), fpc.Integer(1))))],
                [(
                    accum_id,
                    fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)),
                    fpc.Ctx(idx_ctx, fpc.Add(accum_id, fpc.Ref(fpc.Var(tuple_id), fpc.Add(fpc.Var(iter_id), fpc.Integer(1)))))
                )],
                fpc.Var(accum_id)
            )
        )


    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> fpc.Expr:
        nullary_table = _get_nullary_table()
        if type(e) not in nullary_table:
            # unknown operator
            raise NotImplementedError('no FPCore operator for', e)
        return nullary_table[type(e)]

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> fpc.Expr:
        unary_table = _get_unary_table()
        cls = unary_table.get(type(e))
        if cls is not None:
            # known unary operator
            arg = self._visit_expr(e.arg, ctx)
            return cls(arg)
        else:
            match e:
                case Len():
                    # len expression
                    return self._visit_len(e.arg, ctx)
                case Range():
                    # range expression
                    return self._visit_range(e.arg, ctx)
                case Empty():
                    # empty expression
                    return self._visit_empty(e.arg, ctx)
                case Dim():
                    # dim expression
                    return self._visit_dim(e.arg, ctx)
                case Enumerate():
                    # enumerate expression
                    return self._visit_enumerate(e.arg, ctx)
                case Sum():
                    # sum expression
                    return self._visit_sum(e.arg, ctx)
                case _:
                    raise NotImplementedError('no FPCore operator for', e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> fpc.Expr:
        binary_table = _get_binary_table()
        cls = binary_table.get(type(e))
        if cls is not None:
            # known binary operator
            arg0 = self._visit_expr(e.first, ctx)
            arg1 = self._visit_expr(e.second, ctx)
            return cls(arg0, arg1)
        else:
            match e:
                case Size():
                    # size expression
                    return self._visit_size(e.first, e.second, ctx)
                case _:
                    # unknown operator
                    raise NotImplementedError('no FPCore operator for', e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> fpc.Expr:
        ternary_table = _get_ternary_table()
        cls = ternary_table.get(type(e))
        if cls is not None:
            # known ternary operator
            arg0 = self._visit_expr(e.first, ctx)
            arg1 = self._visit_expr(e.second, ctx)
            arg2 = self._visit_expr(e.third, ctx)
            return cls(arg0, arg1, arg2)
        else:
            # unknown operator
            raise NotImplementedError('no FPCore operator for', e)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> fpc.Expr:
        nary_table = _get_nary_table()
        cls = nary_table.get(type(e))
        if cls is not None:
            # known n-ary operator
            return cls(*[self._visit_expr(c, ctx) for c in e.args])
        else:
            match e:
                case Zip():
                    # zip expression
                    return self._visit_zip(e.args, ctx)
                case Min():
                    # min expression
                    return self._visit_min(e.args, ctx)
                case Max():
                    # max expression
                    return self._visit_max(e.args, ctx)
                case _:
                    # unknown operator
                    raise NotImplementedError('no FPCore operator for', e)

    def _visit_compare(self, e: Compare, ctx: None) -> fpc.Expr:
        assert e.ops != [], 'should not be empty'
        match e.ops:
            case [op]:
                # 2-argument case: just compile
                cls = self._compile_compareop(op)
                arg0 = self._visit_expr(e.args[0], ctx)
                arg1 = self._visit_expr(e.args[1], ctx)
                return cls(arg0, arg1)
            case [op, *ops]:
                # N-argument case:
                # TODO: want to evaluate each argument only once;
                #       may need to let-bind in case any argument is
                #       used multiple times
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                curr_group = (op, [args[0], args[1]])
                groups: list[tuple[CompareOp, list[fpc.Expr]]] = [curr_group]
                for op, lhs, rhs in zip(ops, args[1:], args[2:]):
                    if op == curr_group[0] or isinstance(lhs, fpc.ValueExpr):
                        # same op => append
                        # different op (terminal) => append
                        curr_group[1].append(lhs)
                    else:
                        # different op (non-terminal) => new group
                        new_group = (op, [lhs, rhs])
                        groups.append(new_group)
                        curr_group = new_group

                if len(groups) == 1:
                    op, args = groups[0]
                    cls = self._compile_compareop(op)
                    return cls(*args)
                else:
                    args = [self._compile_compareop(op)(*args) for op, args in groups]
                    return fpc.And(*args)
            case _:
                raise NotImplementedError('unreachable', e.ops)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> fpc.Expr:
        return fpc.Array(*[self._visit_expr(c, ctx) for c in e.elts])

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> fpc.Expr:
        return fpc.Array(*[self._visit_expr(c, ctx) for c in e.elts])

    def _visit_list_ref(self, e: ListRef, ctx: None) -> fpc.Expr:
        t: Expr = e
        indices: list[fpc.Expr] = []
        while isinstance(t, ListRef):
            index = self._visit_expr(t.index, ctx)
            indices.append(index)
            t = t.value

        value = self._visit_expr(t, ctx)
        return fpc.Ref(value, *reversed(indices))

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> fpc.Expr:
        value = self._visit_expr(e.value, ctx)
        match e.start, e.stop:
            case None, None:
                # produce a copy
                # (let ([t <value>])
                #   (tensor ([i (size t 0)]) (ref t i)))
                tuple_id = str(self.gensym.fresh('t'))
                iter_id = str(self.gensym.fresh('i'))
                return fpc.Let(
                    [(tuple_id, value)],
                    fpc.Tensor([(iter_id, _size0_expr(tuple_id))],
                        fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))
                )
            case None, _:
                # produce a truncated copy
                # (let ([t <value>])
                #   (tensor ([i <stop>]) (ref t i)))
                assert isinstance(e.stop, Expr) # mypy doesn't like this match statement
                tuple_id = str(self.gensym.fresh('t'))
                iter_id = str(self.gensym.fresh('i'))
                stop = self._visit_expr(e.stop, ctx)
                return fpc.Let(
                    [(tuple_id, value)],
                    fpc.Tensor([(iter_id, stop)],
                        fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))
                )
            case _:
                # default case
                # (let ([t <value>]
                #   (let ([len (size t 0)])
                #     (let ([start (max 0 (min start len))]
                #           [stop (max 0 (min stop len))])
                #       (tensor ([i (fdim end start)]) (ref x (+ i start)))))
                tuple_id = str(self.gensym.fresh('t'))
                len_id = str(self.gensym.fresh('len'))
                start_id = str(self.gensym.fresh('start'))
                stop_id = str(self.gensym.fresh('stop'))
                iter_id = str(self.gensym.fresh('i'))

                assert e.start is not None
                start = self._visit_expr(e.start, ctx)
                if e.stop is None:
                    stop = fpc.Size(fpc.Var(tuple_id), fpc.Integer(0))
                else:
                    stop = self._visit_expr(e.stop, ctx)

                return fpc.Let(
                    [(tuple_id, value)],
                    fpc.Let(
                        [(len_id, _size0_expr(tuple_id))],
                        fpc.Let(
                            [
                                (start_id, fpc.Fmax(fpc.Integer(0), fpc.Fmin(start, fpc.Var(len_id)))),
                                (stop_id, fpc.Fmax(fpc.Integer(0), fpc.Fmin(stop, fpc.Var(len_id))))
                            ],
                            fpc.Tensor(
                                [(iter_id, fpc.Fdim(fpc.Var(stop_id), fpc.Var(start_id)))],
                                fpc.Ref(fpc.Var(tuple_id), fpc.Add(fpc.Var(iter_id), fpc.Var(start_id)))
                            )
                        )
                    )
                )

    def _generate_tuple_set(self, tuple_id: str, iter_id: str, idx_ids: list[str], val_id: str):
        # dimension bindings
        idx_id = idx_ids[0]
        tensor_dims = [(iter_id, _size0_expr(tuple_id))]
        # generate if expression
        cond_expr = fpc.EQ(fpc.Var(iter_id), fpc.Var(idx_id))
        iff_expr = fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id))
        if len(idx_ids) == 1:
            ift_expr = fpc.Var(val_id)
        else:
            let_bindings = [(tuple_id, fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))]
            rec_expr = self._generate_tuple_set(tuple_id, iter_id, idx_ids[1:], val_id)
            ift_expr = fpc.Let(let_bindings, rec_expr)
        if_expr = fpc.If(cond_expr, ift_expr, iff_expr)
        return fpc.Tensor(tensor_dims, if_expr)

    def _visit_list_set(self, e: ListSet, ctx: None) -> fpc.Expr:
        # general case:
        # 
        #   (let ([t <tuple>] [i0 <index>] ... [v <value>]))
        #     (tensor ([k (size t 0)])
        #       (if (= k i)
        #           (let ([t (ref t i0)])
        #             <recurse with i1, ...>)
        #           (ref t i0)
        #
        # where <recurse with i1, ...> is
        #
        #   (tensor ([k (size t 0)])
        #     (if (= k i1)
        #         (let ([t (ref t i1)])
        #           <recurse with i2, ...>)
        #         (ref t i1)
        #
        # and <recurse with iN> is
        #
        #   (tensor ([k (size t 0)])
        #     (if (= k iN) v (ref t iN))
        #

        # generate temporary variables
        tuple_id = str(self.gensym.fresh('t'))
        idx_ids = [str(self.gensym.fresh('i')) for _ in e.indices]
        iter_id = str(self.gensym.fresh('k'))
        val_id = str(self.gensym.fresh('v'))

        # compile each component
        tuple_expr = self._visit_expr(e.value, ctx)
        idx_exprs = [self._visit_expr(idx, ctx) for idx in e.indices]
        val_expr = self._visit_expr(e.expr, ctx)

        # create initial let binding
        let_bindings = [(tuple_id, tuple_expr)]
        for idx_id, idx_expr in zip(idx_ids, idx_exprs):
            let_bindings.append((idx_id, idx_expr))
        let_bindings.append((val_id, val_expr))

        # recursively generate tensor expressions
        tensor_expr = self._generate_tuple_set(tuple_id, iter_id, idx_ids, val_id)
        return fpc.Let(let_bindings, tensor_expr)


    def _visit_list_comp(self, e: ListComp, ctx: None) -> fpc.Expr:
        if len(e.targets) == 1:
            # simple case:
            # (let ([t <iterable>]) (tensor ([i (size t 0)]) (let ([<var> (ref t i)]) <elt>))
            target = e.targets[0]
            iterable = e.iterables[0]

            tuple_id = str(self.gensym.fresh('t'))
            iter_id = str(self.gensym.fresh('i'))
            iterable = self._visit_expr(iterable, ctx)
            elt = self._visit_expr(e.elt, ctx)

            let_bindings = [(tuple_id, iterable)]
            tensor_dims: list[tuple[str, fpc.Expr]] = [(iter_id, _size0_expr(tuple_id))]
            match target:
                case NamedId():
                    ref_bindings = [(str(target), fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))]
                case UnderscoreId():
                    ref_bindings = []
                case TupleBinding():
                    ref_bindings = self._compile_tuple_binding(tuple_id, target, [fpc.Var(iter_id)])
                case _:
                    raise RuntimeError('unreachable', target)
            return fpc.Let(let_bindings, fpc.Tensor(tensor_dims, fpc.LetStar(ref_bindings, elt)))
        else:
            # hard case:
            # (let ([t0 <iterable>] ...)
            #   (let ([n0 (size t0 0)] ...)
            #     (tensor ([k (! :precision integer (* n0 ...))])
            #       (let ([i0 (! :precision integer :round toZero (/ k (* n1 ...)))]
            #             [i1 (! :precision integer :round toZero (fmod (/ k (* n2 ...)) n1))]
            #             ...
            #             [iN (! :precision integer :round toZero (fmod k nN))])
            #         (let ([v0 (ref t0 i0)] ...)
            #           <elt>))))

            # bind the tuples to temporaries
            tuple_ids = [str(self.gensym.fresh('t')) for _ in e.targets]
            tuple_binds: list[tuple[str, fpc.Expr]] = [
                (tid, self._visit_expr(iterable, ctx))
                for tid, iterable in zip(tuple_ids, e.iterables)
            ]
            # bind the sizes to temporaries
            size_ids = [str(self.gensym.fresh('n')) for _ in e.targets]
            size_binds: list[tuple[str, fpc.Expr]] = [
                (sid, _size0_expr(tid))
                for sid, tid in zip(size_ids, tuple_ids)
            ]
            # bind the indices to temporaries
            idx_ctx = { 'precision': 'integer', 'round': 'toZero' }
            idx_ids = [str(self.gensym.fresh('i')) for _ in e.targets]
            idx_binds: list[tuple[str, fpc.Expr]] = []
            for i, iid in enumerate(idx_ids):
                if i == 0:
                    mul_expr = _nary_mul([fpc.Var(id) for id in size_ids[1:]])
                    idx_expr = fpc.Ctx(idx_ctx, fpc.Div(fpc.Var('k'), mul_expr))
                elif i == len(size_ids) - 1:
                    idx_expr = fpc.Ctx(idx_ctx, fpc.Fmod(fpc.Var('k'), fpc.Var(size_ids[i])))
                else:
                    mul_expr = _nary_mul([fpc.Var(id) for id in size_ids[1:]])
                    idx_expr = fpc.Ctx(idx_ctx, fpc.Fmod(fpc.Div(fpc.Var('k'), mul_expr), fpc.Var(size_ids[i])))
                idx_binds.append((iid, idx_expr))
            # iteration variable
            iter_ctx = { 'precision': 'integer'}
            iter_id = str(self.gensym.fresh('k'))
            iter_expr = fpc.Ctx(iter_ctx, _nary_mul([fpc.Var(sid) for sid in size_ids]))
            # reference variables
            ref_binds: list[tuple[str, fpc.Expr]] = []
            for target, tid, iid in zip(e.targets, tuple_ids, idx_ids):
                match target:
                    case NamedId():
                        ref_id = str(self.gensym.refresh(target))
                        ref_bind = (ref_id, fpc.Ref(fpc.Var(tid), fpc.Var(iid)))
                        ref_binds.append(ref_bind)
                    case TupleBinding():
                        ref_binds += self._compile_tuple_binding(tid, target, [fpc.Var(iid)])
            # element expression
            elt = self._visit_expr(e.elt, ctx)
            # compose the expression
            tensor_expr = fpc.Tensor([(iter_id, iter_expr)], fpc.LetStar(idx_binds, fpc.LetStar(ref_binds, elt)))
            return fpc.Let(tuple_binds, fpc.Let(size_binds, tensor_expr))

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> fpc.Expr:
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return fpc.If(cond, ift, iff)

    def _visit_attribute(self, e: Attribute, ctx: None) -> fpc.Expr:
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(e).__name__}')

    def _visit_assign(self, stmt: Assign, ctx: fpc.Expr):
        match stmt.target:
            case Id():
                bindings = [(str(stmt.target), self._visit_expr(stmt.expr, None))]
                return fpc.Let(bindings, ctx)
            case TupleBinding():
                tuple_id = str(self.gensym.fresh('t'))
                tuple_bind = (tuple_id, self._visit_expr(stmt.expr, None))
                destruct_bindings = self._compile_tuple_binding(tuple_id, stmt.target, [])
                return fpc.LetStar([tuple_bind] + destruct_bindings, ctx)
            case _:
                raise RuntimeError('unreachable', stmt.binding)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: fpc.Expr):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_if1(self, stmt: If1Stmt, ret: fpc.Expr):
        # check that only one variable is mutated in the loop
        # the `IfBundling` pass is required to ensure this
        mutated = self.def_use.mutated_in(stmt.body)
        num_mutated = len(mutated)

        if num_mutated == 0:
            # no mutated variables (if block with no side effect)
            # still want to return a valid FPCore
            # (let ([_ (if <cond> (begin <body> 0) 0)]) <ret>)
            cond = self._visit_expr(stmt.cond, None)
            body = self._visit_block(stmt.body, fpc.Integer(0))
            # return the if expression
            return fpc.Let([('_', fpc.If(cond, body, fpc.Integer(0)))], ret)
        elif num_mutated == 1:
            # exactly one mutated variable
            # the mutated variable is the loop variable
            # (let ([<mut> (if <cond> (begin <body> <mut>) <mut>)]) <ret>)
            mut_id = str(mutated.pop())
            cond = self._visit_expr(stmt.cond, None)
            body = self._visit_block(stmt.body, fpc.Var(mut_id))
            # return the if expression
            return fpc.Let([(mut_id, fpc.If(cond, body, fpc.Var(mut_id)))], ret)
        else:
            # more than one mutated variable
            # cannot compile to FPCore
            raise FPCoreCompileError(f'if statements cannot have more than 1 mutated variable: {list(mutated)}')

    def _visit_if(self, stmt: IfStmt, ret: fpc.Expr):
        # check that only one variable is mutated in the loop
        # the `IfBundling` pass is required to ensure this
        mutated_ift = self.def_use.mutated_in(stmt.ift)
        mutated_iff = self.def_use.mutated_in(stmt.iff)
        mutated = sorted(mutated_ift | mutated_iff)

        # identify variables that were introduced in each body
        intros_ift = self.def_use.introed_in(stmt.ift)
        intros_iff = self.def_use.introed_in(stmt.iff)
        intros = sorted(intros_ift & intros_iff) # intersection of fresh variables

        # mutated or introduced variables
        changed = mutated + intros
        num_changed = len(changed)

        if num_changed == 0:
            # no variables mutated or introduced (block has no side effects)
            # still want to return a valid FPCore
            # (let ([_ (if <cond> (begin <ift> 0) (begin <iff> 0))]) <ret>)
            cond = self._visit_expr(stmt.cond, None)
            ift = self._visit_block(stmt.ift, fpc.Integer(0))
            iff = self._visit_block(stmt.iff, fpc.Integer(0))
            # return the if expression
            return fpc.Let([('_', fpc.If(cond, ift, iff))], ret)
        elif num_changed == 1:
            # exactly one variable mutated or introduced
            # the mutated variable is the loop variable
            # (let ([<mut> (if <cond> (begin <ift> <mut>) (begin <iff> <mut>))]) <ret>)
            mut_id = str(changed[0])
            cond = self._visit_expr(stmt.cond, None)
            ift = self._visit_block(stmt.ift, fpc.Var(mut_id))
            iff = self._visit_block(stmt.iff, fpc.Var(mut_id))
            # return the if expression
            return fpc.Let([(mut_id, fpc.If(cond, ift, iff))], ret)
        else:
            # more than one mutated or introduced variable
            # cannot compile to FPCore
            raise FPCoreCompileError(f'if statements cannot have more than 1 mutated or introduced variable: {list(changed)}')

    def _visit_while(self, stmt: WhileStmt, ret: fpc.Expr):
        # check that only one variable is mutated in the loop
        # the `WhileBundling` pass is required to ensure this
        mutated = self.def_use.mutated_in(stmt.body)
        num_mutated = len(mutated)

        if num_mutated == 0:
            # no mutated variables (loop with no side effect)
            # still want to return a valid FPCore
            # (while ([_ 0 (let ([_ <body>]) 0)]) <ret>)
            cond = self._visit_expr(stmt.cond, None)
            body = self._visit_block(stmt.body, fpc.Integer(0))
            return fpc.While(cond, [('_', fpc.Integer(0), body)], ret)
        elif num_mutated == 1:
            # exactly one mutated variable
            # the mutated variable is the loop variable
            # (while ([<loop> <loop> <body>]) <ret>)
            loop_id = str(mutated.pop())
            cond = self._visit_expr(stmt.cond, None)
            body = self._visit_block(stmt.body, fpc.Var(loop_id))
            return fpc.While(cond, [(loop_id, fpc.Var(loop_id), body)], ret)
        else:
            raise FPCoreCompileError(f'while loops cannot have more than 1 mutated variable: {list(mutated)}')


    def _visit_for(self, stmt: ForStmt, ret: fpc.Expr):
        # check that only one variable is mutated in the loop
        # the `ForBundling` pass is required to ensure this
        mutated = self.def_use.mutated_in(stmt.body)
        num_mutated = len(mutated)

        if not isinstance(stmt.target, Id):
            raise FPCoreCompileError(f'for loops must have a single target: {stmt.target} ')
        idx_id = str(stmt.target)

        if num_mutated == 0:
            # no mutated variables (loop with no side effect)
            # still want to return a valid FPCore
            # (let ([<t> <iterable>])
            #   (for ([<i> (size <t> 0)])
            #        ([<i> 0 (! :precision integer :round toZero (+ i 1)_])
            #         [_ 0 (let ([_ <body>]) 0)])))
            #        <ret>))
            tuple_id = str(self.gensym.fresh('t'))
            iterable = self._visit_expr(stmt.iterable, None)
            body = self._visit_block(stmt.body, fpc.Integer(0))
            return fpc.Let(
                [(tuple_id, iterable)],
                fpc.For(
                    [(idx_id, _size0_expr(tuple_id))],
                    [('_', fpc.Integer(0), body)],
                    ret
            ))
        else:
            # exactly one mutated variable
            # the mutated variable is the loop variable
            # (let ([<t> <iterable>])
            #   (for ([<i> (size <t> 0)])
            #         [<loop> <loop> (let ([_ <body>]) <loop>)])))
            #        <ret>))
            loop_id = str(mutated.pop())
            tuple_id = str(self.gensym.fresh('t'))
            iterable = self._visit_expr(stmt.iterable, None)
            body = self._visit_block(stmt.body, fpc.Var(loop_id))
            return fpc.Let(
                [(tuple_id, iterable)],
                fpc.For(
                    [(idx_id, _size0_expr(tuple_id))],
                    [(loop_id, fpc.Var(loop_id), body)],
                    ret
            ))

    def _visit_data(self, data):
        match data:
            case int():
                return fpc.Integer(data)
            case str():
                return fpc.Var(data)
            case tuple() | list():
                return tuple(self._visit_data(d) for d in data)
            case Expr():
                return self._visit_expr(data, None)
            case _:
                raise NotImplementedError(repr(data))

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # check if the context is bound
        if isinstance(stmt.target, NamedId):
            raise FPCoreCompileError('Context statements cannot bind to a variable', stmt.target)

        body = self._visit_block(stmt.body, ctx)
        # extract a context value
        match stmt.ctx:
            case ForeignVal():
                val = stmt.ctx.val
            case _:
                raise FPCoreCompileError('Context expressions must be pre-computed', stmt.ctx)

        # convert to properties
        match val:
            case Context():
                props = FPCoreContext.from_context(val).props
            case FPCoreContext():
                props = val.props
            case _:
                raise FPCoreCompileError('Expected `Context` or `FPCoreContext`', val)

        # transform properties
        for k in props:
            props[k] = fpc.Data(self._visit_data(props[k]))
        return fpc.Ctx(props, body)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        # strip the assertion
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: fpc.Expr):
        raise FPCoreCompileError('FPCore does not support effectful computation')

    def _visit_return(self, stmt: ReturnStmt, ctx: None) -> fpc.Expr:
        return self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: None) -> fpc.Expr:
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: fpc.Expr | None):
        if ctx is None:
            e = self._visit_statement(block.stmts[-1], None)
            stmts = block.stmts[:-1]
        else:
            e = ctx
            stmts = block.stmts

        for stmt in reversed(stmts):
            if isinstance(stmt, ReturnStmt):
                raise FPCoreCompileError('return statements must be at the end of blocks')
            e = self._visit_statement(stmt, e)

        return e

    def _visit_function(self, func: FuncDef, ctx: fpc.Expr | None):
        args = [self._compile_arg(arg) for arg in func.args]
        body = self._visit_block(func.body, ctx)

        # metadata
        if func.meta is None:
            props = {}
        else:
            props = func.meta.copy()

        # context properties
        if func.ctx is not None:
            match func.ctx:
                case Context():
                    fpc_ctx = FPCoreContext.from_context(func.ctx)
                case FPCoreContext():
                    fpc_ctx = func.ctx
                case _:
                    raise RuntimeError('unreachable', func.ctx)
            props.update(fpc_ctx.props)

        # function identifier
        ident = func.name

        # transform properties
        props = { k: fpc.Data(self._visit_data(v)) for k, v in props.items() }

        # special properties
        name = props.get('name')
        pre = props.get('pre')
        spec = props.get('spec')

        return fpc.FPCore(
            inputs=args,
            e=body,
            props=props,
            ident=ident,
            name=name,
            pre=pre,
            spec=spec
        )

    # override to get typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> fpc.Expr:
        return super()._visit_expr(e, ctx)

    # override to get typing hint
    def _visit_statement(self, stmt: Stmt, ctx: fpc.Expr) -> fpc.Expr:
        return super()._visit_statement(stmt, ctx)

class FPCoreCompiler(Backend):
    """Compiler from FPy to FPCore"""

    unsafe_int_cast: bool
    """any unrounded integer is automatically compiled under an integer context"""

    def __init__(self, *, unsafe_int_cast: bool = False):
        self.unsafe_int_cast = unsafe_int_cast

    def compile(self, func: Function, ctx: Context | None = None) -> fpc.FPCore:
        # TODO: handle ctx

        # normalization passes
        ast = ConstFold.apply(func.ast, enable_op=False)
        ast = FuncUpdate.apply(ast)
        ast = ForUnpack.apply(ast)
        ast = ForBundling.apply(ast)
        ast = WhileBundling.apply(ast)
        ast = IfBundling.apply(ast)
        # compile
        def_use = DefineUse.analyze(ast)
        return _FPCoreCompileInstance(ast, def_use, self.unsafe_int_cast).compile()
