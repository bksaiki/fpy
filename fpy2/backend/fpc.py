"""Compilation from FPy IR to FPCore"""

from typing import Optional

import titanfp.fpbench.fpcast as fpc

from ..analysis import DefineUse, DefineUseAnalysis
from ..ast import *
from ..fpc_context import FPCoreContext
from ..function import Function
from ..number import Context
from ..transform import ContextInline, ForBundling, ForUnpack, FuncUpdate, IfBundling, WhileBundling
from ..utils import Gensym

from .backend import Backend

_unary_table: dict[type[UnaryOp], type[fpc.Expr]] = {
    Fabs: fpc.Fabs,
    Sqrt: fpc.Sqrt,
    Neg: fpc.Neg,
    Cbrt: fpc.Cbrt,
    Ceil: fpc.Ceil,
    Floor: fpc.Floor,
    NearbyInt: fpc.Nearbyint,
    Round: fpc.Round,
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
    # rounding
    Cast: fpc.Cast,
}

_binary_table: dict[type[BinaryOp], type[fpc.Expr]] = {
    Add: fpc.Add,
    Sub: fpc.Sub,
    Mul: fpc.Mul,
    Div: fpc.Div,
    Copysign: fpc.Copysign,
    Fdim: fpc.Fdim,
    Fmax: fpc.Fmax,
    Fmin: fpc.Fmin,
    Fmod: fpc.Fmod,
    Remainder: fpc.Remainder,
    Hypot: fpc.Hypot,
    Atan2: fpc.Atan2,
    Pow: fpc.Pow,
}

_ternary_table: dict[type[TernaryOp], type[fpc.Expr]] = {
    Fma: fpc.Fma,
}

_nary_table: dict[type[NaryOp], type[fpc.Expr]] = {
    Or: fpc.Or,
    And: fpc.And,
}

class FPCoreCompileError(Exception):
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


class FPCoreCompileInstance(Visitor):
    """Compilation instance from FPy to FPCore"""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.gensym = Gensym(reserved=set(def_use.defs.keys()))

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
        return fpc.Decnum(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: None):
        return fpc.Hexnum(e.val)

    def _visit_integer(self, e: Integer, ctx: None) -> fpc.Expr:
        return fpc.Integer(e.val)

    def _visit_rational(self, e: Rational, ctx: None):
        return fpc.Rational(e.p, e.q)

    def _visit_constant(self, e: Constant, ctx: None):
        return fpc.Constant(e.val)

    def _visit_digits(self, e: Digits, ctx: None) -> fpc.Expr:
        return fpc.Digits(e.m, e.e, e.b)

    def _visit_call(self, e: Call, ctx: None) -> fpc.Expr:
        args = [self._visit_expr(c, ctx) for c in e.args]
        return fpc.UnknownOperator(*args, name=e.name)

    def _visit_range(self, arg: Expr, ctx: None) -> fpc.Expr:
        # expand range expression
        tuple_id = str(self.gensym.fresh('i'))
        size = self._visit_expr(arg, ctx)
        return fpc.Tensor([(tuple_id, size)], fpc.Var(tuple_id))

    def _visit_size(self, arr: Expr, dim: Expr, ctx) -> fpc.Expr:
        tup = self._visit_expr(arr, ctx)
        idx = self._visit_expr(dim, ctx)
        return fpc.Size(tup, idx)

    def _visit_dim(self, arr: Expr, ctx) -> fpc.Expr:
        tup = self._visit_expr(arr, ctx)
        return fpc.Dim(tup)

    def _visit_shape(self, arr: Expr, ctx) -> fpc.Expr:
        # expand into a for loop
        #  (let ([t <tuple>])
        #    (tensor ([i (dim t)])
        #      (size t i)])))
        tuple_id = str(self.gensym.fresh('t'))
        iter_id = str(self.gensym.fresh('i'))
        tup = self._visit_expr(arr, ctx)
        return fpc.Let(
            [(tuple_id, tup)],
            fpc.Tensor(
                [(iter_id, fpc.Dim(fpc.Var(tuple_id)))],
                fpc.Size(fpc.Var(tuple_id), fpc.Var(iter_id))
            )
        )

    def _visit_zip(self, args: list[Expr], ctx: None) -> fpc.Expr:
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

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> fpc.Expr:
        cls = _unary_table.get(type(e))
        if cls is not None:
            # known unary operator
            arg = self._visit_expr(e.arg, ctx)
            return cls(arg)
        else:
            match e:
                case Range():
                    # range expression
                    return self._visit_range(e.arg, ctx)
                case Dim():
                    # dim expression
                    return self._visit_dim(e.arg, ctx)
                case Shape():
                    # shape expression
                    return self._visit_shape(e.arg, ctx)
                case _:
                    raise NotImplementedError('no FPCore operator for', e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> fpc.Expr:
        cls = _binary_table.get(type(e))
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
        cls = _ternary_table.get(type(e))
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
        cls = _nary_table.get(type(e))
        if cls is not None:
            # known n-ary operator
            return cls(*[self._visit_expr(c, ctx) for c in e.args])
        else:
            match e:
                case Zip():
                    # zip expression
                    return self._visit_zip(e.args, ctx)
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
        return fpc.Array(*[self._visit_expr(c, ctx) for c in e.args])

    def _visit_tuple_ref(self, e: TupleRef, ctx: None) -> fpc.Expr:
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return fpc.Ref(value, *slices)

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

    def _visit_tuple_set(self, e: TupleSet, ctx: None) -> fpc.Expr:
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
        idx_ids = [str(self.gensym.fresh('i')) for _ in e.slices]
        iter_id = str(self.gensym.fresh('k'))
        val_id = str(self.gensym.fresh('v'))

        # compile each component
        tuple_expr = self._visit_expr(e.array, ctx)
        idx_exprs = [self._visit_expr(idx, ctx) for idx in e.slices]
        val_expr = self._visit_expr(e.value, ctx)

        # create initial let binding
        let_bindings = [(tuple_id, tuple_expr)]
        for idx_id, idx_expr in zip(idx_ids, idx_exprs):
            let_bindings.append((idx_id, idx_expr))
        let_bindings.append((val_id, val_expr))

        # recursively generate tensor expressions
        tensor_expr = self._generate_tuple_set(tuple_id, iter_id, idx_ids, val_id)
        return fpc.Let(let_bindings, tensor_expr)


    def _visit_comp_expr(self, e: CompExpr, ctx: None) -> fpc.Expr:
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

    def _visit_assign(self, stmt: Assign, ctx: fpc.Expr):
        match stmt.binding:
            case Id():
                bindings = [(str(stmt.binding), self._visit_expr(stmt.expr, None))]
                return fpc.Let(bindings, ctx)
            case TupleBinding():
                tuple_id = str(self.gensym.fresh('t'))
                tuple_bind = (tuple_id, self._visit_expr(stmt.expr, None))
                destruct_bindings = self._compile_tuple_binding(tuple_id, stmt.binding, [])
                return fpc.LetStar([tuple_bind] + destruct_bindings, ctx)
            case _:
                raise RuntimeError('unreachable', stmt.binding)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: fpc.Expr):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_if1(self, stmt: If1Stmt, ret: fpc.Expr):
        # check that only one variable is mutated in the loop
        # the `IfBundling` pass is required to ensure this
        defs_in, defs_out = self.def_use.blocks[stmt.body]
        mutated = defs_in.mutated_in(defs_out)
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
        defs_in_ift, defs_out_ift = self.def_use.blocks[stmt.ift]
        defs_in_iff, defs_out_iff = self.def_use.blocks[stmt.iff]
        mutated_ift = defs_in_ift.mutated_in(defs_out_ift)
        mutated_iff = defs_in_iff.mutated_in(defs_out_iff)
        mutated  = list(dict.fromkeys(mutated_ift + mutated_iff)) # union with ordering

        # identify variables that were introduced in each body
        intros_ift = defs_in_ift.fresh_in(defs_out_ift)
        intros_iff = defs_in_iff.fresh_in(defs_out_iff)
        intros = list(intros_ift & intros_iff) # intersection of fresh variables

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
        defs_in, defs_out = self.def_use.blocks[stmt.body]
        mutated = defs_in.mutated_in(defs_out)
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
        defs_in, defs_out = self.def_use.blocks[stmt.body]
        mutated = defs_in.mutated_in(defs_out)
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


    def _visit_context_expr(self, e: ContextExpr, ctx: None):
        raise RuntimeError('do not call')

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
        body = self._visit_block(stmt.body, ctx)
        # extract a context value
        match stmt.ctx:
            case ContextExpr() | Var():
                raise FPCoreCompileError('Context expressions must be pre-computed', stmt.ctx)
            case ForeignVal():
                val = stmt.ctx.val
            case _:
                raise RuntimeError('unreachable', stmt.ctx)

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

    def _visit_block(self, block: StmtBlock, ctx: Optional[fpc.Expr]):
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

    def _visit_function(self, func: FuncDef, ctx: Optional[fpc.Expr]):
        args = [self._compile_arg(arg) for arg in func.args]
        body = self._visit_block(func.body, ctx)

        # metadata
        props = func.metadata.copy()
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
    """Compiler from FPy IR to FPCore"""

    def compile(self, func: Function) -> fpc.FPCore:
        # normalization passes
        ast = ContextInline.apply(func.ast, func.env)
        ast = FuncUpdate.apply(ast)
        ast = ForUnpack.apply(ast)
        ast = ForBundling.apply(ast)
        ast = WhileBundling.apply(ast)
        ast = IfBundling.apply(ast)
        # compile
        def_use = DefineUse.analyze(ast)
        return FPCoreCompileInstance(ast, def_use).compile()
