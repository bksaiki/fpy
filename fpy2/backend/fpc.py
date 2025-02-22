"""Compilation from FPy IR to FPCore"""

from typing import Optional

import titanfp.fpbench.fpcast as fpc 

from ..runtime import Function
from ..passes import *
from ..ir import *
from ..utils import Gensym

from .backend import Backend

_op_table = {
    '+': fpc.Add,
    '-': fpc.Sub,
    '*': fpc.Mul,
    '/': fpc.Div,
    'fabs': fpc.Fabs,
    'sqrt': fpc.Sqrt,
    'fma': fpc.Fma,
    'neg': fpc.Neg,
    'copysign': fpc.Copysign,
    'fdim': fpc.Fdim,
    'fmax': fpc.Fmax,
    'fmin': fpc.Fmin,
    'fmod': fpc.Fmod,
    'remainder': fpc.Remainder,
    'hypot': fpc.Hypot,
    'cbrt': fpc.Cbrt,
    'ceil': fpc.Ceil,
    'floor': fpc.Floor,
    'nearbyint': fpc.Nearbyint,
    'round': fpc.Round,
    'trunc': fpc.Trunc,
    'acos': fpc.Acos,
    'asin': fpc.Asin,
    'atan': fpc.Atan,
    'atan2': fpc.Atan2,
    'cos': fpc.Cos,
    'sin': fpc.Sin,
    'tan': fpc.Tan,
    'acosh': fpc.Acosh,
    'asinh': fpc.Asinh,
    'atanh': fpc.Atanh,
    'cosh': fpc.Cosh,
    'sinh': fpc.Sinh,
    'tanh': fpc.Tanh,
    'exp': fpc.Exp,
    'exp2': fpc.Exp2,
    'expm1': fpc.Expm1,
    'log': fpc.Log,
    'log10': fpc.Log10,
    'log1p': fpc.Log1p,
    'log2': fpc.Log2,
    'pow': fpc.Pow,
    'erf': fpc.Erf,
    'erfc': fpc.Erfc,
    'lgamma': fpc.Lgamma,
    'tgamma': fpc.Tgamma,
    'isfinite': fpc.Isfinite,
    'isinf': fpc.Isinf,
    'isnan': fpc.Isnan,
    'isnormal': fpc.Isnormal,
    'signbit': fpc.Signbit,
    'not': fpc.Not,
    'or': fpc.Or,
    'and': fpc.And,
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


class FPCoreCompileInstance(ReduceVisitor):
    """Compilation instance from FPy to FPCore"""
    func: FunctionDef
    gensym: Gensym

    def __init__(self, func: FunctionDef):
        uses = DefineUse().analyze(func)
        self.func = func
        self.gensym = Gensym(reserved=uses.keys())

    def compile(self) -> fpc.FPCore:
        f = self._visit_function(self.func, None)
        assert isinstance(f, fpc.FPCore), 'unexpected result type'
        return f

    def _compile_arg(self, arg: Argument):
        match arg.ty:
            case RealType():
                return arg.name, None, None
            case AnyType():
                return arg.name, None, None
            case _:
                raise FPCoreCompileError('unsupported argument type', arg)

    def _compile_tuple_binding(self, tuple_id: str, binding: TupleBinding, pos: list[int]):
        tuple_binds: list[tuple[str, fpc.Expr]] = []
        for i, elt in enumerate(binding):
            match elt:
                case Id():
                    idxs = [fpc.Integer(idx) for idx in [i, *pos]]
                    tuple_binds.append((str(elt), fpc.Ref(fpc.Var(tuple_id), *idxs)))
                case TupleBinding():
                    tuple_binds += self._compile_tuple_binding(tuple_id, elt, [i, *pos])
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

    def _visit_var(self, e, ctx) -> fpc.Expr:
        return fpc.Var(str(e.name))

    def _visit_bool(self, e: Bool, ctx: None):
        return fpc.Constant('TRUE' if e.val else 'FALSE')

    def _visit_decnum(self, e, ctx) -> fpc.Expr:
        return fpc.Decnum(e.val)

    def _visit_hexnum(self, e, ctx):
        raise fpc.Hexnum(e.val)

    def _visit_integer(self, e, ctx) -> fpc.Expr:
        return fpc.Integer(e.val)

    def _visit_rational(self, e, ctx):
        raise fpc.Rational(e.p, e.q)

    def _visit_constant(self, e, ctx):
        raise fpc.Constant(e.val)

    def _visit_digits(self, e, ctx) -> fpc.Expr:
        return fpc.Digits(e.m, e.e, e.b)

    def _visit_unknown(self, e, ctx) -> fpc.Expr:
        args = [self._visit_expr(c, ctx) for c in e.children]
        return fpc.UnknownOperator(e.name, *args)

    def _visit_nary_expr(self, e, ctx) -> fpc.Expr:
        if e.name == Range.name:
            # expand range expression
            tuple_id = 'i' # only identifier in scope => no need for uniqueness
            size = self._visit_expr(e.children[0], ctx)
            return fpc.Tensor([(tuple_id, size)], fpc.Var(tuple_id))
        else:
            cls = _op_table.get(e.name)
            if cls is None:
                raise NotImplementedError('no FPCore operator for', e.name)
            return cls(*[self._visit_expr(c, ctx) for c in e.children])

    def _visit_compare(self, e: Compare, ctx: None) -> fpc.Expr:
        assert e.ops != [], 'should not be empty'
        match e.ops:
            case [op]:
                # 2-argument case: just compile
                cls = self._compile_compareop(op)
                arg0 = self._visit_expr(e.children[0], ctx)
                arg1 = self._visit_expr(e.children[1], ctx)
                return cls(arg0, arg1)
            case [op, *ops]:
                # N-argument case:
                # TODO: want to evaluate each argument only once;
                #       may need to let-bind in case any argument is
                #       used multiple times
                args = [self._visit_expr(arg, ctx) for arg in e.children]
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

    def _visit_tuple_expr(self, e, ctx) -> fpc.Expr:
        return fpc.Array(*[self._visit_expr(c, ctx) for c in e.children])

    def _visit_tuple_ref(self, e, ctx) -> fpc.Expr:
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return fpc.Ref(value, *slices)

    def _generate_tuple_set(self, tuple_id: str, iter_id: str, idx_ids: list[str], val_id: str):
        # dimension bindings
        idx_id = idx_ids[0]
        tensor_dims = [(iter_id, _size0_expr(tuple_id))]
        # generate if expression
        cond_expr = fpc.EQ(fpc.Var(iter_id), fpc.Var(idx_id))
        iff_expr = fpc.Ref(fpc.Var(tuple_id), fpc.Var(idx_id))
        if len(idx_ids) == 1:
            ift_expr = fpc.Var(val_id)
        else:
            let_bindings = [(tuple_id, fpc.Ref(fpc.Var(tuple_id), fpc.Var(idx_id)))]
            rec_expr = self._generate_tuple_set(tuple_id, iter_id, idx_ids[1:], val_id)
            ift_expr = fpc.Let(let_bindings, rec_expr)
        if_expr = fpc.If(cond_expr, ift_expr, iff_expr)
        return fpc.Tensor(tensor_dims, if_expr)

    def _visit_tuple_set(self, e, ctx) -> fpc.Expr:
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


    def _visit_comp_expr(self, e, ctx) -> fpc.Expr:
        if len(e.vars) == 1:
            # simple case:
            # (let ([t <iterable>]) (tensor ([i (size t)]) (let ([<var> (ref t i)]) <elt>))
            iterable = e.iterables[0]
            var = str(e.vars[0])

            tuple_id = str(self.gensym.fresh('t'))
            iter_id = str(self.gensym.fresh('i'))
            iterable = self._visit_expr(iterable, ctx)
            elt = self._visit_expr(e.elt, ctx)

            let_bindings = [(tuple_id, iterable)]
            tensor_dims: list[tuple[str, fpc.Expr]] = [(iter_id, _size0_expr(tuple_id))]
            ref_bindings: list[tuple[str, fpc.Expr]] = [(var, fpc.Ref(fpc.Var(tuple_id), fpc.Var(iter_id)))]
            return fpc.Let(let_bindings, fpc.Tensor(tensor_dims, fpc.Let(ref_bindings, elt)))
        else:
            # hard case:
            # (let ([t0 <iterable>] ...)
            #   (let ([n0 (size t0)] ...)
            #     (tensor ([k (! :precision integer (* n0 ...))])
            #       (let ([i0 (! :precision integer :round toZero (/ k (* n1 ...)))]
            #             [i1 (! :precision integer :round toZero (fmod (/ k (* n2 ...)) n1))]
            #             ...
            #             [iN (! :precision integer :round toZero (fmod k nN))])
            #         (let ([v0 (ref t0 i0)] ...)
            #           <elt>))))

            # bind the tuples to temporaries
            tuple_ids = [str(self.gensym.fresh('t')) for _ in e.vars]
            tuple_binds: list[tuple[str, fpc.Expr]] = [
                (tid, self._visit_expr(iterable, ctx))
                for tid, iterable in zip(tuple_ids, e.iterables)
            ]
            # bind the sizes to temporaries
            size_ids = [str(self.gensym.fresh('n')) for _ in e.vars]
            size_binds: list[tuple[str, fpc.Expr]] = [
                (sid, _size0_expr(tid))
                for sid, tid in zip(size_ids, tuple_ids)
            ]
            # bind the indices to temporaries
            idx_ctx = { 'precision': 'integer', 'round': 'toZero' }
            idx_ids = [str(self.gensym.fresh('i')) for _ in e.vars]
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
            ref_ids = [str(self.gensym.refresh(var)) for var in e.vars]
            ref_binds: list[tuple[str, fpc.Expr]] = [
                (rid, fpc.Ref(fpc.Var(tid), fpc.Var(iid)))
                for rid, tid, iid in zip(ref_ids, tuple_ids, idx_ids)
            ]
            # element expression
            elt = self._visit_expr(e.elt, ctx)
            # compose the expression
            tensor_expr = fpc.Tensor([(iter_id, iter_expr)], fpc.Let(idx_binds, fpc.Let(ref_binds, elt)))
            return fpc.Let(tuple_binds, fpc.Let(size_binds, tensor_expr))

    def _visit_if_expr(self, e, ctx) -> fpc.Expr:
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return fpc.If(cond, ift, iff)

    def _visit_var_assign(self, stmt: VarAssign, ctx: fpc.Expr):
        bindings = [(str(stmt.var), self._visit_expr(stmt.expr, None))]
        return fpc.Let(bindings, ctx)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: fpc.Expr):
        tuple_id = str(self.gensym.fresh('t'))
        tuple_bind = (tuple_id, self._visit_expr(stmt.expr, None))
        destruct_bindings = self._compile_tuple_binding(tuple_id, stmt.binding, [])
        return fpc.Let([tuple_bind] + destruct_bindings, ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: fpc.Expr):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_if1_stmt(self, stmt, ctx):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_if_stmt(self, stmt, ctx):
        raise FPCoreCompileError(f'cannot compile to FPCore: {type(stmt).__name__}')

    def _visit_while_stmt(self, stmt, ctx: fpc.Expr):
        if len(stmt.phis) != 1:
            raise FPCoreCompileError('while loops must have exactly one phi node')
        phi = stmt.phis[0]
        name, init, update = str(phi.name), str(phi.lhs), str(phi.rhs)
        cond = self._visit_expr(stmt.cond, None)
        body = self._visit_block(stmt.body, fpc.Var(update))
        return fpc.While(cond, [(name, fpc.Var(init), body)], ctx)

    def _visit_for_stmt(self, stmt, ctx):
        if len(stmt.phis) != 1:
            raise FPCoreCompileError('for loops must have exactly one phi node')
        # phi nodes
        phi = stmt.phis[0]
        name, init, update = str(phi.name), str(phi.lhs), str(phi.rhs)
        # fresh variable for the iterable value
        tuple_id = str(self.gensym.fresh('t'))
        iterable = self._visit_expr(stmt.iterable, None)
        body = self._visit_block(stmt.body, fpc.Var(update))
        # index variables and state merging
        dim_binding = (str(stmt.var), _size0_expr(tuple_id))
        while_binding = (name, fpc.Var(init), body)
        return fpc.Let([(tuple_id, iterable)], fpc.For([dim_binding], [while_binding], ctx))

    def _visit_context(self, stmt, ctx):
        body = self._visit_block(stmt.body, ctx)
        return fpc.Ctx(stmt.props, body)

    def _visit_assert(self, stmt, ctx):
        # strip the assertion
        return ctx

    def _visit_return(self, stmt, ctx) -> fpc.Expr:
        return self._visit_expr(stmt.expr, ctx)

    def _visit_phis(self, phis, lctx, rctx):
        raise NotImplementedError('do not call directly')

    def _visit_loop_phis(self, phis, lctx, rctx):
        raise NotImplementedError('do not call directly')

    def _visit_block(self, block, ctx: Optional[fpc.Expr]):
        if ctx is None:
            e = self._visit_statement(block.stmts[-1], None)
            stmts = block.stmts[:-1]
        else:
            e = ctx
            stmts = block.stmts

        for stmt in reversed(stmts):
            if isinstance(stmt, Return):
                raise FPCoreCompileError('return statements must be at the end of blocks')
            e = self._visit_statement(stmt, e)

        return e

    def _visit_function(self, func, ctx: Optional[fpc.Expr]):
        args = [self._compile_arg(arg) for arg in func.args]
        body = self._visit_block(func.body, ctx)
        # TODO: parse data
        ident = func.name
        name = func.ctx.get('name', None)
        pre = func.ctx.get('pre', None)
        spec = func.ctx.get('spec', None)
        return fpc.FPCore(
            inputs=args,
            e=body,
            props=func.ctx.copy(),
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
        ir = func.ir

        # normalization passes
        ir = FuncUpdate.apply(ir)
        ir = ForBundling.apply(ir)
        ir = WhileBundling.apply(ir)
        ir = SimplifyIf.apply(ir)

        return FPCoreCompileInstance(ir).compile()
