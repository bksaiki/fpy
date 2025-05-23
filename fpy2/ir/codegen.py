"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from ..ast.fpyast import *
from ..ast import AstVisitor

from . import ir
from .types import *

_unary_table: dict[UnaryOpKind, type[ir.UnaryExpr]] = {
    UnaryOpKind.NEG: ir.Neg,
    UnaryOpKind.NOT: ir.Not,
    UnaryOpKind.FABS: ir.Fabs,
    UnaryOpKind.SQRT: ir.Sqrt,
    UnaryOpKind.CBRT: ir.Cbrt,
    UnaryOpKind.CEIL: ir.Ceil,
    UnaryOpKind.FLOOR: ir.Floor,
    UnaryOpKind.NEARBYINT: ir.Nearbyint,
    UnaryOpKind.ROUND: ir.Round,
    UnaryOpKind.TRUNC: ir.Trunc,
    UnaryOpKind.ACOS: ir.Acos,
    UnaryOpKind.ASIN: ir.Asin,
    UnaryOpKind.ATAN: ir.Atan,
    UnaryOpKind.COS: ir.Cos,
    UnaryOpKind.SIN: ir.Sin,
    UnaryOpKind.TAN: ir.Tan,
    UnaryOpKind.ACOSH: ir.Acosh,
    UnaryOpKind.ASINH: ir.Asinh,
    UnaryOpKind.ATANH: ir.Atanh,
    UnaryOpKind.COSH: ir.Cosh,
    UnaryOpKind.SINH: ir.Sinh,
    UnaryOpKind.TANH: ir.Tanh,
    UnaryOpKind.EXP: ir.Exp,
    UnaryOpKind.EXP2: ir.Exp2,
    UnaryOpKind.EXPM1: ir.Expm1,
    UnaryOpKind.LOG: ir.Log,
    UnaryOpKind.LOG10: ir.Log10,
    UnaryOpKind.LOG1P: ir.Log1p,
    UnaryOpKind.LOG2: ir.Log2,
    UnaryOpKind.ERF: ir.Erf,
    UnaryOpKind.ERFC: ir.Erfc,
    UnaryOpKind.LGAMMA: ir.Lgamma,
    UnaryOpKind.TGAMMA: ir.Tgamma,
    UnaryOpKind.ISFINITE: ir.IsFinite,
    UnaryOpKind.ISINF: ir.IsInf,
    UnaryOpKind.ISNAN: ir.IsNan,
    UnaryOpKind.ISNORMAL: ir.IsNormal,
    UnaryOpKind.SIGNBIT: ir.Signbit,
    UnaryOpKind.CAST: ir.Cast,
    UnaryOpKind.SHAPE: ir.Shape,
    UnaryOpKind.RANGE: ir.Range,
    UnaryOpKind.DIM: ir.Dim,
}

_binary_table: dict[BinaryOpKind, type[ir.BinaryExpr]] = {
    BinaryOpKind.ADD: ir.Add,
    BinaryOpKind.SUB: ir.Sub,
    BinaryOpKind.MUL: ir.Mul,
    BinaryOpKind.DIV: ir.Div,
    BinaryOpKind.COPYSIGN: ir.Copysign,
    BinaryOpKind.FDIM: ir.Fdim,
    BinaryOpKind.FMAX: ir.Fmax,
    BinaryOpKind.FMIN: ir.Fmin,
    BinaryOpKind.FMOD: ir.Fmod,
    BinaryOpKind.REMAINDER: ir.Remainder,
    BinaryOpKind.HYPOT: ir.Hypot,
    BinaryOpKind.ATAN2: ir.Atan2,
    BinaryOpKind.POW: ir.Pow,
    BinaryOpKind.SIZE: ir.Size,
}

_ternary_table: dict[TernaryOpKind, type[ir.TernaryExpr]] = {
    TernaryOpKind.FMA: ir.Fma,
}

_nary_table: dict[NaryOpKind, type[ir.NaryExpr]] = {
    NaryOpKind.AND: ir.And,
    NaryOpKind.OR: ir.Or
}

class _IRCodegenInstance(AstVisitor):
    """Single-use instance of lowering an AST to an IR."""
    func: FuncDef

    def __init__(self, func: FuncDef):
        self.func = func

    def lower(self) -> ir.FuncDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        return ir.Var(e.name)

    def _visit_bool(self, e: BoolVal, ctx: None):
        return ir.BoolVal(e.val)

    def _visit_foreign(self, e: ForeignVal, ctx: None):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: None):
        return ir.Decnum(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: None):
        return ir.Hexnum(e.val)

    def _visit_integer(self, e: Integer, ctx: None):
        return ir.Integer(e.val)

    def _visit_rational(self, e: Rational, ctx: None):
        return ir.Rational(e.p, e.q)

    def _visit_digits(self, e: Digits, ctx: None):
        return ir.Digits(e.m, e.e, e.b)

    def _visit_constant(self, e: Constant, ctx: None):
        return ir.Constant(e.val)

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        if e.op in _unary_table:
            arg = self._visit_expr(e.arg, ctx)
            return _unary_table[e.op](arg)
        else:
            raise NotImplementedError('unexpected op', e.op)

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        if e.op in _binary_table:
            lhs = self._visit_expr(e.left, ctx)
            rhs = self._visit_expr(e.right, ctx)
            return _binary_table[e.op](lhs, rhs)
        else:
            raise NotImplementedError('unexpected op', e.op)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None):
        arg0 = self._visit_expr(e.arg0, ctx)
        arg1 = self._visit_expr(e.arg1, ctx)
        arg2 = self._visit_expr(e.arg2, ctx)
        if e.op in _ternary_table:
            return _ternary_table[e.op](arg0, arg1, arg2)
        else:
            raise NotImplementedError('unexpected op', e.op)

    def _visit_naryop(self, e: NaryOp, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        if e.op in _nary_table:
            return _nary_table[e.op](*args)
        else:
            raise NotImplementedError('unexpected op', e.op)

    def _visit_compare(self, e: Compare, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.Compare(e.ops, args)

    def _visit_call(self, e: Call, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.UnknownCall(e.op, *args)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        elts = [self._visit_expr(arg, ctx) for arg in e.args]
        return ir.TupleExpr(*elts)

    def _visit_comp_expr(self, e: CompExpr, ctx: None):
        iterables = [self._visit_expr(arg, ctx) for arg in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return ir.CompExpr(list(e.vars), iterables, elt)

    def _visit_tuple_ref(self, e: TupleRef, ctx: None):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return ir.TupleRef(value, *slices)

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return ir.IfExpr(cond, ift, iff)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.SimpleAssign(stmt.var, AnyType(), expr)

    def _visit_tuple_binding(self, vars: TupleBinding):
        new_vars: list[Id | ir.TupleBinding] = []
        for name in vars:
            if isinstance(name, Id):
                new_vars.append(name)
            elif isinstance(name, TupleBinding):
                new_vars.append(self._visit_tuple_binding(name))
            else:
                raise NotImplementedError('unexpected tuple identifier', name)
        return ir.TupleBinding(new_vars)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: None):
        binding = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.TupleUnpack(binding, AnyType(), expr)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: None):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        value = self._visit_expr(stmt.expr, ctx)
        return ir.IndexAssign(stmt.var, slices, value)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift = self._visit_block(stmt.body, ctx)
        return ir.If1Stmt(cond, ift, [])

    def _visit_if(self, stmt: IfStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift = self._visit_block(stmt.ift, ctx)
        iff = self._visit_block(stmt.iff, ctx)
        return ir.IfStmt(cond, ift, iff, [])

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.WhileStmt(cond, body, [])

    def _visit_for(self, stmt: ForStmt, ctx: None):
        iterable = self._visit_expr(stmt.iterable, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.ForStmt(stmt.var, AnyType(), iterable, body, [])

    def _visit_context_expr(self, e: ContextExpr, ctx: None):
        match e.ctor:
            case Var():
                ctor = self._visit_var(e.ctor, ctx)
            case ForeignAttribute():
                ctor = ir.ForeignAttribute(e.ctor.name, e.ctor.attrs)

        args: list[ir.Expr | ir.ForeignAttribute] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(ir.ForeignAttribute(arg.name, arg.attrs))
                case _:
                    args.append(self._visit_expr(arg, ctx))

        kwargs: list[tuple[str, ir.Expr | ir.ForeignAttribute]] = []
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs.append((k, ir.ForeignAttribute(v.name, v.attrs)))
                case _:
                    kwargs.append((k, self._visit_expr(v, ctx)))

        return ir.ContextExpr(ctor, args, kwargs)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        match stmt.ctx:
            case Var():
                context = self._visit_var(stmt.ctx, ctx)
            case ContextExpr():
                context = self._visit_context_expr(stmt.ctx, ctx)
            case ForeignVal():
                context = ir.ForeignVal(stmt.ctx.val)
            case _:
                raise RuntimeError('unreachable', stmt.ctx)
        block = self._visit_block(stmt.body, ctx)
        return ir.ContextStmt(stmt.name, context, block)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        test = self._visit_expr(stmt.test, ctx)
        return ir.AssertStmt(test, stmt.msg)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.EffectStmt(expr)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        return ir.ReturnStmt(self._visit_expr(stmt.expr, ctx))

    def _visit_block(self, block: StmtBlock, ctx: None):
        return ir.StmtBlock([self._visit_statement(stmt, ctx) for stmt in block.stmts])

    def _visit_props(self, props: dict[str, Any]):
        new_props: dict[str, Any] = {}
        for k, v in props.items():
            if isinstance(v, Expr):
                new_props[k] = self._visit_expr(v, None)
            else:
                new_props[k] = v
        return new_props

    def _visit_function(self, func: FuncDef, ctx: None):
        # translate arguments
        args: list[ir.Argument] = []
        for arg in func.args:
            # TODO: use type annotation
            ty = AnyType()
            args.append(ir.Argument(arg.name, ty))

        # translate properties
        props: dict[str, Any] = {}
        for name, val in func.ctx.items():
            if isinstance(val, FuncDef):
                props[name] = self._visit_function(val, ctx)
            else:
                props[name] = val

        # translate body
        e = self._visit_block(func.body, ctx)

        # return type
        ty = AnyType()

        return ir.FuncDef(func.name, args, e, ty, func.ctx, func.free_vars)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> ir.Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: None) -> ir.Stmt:
        return super()._visit_statement(stmt, ctx)


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""

    @staticmethod
    def lower(f: FuncDef) -> ir.FuncDef:
        return _IRCodegenInstance(f).lower()
