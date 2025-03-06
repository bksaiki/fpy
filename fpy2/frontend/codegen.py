"""
This module does intermediate code generation, compiling
the abstract syntax tree (AST) to the intermediate representation (IR).
"""

from .fpyast import *
from .visitor import AstVisitor

from .. import ir

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
    func: FunctionDef

    def __init__(self, func: FunctionDef):
        self.func = func

    def lower(self) -> ir.FunctionDef:
        return self._visit_function(self.func, None)

    def _visit_var(self, e: Var, ctx: None):
        return ir.Var(e.name)

    def _visit_bool(self, e: Bool, ctx: None):
        return ir.Bool(e.val)

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

    def _visit_ref_expr(self, e: RefExpr, ctx: None):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return ir.TupleRef(value, *slices)

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return ir.IfExpr(cond, ift, iff)

    def _visit_var_assign(self, stmt: VarAssign, ctx: None):
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.VarAssign(stmt.var, ir.AnyType(), expr)

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

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: None):
        binding = self._visit_tuple_binding(stmt.binding)
        expr = self._visit_expr(stmt.expr, ctx)
        return ir.TupleAssign(binding, ir.AnyType(), expr)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: None):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        value = self._visit_expr(stmt.expr, ctx)
        return ir.RefAssign(stmt.var, slices, value)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        ift = self._visit_block(stmt.ift, ctx)
        if stmt.iff is None:
            return ir.If1Stmt(cond, ift, [])
        else:
            iff = self._visit_block(stmt.iff, ctx)
            return ir.IfStmt(cond, ift, iff, [])

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: None):
        cond = self._visit_expr(stmt.cond, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.WhileStmt(cond, body, [])

    def _visit_for_stmt(self, stmt: ForStmt, ctx: None):
        iterable = self._visit_expr(stmt.iterable, ctx)
        body = self._visit_block(stmt.body, ctx)
        return ir.ForStmt(stmt.var, ir.AnyType(), iterable, body, [])

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        props = self._visit_props(stmt.props)
        block = self._visit_block(stmt.body, ctx)
        return ir.ContextStmt(stmt.name, props, block)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        test = self._visit_expr(stmt.test, ctx)
        return ir.AssertStmt(test, stmt.msg)

    def _visit_return(self, stmt: Return, ctx: None):
        return ir.Return(self._visit_expr(stmt.expr, ctx))

    def _visit_block(self, block: Block, ctx: None):
        return ir.Block([self._visit_statement(stmt, ctx) for stmt in block.stmts])

    def _visit_props(self, props: dict[str, Any]):
        new_props: dict[str, Any] = {}
        for k, v in props.items():
            if isinstance(v, Expr):
                new_props[k] = self._visit_expr(v, None)
            else:
                new_props[k] = v
        return new_props

    def _visit_function(self, func: FunctionDef, ctx: None):
        # translate arguments
        args: list[ir.Argument] = []
        for arg in func.args:
            # TODO: use type annotation
            ty = ir.AnyType()
            args.append(ir.Argument(arg.name, ty))

        # translate properties
        props: dict[str, Any] = {}
        for name, val in func.ctx.items():
            if isinstance(val, FunctionDef):
                props[name] = self._visit_function(val, ctx)
            else:
                props[name] = val

        # translate body
        e = self._visit_block(func.body, ctx)

        return ir.FunctionDef(func.name, args, e, ir.AnyType(), props)

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: None) -> ir.Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: None) -> ir.Stmt:
        return super()._visit_statement(stmt, ctx)


class IRCodegen:
    """Lowers a FPy AST to FPy IR."""

    @staticmethod
    def lower(f: FunctionDef) -> ir.FunctionDef:
        return _IRCodegenInstance(f).lower()
