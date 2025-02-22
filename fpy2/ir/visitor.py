"""Visitor for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import Any

from .ir import *

class BaseVisitor(ABC):
    """Visitor base class for the FPy IR."""

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_var(self, e: Var, ctx: Any):
        """Visitor method for `Var` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_bool(self, e: Bool, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any):
        """Visitor method for `Decnum` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_hexnum(self, e: Hexnum, ctx: Any):
        """Visitor method for `Hexnum` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any):
        """Visitor method for `Integer` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_rational(self, e: Rational, ctx: Any):
        """Visitor method for `Rational` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_constant(self, e: Constant, ctx: Any):
        """Visitor method for `Constant` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_digits(self, e: Digits, ctx: Any):
        """Visitor method for `Digits` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        """Visitor method for `UnknownCall` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        """Visitor method for `NaryExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any):
        """Visitor method for `Compare` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        """Visitor method for `TupleExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        """Visitor method for `RefExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_set(self, e: TupleSet, ctx: Any):
        """Visitor method for `TupleSet` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        """Visitor method for `CompExpr` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        """Visitor method for `IfExpr` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_var_assign(self, stmt: VarAssign, ctx: Any):
        """Visitor method for `VarAssign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        """Visitor method for `TupleAssign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ref_assign(self, stmt: RefAssign, ctx: Any):
        """Visitor method for `RefAssign` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: Any):
        """Visitor method for `If1Stmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        """Visitor method for `IfStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any):
        """Visitor method for `WhileStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any):
        """Visitor method for `ForStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        """Visitor method for `ContextStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        """Visitor method for `AssertStmt` nodes."""
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: Return, ctx: Any):
        """Visitor method for `Return` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Phi node

    @abstractmethod
    def _visit_phis(self, phis: list[PhiNode], lctx: Any, rctx: Any):
        """
        Visitor method for a `list` of `PhiNode` nodes for non-loop nodes.

        This method is called at the join point of a control flow graph
        when _both_ branches have already been visited.
        """
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_loop_phis(self, phis: list[PhiNode], lctx: Any, rctx: Optional[Any]):
        """
        Visitor method for a `list` of `PhiNode` nodes for loop nodes.

        For loop nodes, this method is called twice:
        - once before visiting the loop body / condition (`rctx` is `None`)
        - once after visiting the loop body
        """
        raise NotImplementedError('virtual method')

    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: Block, ctx: Any):
        """Visitor method for a list of `Stmt` nodes."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Functions

    @abstractmethod
    def _visit_function(self, func: FunctionDef, ctx: Any):
        """Visitor for `fpyast.Function`."""
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any):
        """Dynamic dispatch for all `Expr` nodes."""
        match e:
            case Var():
                return self._visit_var(e, ctx)
            case Bool():
                return self._visit_bool(e, ctx)
            case Decnum():
                return self._visit_decnum(e, ctx)
            case Hexnum():
                return self._visit_hexnum(e, ctx)
            case Integer():
                return self._visit_integer(e, ctx)
            case Rational():
                return self._visit_rational(e, ctx)
            case Constant():
                return self._visit_constant(e, ctx)
            case Digits():
                return self._visit_digits(e, ctx)
            case UnknownCall():
                return self._visit_unknown(e, ctx)
            case NaryExpr():
                return self._visit_nary_expr(e, ctx)
            case Compare():
                return self._visit_compare(e, ctx)
            case TupleExpr():
                return self._visit_tuple_expr(e, ctx)
            case TupleRef():
                return self._visit_tuple_ref(e, ctx)
            case TupleSet():
                return self._visit_tuple_set(e, ctx)
            case CompExpr():
                return self._visit_comp_expr(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)

    def _visit_statement(self, stmt: Stmt, ctx: Any):
        """Dynamic dispatch for all statements."""
        match stmt:
            case VarAssign():
                return self._visit_var_assign(stmt, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(stmt, ctx)
            case RefAssign():
                return self._visit_ref_assign(stmt, ctx)
            case If1Stmt():
                return self._visit_if1_stmt(stmt, ctx)
            case IfStmt():
                return self._visit_if_stmt(stmt, ctx)
            case WhileStmt():
                return self._visit_while_stmt(stmt, ctx)
            case ForStmt():
                return self._visit_for_stmt(stmt, ctx)
            case ContextStmt():
                return self._visit_context(stmt, ctx)
            case AssertStmt():
                return self._visit_assert(stmt, ctx)
            case Return():
                return self._visit_return(stmt, ctx)
            case _:
                raise NotImplementedError('no visitor method for', stmt)

# Derived visitor types

class Visitor(BaseVisitor):
    """Visitor base class for analyzing FPy programs."""

class ReduceVisitor(BaseVisitor):
    """Visitor base class for reducing FPy programs to a value."""

class TransformVisitor(BaseVisitor):
    """Visitor base class for transforming FPy programs"""

# Default visitor implementations

class DefaultVisitor(Visitor):
    """Default visitor: visits all nodes without doing anything."""
    def _visit_var(self, e: Var, ctx: Any):
        pass

    def _visit_bool(self, e: Bool, ctx: Any):
        pass

    def _visit_decnum(self, e: Decnum, ctx: Any):
        pass

    def _visit_hexnum(self, e: Hexnum, ctx: Any):
        pass

    def _visit_integer(self, e: Integer, ctx: Any):
        pass

    def _visit_rational(self, e: Rational, ctx: Any):
        pass

    def _visit_constant(self, e: Constant, ctx: Any):
        pass

    def _visit_digits(self, e: Digits, ctx: Any):
        pass

    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        for c in e.children:
            self._visit_expr(c, ctx)

    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        for c in e.children:
            self._visit_expr(c, ctx)

    def _visit_compare(self, e: Compare, ctx: Any):
        for c in e.children:
            self._visit_expr(c, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        for c in e.children:
            self._visit_expr(c, ctx)

    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        self._visit_expr(e.value, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)

    def _visit_tuple_set(self, e: TupleSet, ctx: Any):
        self._visit_expr(e.array, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.value, ctx)

    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        self._visit_expr(e.elt, ctx)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)

    def _visit_var_assign(self, stmt: VarAssign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_ref_assign(self, stmt: RefAssign, ctx: Any):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any):
        self._visit_expr(stmt.iterable, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        self._visit_expr(stmt.test, ctx)

    def _visit_return(self, stmt: Return, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_phis(self, phis, lctx, rctx):
        pass

    def _visit_loop_phis(self, phi, lctx, rctx):
        pass

    def _visit_block(self, block: Block, ctx: Any):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FunctionDef, ctx: Any):
        self._visit_block(func.body, ctx)


class DefaultTransformVisitor(TransformVisitor):
    """Default transform visitor: identity operation on an FPy program."""

    #######################################################
    # Expressions

    def _visit_var(self, e: Var, ctx: Any):
        return Var(e.name)

    def _visit_bool(self, e: Bool, ctx: Any):
        return Bool(e.val)

    def _visit_decnum(self, e: Decnum, ctx: Any):
        return Decnum(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: Any):
        return Hexnum(e.val)

    def _visit_integer(self, e: Integer, ctx: Any):
        return Integer(e.val)

    def _visit_rational(self, e: Rational, ctx: Any):
        return Rational(e.p, e.q)

    def _visit_digits(self, e: Digits, ctx: Any):
        return Digits(e.m, e.e, e.b)

    def _visit_constant(self, e: Constant, ctx: Any):
        return Constant(e.val)

    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        args = [self._visit_expr(c, ctx) for c in e.children]
        return UnknownCall(e.name, *args)

    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        match e:
            case UnaryExpr():
                arg0 = self._visit_expr(e.children[0], ctx)
                return type(e)(arg0)
            case BinaryExpr():
                arg0 = self._visit_expr(e.children[0], ctx)
                arg1 = self._visit_expr(e.children[1], ctx)
                return type(e)(arg0, arg1)
            case TernaryExpr():
                arg0 = self._visit_expr(e.children[0], ctx)
                arg1 = self._visit_expr(e.children[1], ctx)
                arg2 = self._visit_expr(e.children[2], ctx)
                return type(e)(arg0, arg1, arg2)
            case _:
                args = [self._visit_expr(c, ctx) for c in e.children]
                return type(e)(*args)

    def _visit_compare(self, e: Compare, ctx: Any):
        ops = [op for op in e.ops]
        children = [self._visit_expr(c, ctx) for c in e.children]
        return Compare(ops, children)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        children = [self._visit_expr(c, ctx) for c in e.children]
        return TupleExpr(*children)

    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return TupleRef(value, *slices)

    def _visit_tuple_set(self, e: TupleSet, ctx: Any):
        value = self._visit_expr(e.array, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        expr = self._visit_expr(e.value, ctx)
        return TupleSet(value, slices, expr)

    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return CompExpr(e.vars, iterables, elt)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return IfExpr(cond, ift, iff)

    #######################################################
    # Statements

    def _visit_var_assign(self, stmt: VarAssign, ctx: Any):
        val = self._visit_expr(stmt.expr, ctx)
        s = VarAssign(stmt.var, stmt.ty, val)
        return s, ctx

    def _copy_tuple_binding(self, binding: TupleBinding):
        new_vars: list[Id | TupleBinding] = []
        for elt in binding:
            if isinstance(elt, Id):
                new_vars.append(elt)
            elif isinstance(elt, TupleBinding):
                new_vars.append(self._copy_tuple_binding(elt))
            else:
                raise NotImplementedError('unexpected tuple element', elt)
        return TupleBinding(new_vars)

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any):
        vars = self._copy_tuple_binding(stmt.binding)
        val = self._visit_expr(stmt.expr, ctx)
        s = TupleAssign(vars, stmt.ty, val)
        return s, ctx

    def _visit_ref_assign(self, stmt: RefAssign, ctx: Any):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = RefAssign(stmt.var, slices, expr)
        return s, ctx

    def _visit_if1_stmt(self, stmt: If1Stmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        body, rctx = self._visit_block(stmt.body, ctx)
        phis, ctx = self._visit_phis(stmt.phis, ctx, rctx)
        s = If1Stmt(cond, body, phis)
        return s, ctx

    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, lctx = self._visit_block(stmt.ift, ctx)
        iff, rctx = self._visit_block(stmt.iff, ctx)
        phis, ctx = self._visit_phis(stmt.phis, lctx, rctx)
        s = IfStmt(cond, ift, iff, phis)
        return s, ctx

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any):
        init_phis, init_ctx = self._visit_loop_phis(stmt.phis, ctx, None)
        cond = self._visit_expr(stmt.cond, init_ctx)
        body, rctx = self._visit_block(stmt.body, init_ctx)

        phis, ctx = self._visit_loop_phis(init_phis, ctx, rctx)
        s = WhileStmt(cond, body, phis)
        return s, ctx

    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any):
        iterable = self._visit_expr(stmt.iterable, ctx)
        init_phis, init_ctx = self._visit_loop_phis(stmt.phis, ctx, None)
        body, rctx = self._visit_block(stmt.body, init_ctx)

        phis, ctx = self._visit_loop_phis(init_phis, ctx, rctx)
        s = ForStmt(stmt.var, stmt.ty, iterable, body, phis)
        return s, ctx

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        body, ctx = self._visit_block(stmt.body, ctx)
        s = ContextStmt(stmt.name, stmt.props.copy(), body)
        return s, ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        test = self._visit_expr(stmt.test, ctx)
        s = AssertStmt(test, stmt.msg)
        return s, ctx

    def _visit_return(self, stmt: Return, ctx: Any):
        s = Return(self._visit_expr(stmt.expr, ctx))
        return s, ctx

    #######################################################
    # Phi node

    def _visit_phis(self, phis: list[PhiNode], lctx: Any, rctx: Any):
        phis = [PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty) for phi in phis]
        return phis, lctx # merge function just selects `lctx`

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: Any, rctx: Optional[Any]):
        phis = [PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty) for phi in phis]
        return phis, lctx # merge function just selects `lctx`

    #######################################################
    # Block

    def _visit_block(self, block: Block, ctx: Any):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            stmt, ctx = self._visit_statement(stmt, ctx)
            stmts.append(stmt)
        return Block(stmts), ctx

    #######################################################
    # Function

    def _visit_function(self, func: FunctionDef, ctx: Any):
        body, _ = self._visit_block(func.body, ctx)
        return FunctionDef(func.name, func.args, body, func.ty, func.ctx)

    #######################################################
    # Dynamic dispatch

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: Any) -> tuple[Stmt, Any]:
        return super()._visit_statement(stmt, ctx)
