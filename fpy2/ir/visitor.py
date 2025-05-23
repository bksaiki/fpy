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
        ...

    @abstractmethod
    def _visit_bool(self, e: BoolVal, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_foreign(self, e: ForeignVal, ctx: Any) -> Any:
        """Visitor method for `ForeignVal` nodes."""
        ...

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any):
        """Visitor method for `Decnum` nodes."""
        ...

    @abstractmethod
    def _visit_hexnum(self, e: Hexnum, ctx: Any):
        """Visitor method for `Hexnum` nodes."""
        ...

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any):
        """Visitor method for `Integer` nodes."""
        ...

    @abstractmethod
    def _visit_rational(self, e: Rational, ctx: Any):
        """Visitor method for `Rational` nodes."""
        ...

    @abstractmethod
    def _visit_constant(self, e: Constant, ctx: Any):
        """Visitor method for `Constant` nodes."""
        ...

    @abstractmethod
    def _visit_digits(self, e: Digits, ctx: Any):
        """Visitor method for `Digits` nodes."""
        ...

    @abstractmethod
    def _visit_unknown(self, e: UnknownCall, ctx: Any):
        """Visitor method for `UnknownCall` nodes."""
        ...

    @abstractmethod
    def _visit_nary_expr(self, e: NaryExpr, ctx: Any):
        """Visitor method for `NaryExpr` nodes."""
        ...

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any):
        """Visitor method for `Compare` nodes."""
        ...

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        """Visitor method for `TupleExpr` nodes."""
        ...

    @abstractmethod
    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        """Visitor method for `RefExpr` nodes."""
        ...

    @abstractmethod
    def _visit_tuple_set(self, e: TupleSet, ctx: Any):
        """Visitor method for `TupleSet` nodes."""
        ...

    @abstractmethod
    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        """Visitor method for `CompExpr` nodes."""
        ...

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        """Visitor method for `IfExpr` nodes."""
        ...

    @abstractmethod
    def _visit_context_expr(self, e: ContextExpr, ctx: Any):
        """Visitor method for `ContextExpr` nodes."""
        ...

    #######################################################
    # Statements

    @abstractmethod
    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Any):
        """Visitor method for `VarAssign` nodes."""
        ...

    @abstractmethod
    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Any):
        """Visitor method for `TupleAssign` nodes."""
        ...

    @abstractmethod
    def _visit_index_assign(self, stmt: IndexAssign, ctx: Any):
        """Visitor method for `RefAssign` nodes."""
        ...

    @abstractmethod
    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        """Visitor method for `If1Stmt` nodes."""
        ...

    @abstractmethod
    def _visit_if(self, stmt: IfStmt, ctx: Any):
        """Visitor method for `IfStmt` nodes."""
        ...

    @abstractmethod
    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        """Visitor method for `WhileStmt` nodes."""
        ...

    @abstractmethod
    def _visit_for(self, stmt: ForStmt, ctx: Any):
        """Visitor method for `ForStmt` nodes."""
        ...

    @abstractmethod
    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        """Visitor method for `ContextStmt` nodes."""
        ...

    @abstractmethod
    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        """Visitor method for `AssertStmt` nodes."""
        ...

    @abstractmethod
    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        """Visitor method for `EffectStmt` nodes."""
        ...

    @abstractmethod
    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        """Visitor method for `Return` nodes."""
        ...

    #######################################################
    # Phi node

    def _visit_phis(self, phis: list[PhiNode], lctx: Any, rctx: Any):
        """
        Visitor method for a `list` of `PhiNode` nodes for non-loop nodes.

        This method is called at the join point of a control flow graph
        when _both_ branches have already been visited.

        Implementors of `Visitor` do not have to implement this method.
        """
        raise NotImplementedError('must be overriden')

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: Any, rctx: Optional[Any]):
        """
        Visitor method for a `list` of `PhiNode` nodes for loop nodes.

        For loop nodes, this method is called twice:
        - once before visiting the loop body / condition (`rctx` is `None`)
        - once after visiting the loop body

        Implementors of `Visitor` do not have to implement this method.
        """
        raise NotImplementedError('must be overriden')

    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: StmtBlock, ctx: Any):
        """Visitor method for a list of `Stmt` nodes."""
        ...

    #######################################################
    # Functions

    @abstractmethod
    def _visit_function(self, func: FuncDef, ctx: Any):
        """Visitor for `fpyast.Function`."""
        ...

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any):
        """Dynamic dispatch for all `Expr` nodes."""
        match e:
            case Var():
                return self._visit_var(e, ctx)
            case BoolVal():
                return self._visit_bool(e, ctx)
            case ForeignVal():
                return self._visit_foreign(e, ctx)
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
            case ContextExpr():
                return self._visit_context_expr(e, ctx)
            case _:
                raise NotImplementedError('no visitor method for', e)

    def _visit_statement(self, stmt: Stmt, ctx: Any):
        """Dynamic dispatch for all statements."""
        match stmt:
            case SimpleAssign():
                return self._visit_simple_assign(stmt, ctx)
            case TupleUnpack():
                return self._visit_tuple_unpack(stmt, ctx)
            case IndexAssign():
                return self._visit_index_assign(stmt, ctx)
            case If1Stmt():
                return self._visit_if1(stmt, ctx)
            case IfStmt():
                return self._visit_if(stmt, ctx)
            case WhileStmt():
                return self._visit_while(stmt, ctx)
            case ForStmt():
                return self._visit_for(stmt, ctx)
            case ContextStmt():
                return self._visit_context(stmt, ctx)
            case AssertStmt():
                return self._visit_assert(stmt, ctx)
            case EffectStmt():
                return self._visit_effect(stmt, ctx)
            case ReturnStmt():
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

    def _visit_bool(self, e: BoolVal, ctx: Any):
        pass

    def _visit_foreign(self, e: ForeignVal, ctx: Any):
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

    def _visit_context_expr(self, e: ContextExpr, ctx: Any):
        for arg in e.args:
            if not isinstance(arg, ForeignAttribute):
                self._visit_expr(arg, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: Any):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        self._visit_expr(stmt.iterable, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        self._visit_expr(stmt.ctx, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        self._visit_expr(stmt.test, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_block(self, block: StmtBlock, ctx: Any):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: Any):
        self._visit_block(func.body, ctx)


class DefaultTransformVisitor(TransformVisitor):
    """Default transform visitor: identity operation on an FPy program."""

    #######################################################
    # Expressions

    def _visit_var(self, e: Var, ctx: Any):
        return Var(e.name)

    def _visit_bool(self, e: BoolVal, ctx: Any):
        return BoolVal(e.val)

    def _visit_foreign(self, e: ForeignVal, ctx: Any):
        return ForeignVal(e.val)

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

    def _visit_context_expr(self, e: ContextExpr, ctx: Any):
        match e.ctor:
            case Var():
                ctor = self._visit_var(e.ctor, ctx)
            case ForeignAttribute():
                ctor = ForeignAttribute(e.ctor.name, e.ctor.attrs)
            case _:
                raise RuntimeError('unreachable', e)

        args: list[Expr | ForeignAttribute] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(ForeignAttribute(arg.name, arg.attrs))
                case _:
                    args.append(self._visit_expr(arg, ctx))

        kwargs: list[tuple[str, Expr | ForeignAttribute]] = []
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs.append((k, ForeignAttribute(v.name, v.attrs)))
                case _:
                    kwargs.append((k, self._visit_expr(v, ctx)))

        return ContextExpr(ctor, args, kwargs)

    #######################################################
    # Statements

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Any):
        val = self._visit_expr(stmt.expr, ctx)
        s = SimpleAssign(stmt.var, stmt.ty, val)
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

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Any):
        vars = self._copy_tuple_binding(stmt.binding)
        val = self._visit_expr(stmt.expr, ctx)
        s = TupleUnpack(vars, stmt.ty, val)
        return s, ctx

    def _visit_index_assign(self, stmt: IndexAssign, ctx: Any):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = IndexAssign(stmt.var, slices, expr)
        return s, ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        body, rctx = self._visit_block(stmt.body, ctx)
        phis, ctx = self._visit_phis(stmt.phis, ctx, rctx)
        s = If1Stmt(cond, body, phis)
        return s, ctx

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, lctx = self._visit_block(stmt.ift, ctx)
        iff, rctx = self._visit_block(stmt.iff, ctx)
        phis, ctx = self._visit_phis(stmt.phis, lctx, rctx)
        s = IfStmt(cond, ift, iff, phis)
        return s, ctx

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        init_phis, init_ctx = self._visit_loop_phis(stmt.phis, ctx, None)
        cond = self._visit_expr(stmt.cond, init_ctx)
        body, rctx = self._visit_block(stmt.body, init_ctx)

        phis, ctx = self._visit_loop_phis(init_phis, ctx, rctx)
        s = WhileStmt(cond, body, phis)
        return s, ctx

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        iterable = self._visit_expr(stmt.iterable, ctx)
        init_phis, init_ctx = self._visit_loop_phis(stmt.phis, ctx, None)
        body, rctx = self._visit_block(stmt.body, init_ctx)

        phis, ctx = self._visit_loop_phis(init_phis, ctx, rctx)
        s = ForStmt(stmt.var, stmt.ty, iterable, body, phis)
        return s, ctx

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        match stmt.ctx:
            case Var():
                context = self._visit_var(stmt.ctx, ctx)
            case ContextExpr():
                context = self._visit_context_expr(stmt.ctx, ctx)
            case ForeignVal():
                context = ForeignVal(stmt.ctx.val)
            case _:
                raise RuntimeError('unreachable', stmt.ctx)
        body, ctx = self._visit_block(stmt.body, ctx)
        s = ContextStmt(stmt.name, context, body)
        return s, ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        test = self._visit_expr(stmt.test, ctx)
        s = AssertStmt(test, stmt.msg)
        return s, ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        expr = self._visit_expr(stmt.expr, ctx)
        s = EffectStmt(expr)
        return s, ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        s = ReturnStmt(self._visit_expr(stmt.expr, ctx))
        return s, ctx

    #######################################################
    # Phi node

    def _visit_phis(self, phis: list[PhiNode], lctx: Any, rctx: Any):
        # does nothing, just copies the phis
        phis = [PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty) for phi in phis]
        return phis, lctx

    def _visit_loop_phis(self, phis: list[PhiNode], lctx: Any, rctx: Optional[Any]):
        # does nothing, just copies the phis
        phis = [PhiNode(phi.name, phi.lhs, phi.rhs, phi.ty) for phi in phis]
        return phis, lctx

    #######################################################
    # Block

    def _visit_block(self, block: StmtBlock, ctx: Any):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            stmt, ctx = self._visit_statement(stmt, ctx)
            stmts.append(stmt)
        return StmtBlock(stmts), ctx

    #######################################################
    # Function

    def _visit_function(self, func: FuncDef, ctx: Any):
        body, _ = self._visit_block(func.body, ctx)
        return FuncDef(func.name, func.args, body, func.ty, func.ctx, func.free_vars)

    #######################################################
    # Dynamic dispatch

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: Any) -> tuple[Stmt, Any]:
        return super()._visit_statement(stmt, ctx)
