"""Visitor for the AST of the FPy language."""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *

class AstVisitor(ABC):
    """
    Visitor base class for FPy AST nodes.
    """

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_var(self, e: Var, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_bool(self, e: BoolVal, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_hexnum(self, e: Hexnum, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_rational(self, e: Rational, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_digits(self, e: Digits, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_constant(self, e: Constant, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_unaryop(self, e: UnaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_binaryop(self, e: BinaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ternaryop(self, e: TernaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_naryop(self, e: NaryOp, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_call(self, e: Call, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_comp_expr(self, e: CompExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')
    
    @abstractmethod
    def _visit_tuple_ref(self, e: TupleRef, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_index_assign(self, stmt: IndexAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if(self, stmt: IfStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while(self, stmt: WhileStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for(self, stmt: ForStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_context(self, stmt: ContextStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_assert(self, stmt: AssertStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_effect(self, stmt: EffectStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: ReturnStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')


    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: StmtBlock, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Function

    @abstractmethod
    def _visit_function(self, func: FuncDef, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an expression."""
        match e:
            case Var():
                return self._visit_var(e, ctx)
            case BoolVal():
                return self._visit_bool(e, ctx)
            case Decnum():
                return self._visit_decnum(e, ctx)
            case Hexnum():
                return self._visit_hexnum(e, ctx)
            case Integer():
                return self._visit_integer(e, ctx)
            case Rational():
                return self._visit_rational(e, ctx)
            case Digits():
                return self._visit_digits(e, ctx)
            case Constant():
                return self._visit_constant(e, ctx)
            case UnaryOp():
                return self._visit_unaryop(e, ctx)
            case BinaryOp():
                return self._visit_binaryop(e, ctx)
            case TernaryOp():
                return self._visit_ternaryop(e, ctx)
            case NaryOp():
                return self._visit_naryop(e, ctx)
            case Compare():
                return self._visit_compare(e, ctx)
            case Call():
                return self._visit_call(e, ctx)
            case TupleExpr():
                return self._visit_tuple_expr(e, ctx)
            case CompExpr():
                return self._visit_comp_expr(e, ctx)
            case TupleRef():
                return self._visit_tuple_ref(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError(f'unreachable {e}')

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement."""
        match stmt:
            case SimpleAssign():
                return self._visit_simple_assign(stmt, ctx)
            case TupleUnpack():
                return self._visit_tuple_unpack(stmt, ctx)
            case IndexAssign():
                return self._visit_index_assign(stmt, ctx)
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
                raise NotImplementedError(f'unreachable: {stmt}')

#####################################################################
# Default visitor

class DefaultAstVisitor(AstVisitor):
    """Default visitor: visits all nodes without doing anything."""
    def _visit_var(self, e: Var, ctx: Any):
        pass

    def _visit_bool(self, e: BoolVal, ctx: Any):
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

    def _visit_unaryop(self, e: UnaryOp, ctx: Any):
        self._visit_expr(e.arg, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: Any):
        self._visit_expr(e.left, ctx)
        self._visit_expr(e.right, ctx)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Any):
        self._visit_expr(e.arg0, ctx)
        self._visit_expr(e.arg1, ctx)
        self._visit_expr(e.arg2, ctx)

    def _visit_naryop(self, e: NaryOp, ctx: Any):
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_compare(self, e: Compare, ctx: Any):
        for c in e.args:
            self._visit_expr(c, ctx)

    def _visit_call(self, e: Call, ctx: None):
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        for c in e.args:
            self._visit_expr(c, ctx)

    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        self._visit_expr(e.value, ctx)
        for s in e.slices:
            self._visit_expr(s, ctx)

    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        self._visit_expr(e.elt, ctx)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_index_assign(self, stmt: IndexAssign, ctx: Any):
        for s in stmt.slices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        if stmt.iff is not None:
            self._visit_block(stmt.iff, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        self._visit_expr(stmt.iterable, ctx)
        self._visit_block(stmt.body, ctx)

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
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
