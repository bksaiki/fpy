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
    def _visit_bool(self, e: Bool, ctx: Any) -> Any:
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
    def _visit_ref_expr(self, e: RefExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Statements

    @abstractmethod
    def _visit_var_assign(self, stmt: VarAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_ref_assign(self, stmt: RefAssign, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_if_stmt(self, stmt: IfStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_while_stmt(self, stmt: WhileStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_for_stmt(self, stmt: ForStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_context(self, stmt: ContextStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_assert(self, stmt: AssertStmt, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    @abstractmethod
    def _visit_return(self, stmt: Return, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: Block, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Function

    @abstractmethod
    def _visit_function(self, func: FunctionDef, ctx: Any) -> Any:
        raise NotImplementedError('virtual method')

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an expression."""
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
            case RefExpr():
                return self._visit_ref_expr(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case _:
                raise NotImplementedError(f'unreachable {e}')

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement."""
        match stmt:
            case VarAssign():
                return self._visit_var_assign(stmt, ctx)
            case TupleAssign():
                return self._visit_tuple_assign(stmt, ctx)
            case RefAssign():
                return self._visit_ref_assign(stmt, ctx)
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
                raise NotImplementedError(f'unreachable: {stmt}')
