"""Visitor for the AST of the FPy language."""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *

class Visitor(ABC):
    """
    Visitor base class for FPy AST nodes.
    """

    #######################################################
    # Expressions

    @abstractmethod
    def _visit_var(self, e: Var, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_bool(self, e: BoolVal, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_foreign(self, e: ForeignVal, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_decnum(self, e: Decnum, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_hexnum(self, e: Hexnum, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_integer(self, e: Integer, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_rational(self, e: Rational, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_digits(self, e: Digits, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_constant(self, e: Constant, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_unaryop(self, e: UnaryOp, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_binaryop(self, e: BinaryOp, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_ternaryop(self, e: TernaryOp, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_naryop(self, e: NaryOp, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_call(self, e: Call, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_comp_expr(self, e: CompExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_tuple_ref(self, e: TupleRef, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_tuple_set(self, e: TupleSet, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_context_expr(self, e: ContextExpr, ctx: Any) -> Any:
        ...

    #######################################################
    # Statements

    @abstractmethod
    def _visit_assign(self, stmt: Assign, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_if1(self, stmt: If1Stmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_if(self, stmt: IfStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_while(self, stmt: WhileStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_for(self, stmt: ForStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_context(self, stmt: ContextStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_assert(self, stmt: AssertStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_effect(self, stmt: EffectStmt, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_return(self, stmt: ReturnStmt, ctx: Any) -> Any:
        ...


    #######################################################
    # Block

    @abstractmethod
    def _visit_block(self, block: StmtBlock, ctx: Any) -> Any:
        ...

    #######################################################
    # Function

    @abstractmethod
    def _visit_function(self, func: FuncDef, ctx: Any) -> Any:
        ...

    #######################################################
    # Dynamic dispatch

    def _visit_expr(self, e: Expr, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for an expression."""
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
            case TupleSet():
                return self._visit_tuple_set(e, ctx)
            case IfExpr():
                return self._visit_if_expr(e, ctx)
            case ContextExpr():
                return self._visit_context_expr(e, ctx)
            case _:
                raise NotImplementedError(f'unreachable {e}')

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement."""
        match stmt:
            case Assign():
                return self._visit_assign(stmt, ctx)
            case IndexedAssign():
                return self._visit_indexed_assign(stmt, ctx)
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
                raise NotImplementedError(f'unreachable: {stmt}')

#####################################################################
# Default visitor

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

    def _visit_unaryop(self, e: UnaryOp, ctx: Any):
        self._visit_expr(e.arg, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: Any):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Any):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)

    def _visit_naryop(self, e: NaryOp, ctx: Any):
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_compare(self, e: Compare, ctx: Any):
        for c in e.args:
            self._visit_expr(c, ctx)

    def _visit_call(self, e: Call, ctx: Any):
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        for c in e.args:
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

    def _visit_assign(self, stmt: Assign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
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

#####################################################################
# Default transform visitor

class DefaultTransformVisitor(Visitor):
    """Default visitor: visits all nodes without doing anything."""

    def _visit_var(self, e: Var, ctx: Any):
        return Var(e.name, e.loc)

    def _visit_bool(self, e: BoolVal, ctx: Any):
        return BoolVal(e.val, e.loc)

    def _visit_foreign(self, e: ForeignVal, ctx: Any):
        return ForeignVal(e.val, e.loc)

    def _visit_decnum(self, e: Decnum, ctx: Any):
        return Decnum(e.val, e.loc)

    def _visit_hexnum(self, e: Hexnum, ctx: Any):
        return Hexnum(e.val, e.loc)

    def _visit_integer(self, e: Integer, ctx: Any):
        return Integer(e.val, e.loc)

    def _visit_rational(self, e: Rational, ctx: Any):
        return Rational(e.p, e.q, e.loc)

    def _visit_constant(self, e: Constant, ctx: Any):
        return Constant(e.val, e.loc)

    def _visit_digits(self, e: Digits, ctx: Any):
        return Digits(e.m, e.e, e.b, e.loc)

    def _visit_unaryop(self, e: UnaryOp, ctx: Any):
        arg = self._visit_expr(e.arg, ctx)
        return type(e)(arg, e.loc)

    def _visit_binaryop(self, e: BinaryOp, ctx: Any):
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        return type(e)(first, second, e.loc)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Any):
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        third = self._visit_expr(e.third, ctx)
        return type(e)(first, second, third, e.loc)

    def _visit_naryop(self, e: NaryOp, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return type(e)(args, e.loc)

    def _visit_compare(self, e: Compare, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return Compare(e.ops, args, e.loc)

    def _visit_call(self, e: Call, ctx: None):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return Call(e.name, args, e.loc)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return TupleExpr(args, e.loc)

    def _visit_tuple_ref(self, e: TupleRef, ctx: Any):
        value = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        return TupleRef(value, slices, e.loc)

    def _visit_tuple_set(self, e: TupleSet, ctx: Any):
        array = self._visit_expr(e.array, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.slices]
        value = self._visit_expr(e.value, ctx)
        return TupleSet(array, slices, value, e.loc)

    def _visit_comp_expr(self, e: CompExpr, ctx: Any):
        targets = [self._visit_binding(target, ctx) for target in e.targets]
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return CompExpr(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return IfExpr(cond, ift, iff, e.loc)

    def _visit_context_expr(self, e: ContextExpr, ctx: Any):
        match e.ctor:
            case Var():
                ctor = self._visit_var(e.ctor, ctx)
            case ForeignAttribute():
                ctor = ForeignAttribute(e.ctor.name, e.ctor.attrs, e.loc)
            case _:
                raise RuntimeError('unreachable', e.ctor)

        args: list[Expr | ForeignAttribute] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(ForeignAttribute(arg.name, arg.attrs, arg.loc))
                case _:
                    args.append(self._visit_expr(arg, ctx))

        kwargs: list[tuple[str, Expr | ForeignAttribute]] = []
        for name, arg in e.kwargs:
            match arg:
                case ForeignAttribute():
                    kwargs.append((name, ForeignAttribute(arg.name, arg.attrs, arg.loc)))
                case _:
                    kwargs.append((name, self._visit_expr(arg, ctx)))

        return ContextExpr(ctor, args, kwargs, e.loc)

    def _visit_binding(self, binding: Id | TupleBinding, ctx: Any):
        match binding:
            case Id():
                return binding
            case TupleBinding():
                return self._visit_tuple_binding(binding, ctx)
            case _:
                raise RuntimeError('unreachable', binding)

    def _visit_tuple_binding(self, binding: TupleBinding, ctx: Any):
        elts = [self._visit_binding(var, ctx) for var in binding]
        return TupleBinding(elts, binding.loc)

    def _visit_assign(self, stmt: Assign, ctx: Any):
        binding = self._visit_binding(stmt.binding, ctx)
        expr = self._visit_expr(stmt.expr, ctx)
        s = Assign(binding, stmt.type, expr, stmt.loc)
        return s, ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
        slices = [self._visit_expr(s, ctx) for s in stmt.slices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = IndexedAssign(stmt.var, slices, expr, stmt.loc)
        return s, ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = If1Stmt(cond, body, stmt.loc)
        return s, ctx

    def _visit_if(self, stmt: IfStmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)
        s = IfStmt(cond, ift, iff, stmt.loc)
        return s, ctx

    def _visit_while(self, stmt: WhileStmt, ctx: Any):
        cond = self._visit_expr(stmt.cond, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = WhileStmt(cond, body, stmt.loc)
        return s, ctx

    def _visit_for(self, stmt: ForStmt, ctx: Any):
        target = self._visit_binding(stmt.target, ctx)
        iterable = self._visit_expr(stmt.iterable, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = ForStmt(target, iterable, body, stmt.loc)
        return s, ctx

    def _visit_context(self, stmt: ContextStmt, ctx: Any):
        match stmt.ctx:
            case Var():
                context = self._visit_var(stmt.ctx, ctx)
            case ContextExpr():
                context = self._visit_context_expr(stmt.ctx, ctx)
            case ForeignVal():
                context = ForeignVal(stmt.ctx.val, stmt.loc)
            case _:
                raise RuntimeError('unreachable', stmt.ctx)

        body, _ = self._visit_block(stmt.body, ctx)
        s = ContextStmt(stmt.name, context, body, stmt.loc)
        return s, ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        test = self._visit_expr(stmt.test, ctx)
        s = AssertStmt(test, stmt.msg, stmt.loc)
        return s, ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        expr = self._visit_expr(stmt.expr, ctx)
        s = EffectStmt(expr, stmt.loc)
        return s, ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        expr = self._visit_expr(stmt.expr, ctx)
        s = ReturnStmt(expr, stmt.loc)
        return s, ctx

    def _visit_block(self, block: StmtBlock, ctx: Any):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            s, ctx = self._visit_statement(stmt, ctx)
            stmts.append(s)
        return StmtBlock(stmts), ctx

    def _visit_function(self, func: FuncDef, ctx: Any):
        args: list[Argument] = []
        for arg in func.args:
            args.append(Argument(arg.name, arg.type, arg.loc))
        body, _ = self._visit_block(func.body, ctx)
        return FuncDef(
            func.name,
            args,
            body,
            metadata=func.metadata,
            free_vars=func.free_vars,
            ctx=func.ctx,
            loc=func.loc
        )

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: Any) -> tuple[Stmt, Any]:
        return super()._visit_statement(stmt, ctx)
