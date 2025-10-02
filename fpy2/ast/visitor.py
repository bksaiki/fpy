"""Visitor for the AST of the FPy language."""

from abc import ABC, abstractmethod
from typing import Any

from .fpyast import *

_expr_dispatch: dict[type[Expr], str] = {
    Var: "_visit_var",
    BoolVal: "_visit_bool",
    ForeignVal: "_visit_foreign",
    Decnum: "_visit_decnum",
    Hexnum: "_visit_hexnum",
    Integer: "_visit_integer",
    Rational: "_visit_rational",
    Digits: "_visit_digits",
    NullaryOp: "_visit_nullaryop",
    UnaryOp: "_visit_unaryop",
    BinaryOp: "_visit_binaryop",
    TernaryOp: "_visit_ternaryop",
    NaryOp: "_visit_naryop",
    Compare: "_visit_compare",
    Call: "_visit_call",
    TupleExpr: "_visit_tuple_expr",
    ListExpr: "_visit_list_expr",
    ListComp: "_visit_list_comp",
    ListRef: "_visit_list_ref",
    ListSlice: "_visit_list_slice",
    ListSet: "_visit_list_set",
    IfExpr: "_visit_if_expr",
    Attribute: "_visit_attribute",

    Round: "_visit_round",
    RoundExact: "_visit_round",
    RoundAt: "_visit_round_at",
}

_stmt_dispatch: dict[type[Stmt], str] = {
    Assign: "_visit_assign",
    IndexedAssign: "_visit_indexed_assign",
    If1Stmt: "_visit_if1",
    IfStmt: "_visit_if",
    WhileStmt: "_visit_while",
    ForStmt: "_visit_for",
    ContextStmt: "_visit_context",
    AssertStmt: "_visit_assert",
    EffectStmt: "_visit_effect",
    ReturnStmt: "_visit_return",
    PassStmt: "_visit_pass",
}

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
    def _visit_nullaryop(self, e: NullaryOp, ctx: Any) -> Any:
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

    def _visit_round(self, e: Round | RoundExact, ctx: Any) -> Any:
        return self._visit_unaryop(e, ctx)

    def _visit_round_at(self, e: RoundAt, ctx: Any) -> Any:
        return self._visit_binaryop(e, ctx)

    @abstractmethod
    def _visit_call(self, e: Call, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_compare(self, e: Compare, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_list_expr(self, e: ListExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_list_comp(self, e: ListComp, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_list_ref(self, e: ListRef, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_list_slice(self, e: ListSlice, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_list_set(self, e: ListSet, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_if_expr(self, e: IfExpr, ctx: Any) -> Any:
        ...

    @abstractmethod
    def _visit_attribute(self, e: Attribute, ctx: Any) -> Any:
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

    @abstractmethod
    def _visit_pass(self, stmt: PassStmt, ctx: Any) -> Any:
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
        # Walk the inheritance chain until we find a dispatch method or reach Expr
        for cls in type(e).__mro__:
            if cls is Expr:
                break
            method = _expr_dispatch.get(cls, None)
            if method is not None:
                func = getattr(self, method)
                return func(e, ctx)
        raise NotImplementedError(f'no dispatch method for `{e}`')

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        """Dispatch to the appropriate visit method for a statement."""
        for cls in type(stmt).__mro__:
            if cls is Expr:
                break
            method = _stmt_dispatch.get(cls, None)
            if method is not None:
                func = getattr(self, method)
                return func(stmt, ctx)
        raise NotImplementedError(f'no dispatch method for `{stmt}`')

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

    def _visit_digits(self, e: Digits, ctx: Any):
        pass

    def _visit_nullaryop(self, e: NullaryOp, ctx: Any):
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

    def _visit_call(self, e: Call, ctx: Any):
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_compare(self, e: Compare, ctx: Any):
        for c in e.args:
            self._visit_expr(c, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        for c in e.elts:
            self._visit_expr(c, ctx)

    def _visit_list_expr(self, e: ListExpr, ctx: Any):
        for c in e.elts:
            self._visit_expr(c, ctx)

    def _visit_list_ref(self, e: ListRef, ctx: Any):
        self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)

    def _visit_list_slice(self, e: ListSlice, ctx: Any):
        self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)

    def _visit_list_set(self, e: ListSet, ctx: Any):
        self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(e.expr, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        self._visit_expr(e.elt, ctx)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        self._visit_expr(e.cond, ctx)
        self._visit_expr(e.ift, ctx)
        self._visit_expr(e.iff, ctx)

    def _visit_attribute(self, e: Attribute, ctx: Any):
        self._visit_expr(e.value, ctx)

    def _visit_assign(self, stmt: Assign, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
        for s in stmt.indices:
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
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: Any):
        pass

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
        return Hexnum(e.func, e.val, e.loc)

    def _visit_integer(self, e: Integer, ctx: Any):
        return Integer(e.val, e.loc)

    def _visit_rational(self, e: Rational, ctx: Any):
        return Rational(e.func, e.p, e.q, e.loc)

    def _visit_digits(self, e: Digits, ctx: Any):
        return Digits(e.func, e.m, e.e, e.b, e.loc)

    def _visit_nullaryop(self, e: NullaryOp, ctx: Any):
        return type(e)(e.func, e.loc)

    def _visit_unaryop(self, e: UnaryOp, ctx: Any):
        arg = self._visit_expr(e.arg, ctx)
        if isinstance(e, NamedUnaryOp):
            return type(e)(e.func, arg, e.loc)
        else:
            return type(e)(arg, e.loc)

    def _visit_binaryop(self, e: BinaryOp, ctx: Any):
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        if isinstance(e, NamedBinaryOp):
            return type(e)(e.func, first, second, e.loc)
        else:
            return type(e)(first, second, e.loc)

    def _visit_ternaryop(self, e: TernaryOp, ctx: Any):
        first = self._visit_expr(e.first, ctx)
        second = self._visit_expr(e.second, ctx)
        third = self._visit_expr(e.third, ctx)
        if isinstance(e, NamedTernaryOp):
            return type(e)(e.func, first, second, third, e.loc)
        else:
            return type(e)(first, second, third, e.loc)

    def _visit_naryop(self, e: NaryOp, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        if isinstance(e, NamedNaryOp):
            return type(e)(e.func, args, e.loc)
        else:
            return type(e)(args, e.loc)

    def _visit_call(self, e: Call, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        kwargs = [ (k, self._visit_expr(v, ctx)) for k, v in e.kwargs ]
        return Call(e.func, e.fn, args, kwargs, e.loc)

    def _visit_compare(self, e: Compare, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        return Compare(e.ops, args, e.loc)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.elts]
        return TupleExpr(args, e.loc)

    def _visit_list_expr(self, e: ListExpr, ctx: Any):
        args = [self._visit_expr(arg, ctx) for arg in e.elts]
        return ListExpr(args, e.loc)

    def _visit_list_ref(self, e: ListRef, ctx: Any):
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        return ListRef(value, index, e.loc)

    def _visit_list_slice(self, e: ListSlice, ctx: Any):
        value = self._visit_expr(e.value, ctx)
        start = None if e.start is None else self._visit_expr(e.start, ctx)
        stop = None if e.stop is None else self._visit_expr(e.stop, ctx)
        return ListSlice(value, start, stop, e.loc)

    def _visit_list_set(self, e: ListSet, ctx: Any):
        array = self._visit_expr(e.value, ctx)
        slices = [self._visit_expr(s, ctx) for s in e.indices]
        value = self._visit_expr(e.expr, ctx)
        return ListSet(array, slices, value, e.loc)

    def _visit_list_comp(self, e: ListComp, ctx: Any):
        targets = [self._visit_binding(target, ctx) for target in e.targets]
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx: Any):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return IfExpr(cond, ift, iff, e.loc)

    def _visit_attribute(self, e: Attribute, ctx: Any):
        value = self._visit_expr(e.value, ctx)
        return Attribute(value, e.attr, ctx)

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
        binding = self._visit_binding(stmt.target, ctx)
        expr = self._visit_expr(stmt.expr, ctx)
        s = Assign(binding, stmt.type, expr, stmt.loc)
        return s, ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any):
        slices = [self._visit_expr(s, ctx) for s in stmt.indices]
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
        context = self._visit_expr(stmt.ctx, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = ContextStmt(stmt.target, context, body, stmt.loc)
        return s, ctx

    def _visit_assert(self, stmt: AssertStmt, ctx: Any):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            s = AssertStmt(test, None, stmt.loc)
            return s, ctx
        else:
            msg = self._visit_expr(stmt.msg, ctx)
            s = AssertStmt(test, msg, stmt.loc)
            return s, ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: Any):
        expr = self._visit_expr(stmt.expr, ctx)
        s = EffectStmt(expr, stmt.loc)
        return s, ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: Any):
        expr = self._visit_expr(stmt.expr, ctx)
        s = ReturnStmt(expr, stmt.loc)
        return s, ctx

    def _visit_pass(self, stmt: PassStmt, ctx: Any):
        s = PassStmt(stmt.loc)
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
            func.free_vars,
            func.ctx,
            body,
            func.spec,
            func.meta,
            func.env,
            loc=func.loc
        )

    # override for typing hint
    def _visit_expr(self, e: Expr, ctx: Any) -> Expr:
        return super()._visit_expr(e, ctx)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: Any) -> tuple[Stmt, Any]:
        return super()._visit_statement(stmt, ctx)
