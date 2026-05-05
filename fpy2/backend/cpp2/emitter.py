"""
cpp2 backend: emitter (Phase 2 — scalar arithmetic).

The emitter walks the (post-monomorphization) :class:`FuncDef` and
produces a C++ source string.  This first cut covers the language's
*scalar-arithmetic* subset only:

- Function signatures (arg/return types come from
  :class:`Cpp2PipelineResult.def_storage`).
- :class:`Assign` and :class:`ReturnStmt`.
- :class:`Var`, numeric literals.
- :class:`UnaryOp`/:class:`BinaryOp` for ``+``/``-``/``*``/``/``,
  :class:`Neg`, :class:`Abs`.
- :class:`ContextStmt` is descended into without emitting
  ``fesetround`` (rounding-boundary handling lands in a later phase).

Anything else raises :class:`Cpp2CompileError`.  Subsequent phases will
add booleans, control flow, lists, and rounding boundaries.
"""

from fractions import Fraction

from ...analysis import DefineUseAnalysis, FormatAnalysis
from ...ast.fpyast import (
    Abs, Add, Assign, BinaryOp, BoolVal, Compare, Decnum, Digits, Div,
    Expr, ForStmt, FuncDef, Hexnum, If1Stmt, IfStmt, Integer, Mul, NamedId,
    Neg, Range1, Range2, Range3, Rational, ReturnStmt, Stmt, StmtBlock,
    Sub, UnaryOp, Var, ContextStmt, UnderscoreId, WhileStmt,
)
from ...ast.visitor import Visitor
from ...utils.compare import CompareOp

from .storage import StorageSelectionError
from .storage_infer import StorageAnalysis
from .types import CppType


class _IndentedWriter:
    """Tiny line-oriented C++ source builder."""

    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0

    def add_line(self, line: str = ''):
        if line:
            self._lines.append('    ' * self._depth + line)
        else:
            self._lines.append('')

    def indent(self):
        self._depth += 1

    def dedent(self):
        self._depth -= 1

    def render(self) -> str:
        return '\n'.join(self._lines) + '\n'


# Maps from AST op nodes to their infix C++ operators.
_BINARY_OPS: dict[type, str] = {
    Add: '+',
    Sub: '-',
    Mul: '*',
    Div: '/',
}


class Cpp2EmitError(Exception):
    """Raised for unsupported AST shapes during emission."""
    pass


class _Cpp2Emitter(Visitor):
    """Single-use visitor that produces a C++ source string."""

    ast: FuncDef
    storage: StorageAnalysis
    def_use: DefineUseAnalysis
    format_info: FormatAnalysis
    writer: _IndentedWriter

    def __init__(
        self,
        ast: FuncDef,
        storage: StorageAnalysis,
        def_use: DefineUseAnalysis,
        format_info: FormatAnalysis,
    ):
        self.ast = ast
        self.storage = storage
        self.def_use = def_use
        self.format_info = format_info
        self.writer = _IndentedWriter()

    # ------------------------------------------------------------------
    # Public entry

    def emit(self) -> str:
        self._visit_function(self.ast, None)
        return self.writer.render()

    # ------------------------------------------------------------------
    # Helpers

    def _name_for_var_use(self, var: Var) -> str:
        d = self.def_use.find_def_from_use(var)
        return self.storage.def_to_name[d]

    def _name_for_def_at_site(self, name: NamedId, site) -> str:
        d = self.def_use.find_def_from_site(name, site)
        return self.storage.def_to_name[d]

    def _storage_for_arg(self, arg) -> CppType:
        d = self.def_use.find_def_from_site(arg.name, arg)
        return self.storage.class_storage[self.storage.def_class[d]]

    # ------------------------------------------------------------------
    # Function emission

    def _visit_function(self, func: FuncDef, ctx):
        # Determine return type from the return statement's expression
        # bound (the slice assumes a single return, possibly nested in a
        # ``with`` block).
        ret_ty = self._infer_return_storage(func)
        ret_str = ret_ty.format() if ret_ty is not None else 'void'

        # Emit arg list.  Each argument's class is anchored to the bare
        # source name in the phi-web, so it's safe to use ``arg.name``
        # directly here (and any body-side reassignment that flows
        # through the arg's phi-class will write to the same variable).
        arg_strs: list[str] = []
        for arg in func.args:
            if not isinstance(arg.name, NamedId):
                raise Cpp2EmitError(f'unsupported arg pattern: {arg.name!r}')
            storage = self._storage_for_arg(arg)
            arg_strs.append(f'{storage.format()} {arg.name}')
        sig = f'{ret_str} {func.name}({", ".join(arg_strs)})'

        self.writer.add_line(sig + ' {')
        self.writer.indent()
        self._visit_block(func.body, None)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_hoist_for_class(self, c):
        """
        Emit a zero-initialised C++ variable declaration for a single
        storage class (used to anchor declarations just before the
        ``IfStmt`` that introduces a fresh-in-both-branches name).
        """
        name = self.storage.def_to_name[self.storage.class_members[c][0]]
        storage = self.storage.class_storage[c]
        # Zero-initialise via ``T name{};`` so reads-before-writes
        # are well-defined (FPy analyses ensure this can't happen,
        # but the initialiser also serves as a paper-trail).
        self.writer.add_line(f'{storage.format()} {name}{{}};')

    def _infer_return_storage(self, func: FuncDef) -> CppType | None:
        """Pull the type of the function's return expression from the
        format analysis (Phase-2 slice: assumes one return statement,
        possibly nested inside a ``with`` block)."""
        from .storage import choose_storage

        def find_return(block: StmtBlock) -> ReturnStmt | None:
            for stmt in block.stmts:
                if isinstance(stmt, ReturnStmt):
                    return stmt
                if isinstance(stmt, ContextStmt):
                    nested = find_return(stmt.body)
                    if nested is not None:
                        return nested
            return None

        ret = find_return(func.body)
        if ret is None:
            return None
        fmt = self.format_info.by_expr.get(ret.expr)
        try:
            return choose_storage(fmt)
        except StorageSelectionError as e:
            raise Cpp2EmitError(f'return type: {e}') from e

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_block(self, block: StmtBlock, ctx):
        for stmt in block.stmts:
            # Storage classes anchored to this stmt (currently only
            # ``IfStmt``s with is_intro phi merges) declare just before
            # the stmt, narrowing the variable's scope.
            for c in self.storage.hoists_before.get(stmt, ()):
                self._emit_hoist_for_class(c)
            self._visit_statement(stmt, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        if not isinstance(stmt.target, NamedId):
            raise Cpp2EmitError(
                f'unsupported assignment target {stmt.target!r}; '
                'tuple unpacking and underscore targets land in a later phase'
            )
        rhs = self._visit_expr(stmt.expr, ctx)
        # ``StorageInfer`` maps this Assign's SSA def to a C++ variable
        # and tells us whether the assign should declare the variable
        # (single-writer class) or just reassign into a hoisted decl
        # (multi-writer class).
        target_def = self.def_use.find_def_from_site(stmt.target, stmt)
        target_name = self.storage.def_to_name[target_def]
        if target_def in self.storage.declare_at_assign:
            storage = self.storage.class_storage[self.storage.def_class[target_def]]
            self.writer.add_line(f'{storage.format()} {target_name} = {rhs};')
        else:
            self.writer.add_line(f'{target_name} = {rhs};')

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'return {rhs};')

    def _visit_context(self, stmt: ContextStmt, ctx):
        # Phase 2: rounding boundaries are not yet emitted.  Descend
        # into the body and trust the format-inference / storage choices
        # made earlier.  ``with`` itself doesn't generate C++ statements.
        if not isinstance(stmt.target, UnderscoreId):
            raise Cpp2EmitError(
                'binding the active context to a name is not supported '
                'in the Phase-2 slice'
            )
        self._visit_block(stmt.body, ctx)

    def _visit_statement(self, stmt: Stmt, ctx):
        # Dispatch into the visitor methods, raising cleanly for
        # unsupported shapes.
        try:
            return super()._visit_statement(stmt, ctx)
        except NotImplementedError:
            raise Cpp2EmitError(
                f'unsupported statement kind: {type(stmt).__name__}'
            )

    # ------------------------------------------------------------------
    # Expression visitors — return a C++ source fragment

    def _visit_expr(self, e: Expr, ctx) -> str:  # type: ignore[override]
        try:
            return super()._visit_expr(e, ctx)
        except NotImplementedError:
            raise Cpp2EmitError(
                f'unsupported expression kind: {type(e).__name__}'
            )

    def _visit_var(self, e: Var, ctx) -> str:
        # Resolve the use to its SSA def, then look up the C++
        # identifier ``StorageInfer`` assigned to that def's class.
        if not isinstance(e.name, NamedId):
            raise Cpp2EmitError(f'non-named variable use: {e.name!r}')
        return self._name_for_var_use(e)

    def _visit_decnum(self, e: Decnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_integer(self, e: Integer, ctx) -> str:
        # Integer literals print directly.
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_digits(self, e: Digits, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _emit_numeric_literal(self, v: Fraction) -> str:
        """
        Emit a numeric literal as a C++ expression.

        Integer-valued rationals → integer literal.
        Otherwise → ``(double)num / denom`` to force float division.
        """
        if v.denominator == 1:
            return str(v.numerator)
        return f'((double){v.numerator} / (double){v.denominator})'

    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        arg = self._visit_expr(e.arg, ctx)
        if isinstance(e, Neg):
            return f'(-{arg})'
        if isinstance(e, Abs):
            return f'std::fabs({arg})'
        raise Cpp2EmitError(f'unsupported unary op: {type(e).__name__}')

    def _visit_binaryop(self, e: BinaryOp, ctx) -> str:
        op = _BINARY_OPS.get(type(e))
        if op is None:
            raise Cpp2EmitError(f'unsupported binary op: {type(e).__name__}')
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        return f'({lhs} {op} {rhs})'

    # ------------------------------------------------------------------
    # Stubs for AST nodes not handled in Phase 2 — each raises a clean
    # error pointing at the node kind.  Subsequent phases will replace
    # these one at a time.

    def _unsupported(self, kind: str):
        raise Cpp2EmitError(f'cpp2 Phase 2 does not handle {kind}')

    def _visit_bool(self, e: BoolVal, ctx) -> str:
        return 'true' if e.val else 'false'

    def _visit_compare(self, e: Compare, ctx) -> str:
        # Chained comparisons (``a < b < c``) expand to a conjunction
        # of pairwise comparisons.  FPy's evaluation order is
        # left-to-right with no short-circuit semantics that differ
        # from C++'s — the operands are pure expressions, so ``&&``
        # is fine.
        args = [self._visit_expr(a, ctx) for a in e.args]
        clauses = [
            f'({args[i]} {op.symbol()} {args[i + 1]})'
            for i, op in enumerate(e.ops)
        ]
        if len(clauses) == 1:
            return clauses[0]
        return '(' + ' && '.join(clauses) + ')'

    def _visit_foreign(self, e, ctx): self._unsupported('ForeignVal')
    def _visit_attribute(self, e, ctx): self._unsupported('Attribute')
    def _visit_nullaryop(self, e, ctx): self._unsupported('NullaryOp')
    def _visit_ternaryop(self, e, ctx): self._unsupported('TernaryOp')
    def _visit_naryop(self, e, ctx): self._unsupported('NaryOp')
    def _visit_round(self, e, ctx): self._unsupported('Round / RoundExact')
    def _visit_round_at(self, e, ctx): self._unsupported('RoundAt')
    def _visit_call(self, e, ctx): self._unsupported('Call')
    def _visit_tuple_expr(self, e, ctx): self._unsupported('TupleExpr')
    def _visit_list_expr(self, e, ctx): self._unsupported('ListExpr')
    def _visit_list_comp(self, e, ctx): self._unsupported('ListComp')
    def _visit_list_ref(self, e, ctx): self._unsupported('ListRef')
    def _visit_list_slice(self, e, ctx): self._unsupported('ListSlice')
    def _visit_list_set(self, e, ctx): self._unsupported('ListSet')
    def _visit_if_expr(self, e, ctx): self._unsupported('IfExpr')
    def _visit_indexed_assign(self, stmt, ctx): self._unsupported('IndexedAssign')

    def _visit_if1(self, stmt: If1Stmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'if ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_if(self, stmt: IfStmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'if ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.ift, ctx)
        self.writer.dedent()
        self.writer.add_line('} else {')
        self.writer.indent()
        self._visit_block(stmt.iff, ctx)
        self.writer.dedent()
        self.writer.add_line('}')
    def _visit_while(self, stmt: WhileStmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'while ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')
    def _visit_for(self, stmt: ForStmt, ctx):
        if not isinstance(stmt.target, NamedId):
            raise Cpp2EmitError(
                'tuple-binding for-loop targets are not yet supported'
            )
        target_def = self.def_use.find_def_from_site(stmt.target, stmt)
        target = self.storage.def_to_name[target_def]
        # Fold the type into the for header iff the counter is a
        # single-writer class (the common case).  Otherwise the counter
        # was hoisted at the function top and we just reassign here.
        if target_def in self.storage.declare_at_assign:
            storage = self.storage.class_storage[self.storage.def_class[target_def]]
            decl = f'{storage.format()} {target}'
        else:
            decl = target
        match stmt.iterable:
            case Range1():
                stop = self._visit_expr(stmt.iterable.arg, ctx)
                header = f'for ({decl} = 0; {target} < {stop}; ++{target})'
            case Range2():
                start = self._visit_expr(stmt.iterable.first, ctx)
                stop = self._visit_expr(stmt.iterable.second, ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; ++{target})'
                )
            case Range3():
                start = self._visit_expr(stmt.iterable.args[0], ctx)
                stop = self._visit_expr(stmt.iterable.args[1], ctx)
                step = self._visit_expr(stmt.iterable.args[2], ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; {target} += {step})'
                )
            case _:
                raise Cpp2EmitError(
                    'cpp2 Phase 3 only supports for-loops over '
                    f'range(...); got {type(stmt.iterable).__name__}'
                )
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')
    def _visit_assert(self, stmt, ctx): self._unsupported('AssertStmt')
    def _visit_effect(self, stmt, ctx): self._unsupported('EffectStmt')
    def _visit_pass(self, stmt, ctx): self._unsupported('PassStmt')
