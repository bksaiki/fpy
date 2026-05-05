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

from ...ast.fpyast import (
    Abs, Add, Assign, BinaryOp, Decnum, Digits, Div,
    Expr, FuncDef, Hexnum, Integer, Mul, NamedId, Neg, Rational,
    ReturnStmt, Stmt, StmtBlock, Sub, UnaryOp, Var,
    ContextStmt, UnderscoreId,
)
from ...ast.visitor import Visitor

from .storage import StorageSelectionError
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

    def __init__(
        self,
        ast: FuncDef,
        def_storage,  # dict[Definition, CppType]
        def_use,
        format_info,
    ):
        self.ast = ast
        self.def_storage = def_storage
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

    def _storage_for_var_use(self, var: Var) -> CppType:
        d = self.def_use.find_def_from_use(var)
        return self.def_storage[d]

    def _storage_for_arg(self, arg) -> CppType:
        d = self.def_use.find_def_from_site(arg.name, arg)
        return self.def_storage[d]

    def _storage_for_assign_target(self, name: NamedId, site) -> CppType:
        d = self.def_use.find_def_from_site(name, site)
        return self.def_storage[d]

    # ------------------------------------------------------------------
    # Function emission

    def _visit_function(self, func: FuncDef, ctx):
        # Determine return type from the return statement's expression
        # bound (the slice assumes a single return, possibly nested in a
        # ``with`` block).
        ret_ty = self._infer_return_storage(func)
        ret_str = ret_ty.format() if ret_ty is not None else 'void'

        # Emit arg list.  Argument defs are declared in the signature;
        # they don't get hoisted with other locals.
        arg_strs: list[str] = []
        arg_defs: set = set()
        for arg in func.args:
            if not isinstance(arg.name, NamedId):
                raise Cpp2EmitError(f'unsupported arg pattern: {arg.name!r}')
            storage = self._storage_for_arg(arg)
            arg_strs.append(f'{storage.format()} {arg.name}')
            arg_defs.add(self.def_use.find_def_from_site(arg.name, arg))
        sig = f'{ret_str} {func.name}({", ".join(arg_strs)})'

        self.writer.add_line(sig + ' {')
        self.writer.indent()
        # Hoist all non-arg, non-free local declarations to the top of
        # the function body.  Each source name gets one C++ variable
        # whose type is the storage aggregated across all SSA defs.
        # Subsequent assignments are plain reassignments.  This makes
        # the phi-merge case in the upcoming control-flow phases trivial:
        # both branches just assign into the same C++ variable.
        free_defs = {
            self.def_use.find_def_from_site(v, func) for v in func.free_vars
        }
        self._emit_local_declarations(arg_defs | free_defs)
        self._visit_block(func.body, None)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_local_declarations(self, skip_defs: set):
        """
        Pre-walk all definitions and declare one zero-initialised C++
        variable per source name, using the storage aggregated across
        SSA defs.  Skips the *skip_defs* set (function arguments and
        free variables).
        """
        # Group def → storage by source name, taking the storage that's
        # the same across all defs of that name (storage aggregation
        # already happened in the pipeline).
        by_name: dict[str, CppType] = {}
        for d, storage in self.def_storage.items():
            if d in skip_defs:
                continue
            name = str(d.name)
            existing = by_name.get(name)
            if existing is None:
                by_name[name] = storage
            else:
                assert existing == storage, (
                    f'inconsistent aggregated storage for `{name}`: '
                    f'{existing!r} vs {storage!r}'
                )
        for name in sorted(by_name):
            storage = by_name[name]
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
            self._visit_statement(stmt, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        if not isinstance(stmt.target, NamedId):
            raise Cpp2EmitError(
                f'unsupported assignment target {stmt.target!r}; '
                'tuple unpacking and underscore targets land in a later phase'
            )
        rhs = self._visit_expr(stmt.expr, ctx)
        # Locals are declared at the top of the function body via
        # ``_emit_local_declarations``, so each Assign is a plain
        # reassignment.  This means every SSA def of the same source
        # name shares one C++ variable, and phi merges fall out for free.
        self.writer.add_line(f'{stmt.target} = {rhs};')

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
        # Storage type is implicit — the variable was declared at its
        # def site.  The use just refers to its name.
        if not isinstance(e.name, NamedId):
            raise Cpp2EmitError(f'non-named variable use: {e.name!r}')
        return str(e.name)

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

    def _visit_bool(self, e, ctx): self._unsupported('BoolVal')
    def _visit_foreign(self, e, ctx): self._unsupported('ForeignVal')
    def _visit_attribute(self, e, ctx): self._unsupported('Attribute')
    def _visit_nullaryop(self, e, ctx): self._unsupported('NullaryOp')
    def _visit_ternaryop(self, e, ctx): self._unsupported('TernaryOp')
    def _visit_naryop(self, e, ctx): self._unsupported('NaryOp')
    def _visit_round(self, e, ctx): self._unsupported('Round / RoundExact')
    def _visit_round_at(self, e, ctx): self._unsupported('RoundAt')
    def _visit_call(self, e, ctx): self._unsupported('Call')
    def _visit_compare(self, e, ctx): self._unsupported('Compare')
    def _visit_tuple_expr(self, e, ctx): self._unsupported('TupleExpr')
    def _visit_list_expr(self, e, ctx): self._unsupported('ListExpr')
    def _visit_list_comp(self, e, ctx): self._unsupported('ListComp')
    def _visit_list_ref(self, e, ctx): self._unsupported('ListRef')
    def _visit_list_slice(self, e, ctx): self._unsupported('ListSlice')
    def _visit_list_set(self, e, ctx): self._unsupported('ListSet')
    def _visit_if_expr(self, e, ctx): self._unsupported('IfExpr')
    def _visit_indexed_assign(self, stmt, ctx): self._unsupported('IndexedAssign')
    def _visit_if1(self, stmt, ctx): self._unsupported('If1Stmt')
    def _visit_if(self, stmt, ctx): self._unsupported('IfStmt')
    def _visit_while(self, stmt, ctx): self._unsupported('WhileStmt')
    def _visit_for(self, stmt, ctx): self._unsupported('ForStmt')
    def _visit_assert(self, stmt, ctx): self._unsupported('AssertStmt')
    def _visit_effect(self, stmt, ctx): self._unsupported('EffectStmt')
    def _visit_pass(self, stmt, ctx): self._unsupported('PassStmt')
