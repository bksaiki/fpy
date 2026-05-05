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
    Abs, Add, Argument, Assign, BinaryOp, BoolVal, Compare, Decnum, Digits, Div,
    Enumerate, Expr, ForStmt, FuncDef, Hexnum, If1Stmt, IfStmt, IndexedAssign,
    Integer, Len, ListComp, ListExpr, ListRef, ListSlice, Mul, NamedId,
    NaryOp, Neg, Range1, Range2, Range3, Rational, ReturnStmt, Stmt, StmtBlock,
    Sub, Sum, TupleBinding, TupleExpr, UnaryOp, Var, ContextStmt, UnderscoreId,
    WhileStmt, Zip,
)
from ...ast.visitor import Visitor
from ...utils.compare import CompareOp

from .storage import StorageSelectionError, choose_storage
from .storage_infer import StorageAnalysis
from .types import CppList, CppTuple, CppType


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
        return '\n'.join(self._lines)


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
        self._tmp_counter = 0

    def _fresh_temp(self) -> str:
        """Allocate a fresh emitter-only temporary identifier.

        Used by visitors that need to emit setup statements alongside
        an expression result (currently :class:`ListComp`).  The
        returned name uses a double-underscore prefix to keep it
        distinct from any identifier the source program could
        introduce.
        """
        self._tmp_counter += 1
        return f'__cpp2_tmp{self._tmp_counter}'

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

    def _storage_for_arg(self, arg: Argument) -> CppType:
        assert isinstance(arg.name, NamedId)
        d = self.def_use.find_def_from_site(arg.name, arg)
        return self.storage.class_storage[self.storage.def_class[d]]

    def _emit_bind(self, name: NamedId, site, rhs: str) -> None:
        """Emit a single ``T name = rhs;`` (declare-on-assign) or
        ``name = rhs;`` (reassign) line for a NamedId target whose
        SSA def is registered at *site*.

        Whether to declare or reassign is decided by the
        :class:`StorageAnalysis`."""
        target_def = self.def_use.find_def_from_site(name, site)
        target_name = self.storage.def_to_name[target_def]
        if target_def in self.storage.declare_at_assign:
            storage = self.storage.class_storage[
                self.storage.def_class[target_def]
            ]
            self.writer.add_line(f'{storage.format()} {target_name} = {rhs};')
        else:
            self.writer.add_line(f'{target_name} = {rhs};')

    def _destructure(
        self,
        binding: TupleBinding,
        src: str,
        site,
    ) -> None:
        """Emit destructuring assigns extracting each element of
        *binding* from the tuple-valued local *src*.  The SSA defs
        for every NamedId in *binding* are registered at *site* (the
        enclosing :class:`Assign` / :class:`ForStmt` / :class:`ListComp`
        node).  Underscore positions are skipped; nested tuple bindings
        recurse via a fresh temp."""
        for i, elt in enumerate(binding.elts):
            if isinstance(elt, UnderscoreId):
                continue
            access = f'std::get<{i}>({src})'
            if isinstance(elt, NamedId):
                self._emit_bind(elt, site, access)
            elif isinstance(elt, TupleBinding):
                sub_tmp = self._fresh_temp()
                self.writer.add_line(f'auto {sub_tmp} = {access};')
                self._destructure(elt, sub_tmp, site)
            else:
                raise Cpp2EmitError(
                    f'unsupported tuple-binding element {elt!r}'
                )

    def _storage_for_expr(self, e: Expr) -> CppType:
        """The C++ storage chosen for an expression's result.

        Falls back to ``Cpp2EmitError`` if format inference produced
        nothing storable (e.g., a symbolic ``REAL_FORMAT``).
        """
        fmt = self.format_info.by_expr.get(e)
        try:
            return choose_storage(fmt)
        except StorageSelectionError as err:
            raise Cpp2EmitError(
                f'cannot pick storage for {type(e).__name__}: {err}'
            ) from err

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
        rhs = self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.target, NamedId):
            # ``StorageInfer`` maps this Assign's SSA def to a C++
            # variable and tells us whether the assign should declare
            # the variable (single-writer class) or just reassign into
            # a hoisted decl (multi-writer class).
            self._emit_bind(stmt.target, stmt, rhs)
        elif isinstance(stmt.target, TupleBinding):
            # ``(a, b) = expr``: bind the rhs to a tuple-valued temp
            # once, then destructure.  Each NamedId in the binding has
            # its own SSA def registered at the Assign statement.
            tmp = self._fresh_temp()
            self.writer.add_line(f'auto {tmp} = {rhs};')
            self._destructure(stmt.target, tmp, stmt)
        else:
            raise Cpp2EmitError(
                f'unsupported assignment target {stmt.target!r}'
            )

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

    # ------------------------------------------------------------------
    # Expression visitors — return a C++ source fragment

    def _visit_var(self, e: Var, ctx) -> str:
        # Resolve the use to its SSA def, then look up the C++
        # identifier ``StorageInfer`` assigned to that def's class.
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
        if isinstance(e, Len):
            # ``len(xs)`` — the result format is INTEGER, which storage
            # selection rounds to a concrete C++ integer type.  Casting
            # ``size()`` (a ``size_t``) keeps the inferred type stable
            # across platforms where ``size_t`` differs from ``int64_t``.
            result_ty = self._storage_for_expr(e)
            return f'static_cast<{result_ty.format()}>({arg}.size())'
        if isinstance(e, Sum):
            # ``sum(xs)`` → ``std::accumulate(begin, end, T(0))`` with
            # ``T`` taken from format inference (so the accumulator's
            # type matches the inferred bound on the result).
            result_ty = self._storage_for_expr(e)
            return (
                f'std::accumulate({arg}.begin(), {arg}.end(), '
                f'static_cast<{result_ty.format()}>(0))'
            )
        if isinstance(e, Enumerate):
            return self._emit_enumerate(e, arg)
        raise Cpp2EmitError(f'unsupported unary op: {type(e).__name__}')

    def _emit_enumerate(self, e: Enumerate, src_str: str) -> str:
        """``enumerate(xs)`` builds a ``std::vector<std::tuple<I, T>>``
        where ``I`` is the index integer type and ``T`` is the source
        element type — both come from format inference on the
        Enumerate node itself.
        """
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)
                and len(result_ty.elt.elts) == 2):
            raise Cpp2EmitError(
                'expected list[(int, T)] for enumerate result, '
                f'got {result_ty!r}'
            )
        idx_ty = result_ty.elt.elts[0]

        src = self._fresh_temp()
        self.writer.add_line(f'auto {src} = {src_str};')
        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result}({src}.size());'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < {src}.size(); ++{i}) {{'
        )
        self.writer.indent()
        self.writer.add_line(
            f'{result}[{i}] = std::make_tuple('
            f'static_cast<{idx_ty.format()}>({i}), {src}[{i}]);'
        )
        self.writer.dedent()
        self.writer.add_line('}')
        return result

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
        raise Cpp2EmitError(f'cpp2 emitter does not handle {kind}')

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
    def _visit_naryop(self, e: NaryOp, ctx) -> str:
        if isinstance(e, Zip):
            return self._emit_zip(e, ctx)
        raise Cpp2EmitError(f'unsupported nary op: {type(e).__name__}')

    def _emit_zip(self, e: Zip, ctx) -> str:
        """``zip(xs1, …, xsN)`` builds a
        ``std::vector<std::tuple<T1, …, TN>>`` whose length matches the
        first iterable.  Each iterable is bound to a temp once to
        evaluate side-effects in source order; the tuple type comes
        from format inference on the Zip node."""
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)):
            raise Cpp2EmitError(
                f'expected list[tuple[...]] for zip result, got {result_ty!r}'
            )

        srcs: list[str] = []
        for arg in e.args:
            arg_str = self._visit_expr(arg, ctx)
            s = self._fresh_temp()
            self.writer.add_line(f'auto {s} = {arg_str};')
            srcs.append(s)

        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result}({srcs[0]}.size());'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < {srcs[0]}.size(); ++{i}) {{'
        )
        self.writer.indent()
        elts = ', '.join(f'{s}[{i}]' for s in srcs)
        self.writer.add_line(f'{result}[{i}] = std::make_tuple({elts});')
        self.writer.dedent()
        self.writer.add_line('}')
        return result
    def _visit_round(self, e, ctx): self._unsupported('Round / RoundExact')
    def _visit_round_at(self, e, ctx): self._unsupported('RoundAt')
    def _visit_call(self, e, ctx): self._unsupported('Call')
    def _visit_tuple_expr(self, e: TupleExpr, ctx) -> str:
        # ``(a, b, c)`` → ``std::make_tuple(a, b, c)``.  Type deduction
        # from the argument types matches what the format-inference
        # pipeline assigns to the tuple slot — no explicit type prefix.
        elts = ', '.join(self._visit_expr(elt, ctx) for elt in e.elts)
        return f'std::make_tuple({elts})'
    def _visit_list_expr(self, e: ListExpr, ctx) -> str:
        # ``[a, b, c]`` → ``std::vector<T>{a, b, c}`` where ``T`` comes
        # from format inference on the list expression.
        elts = ', '.join(self._visit_expr(elt, ctx) for elt in e.elts)
        result_ty = self._storage_for_expr(e)
        return f'{result_ty.format()}{{{elts}}}'

    def _visit_list_comp(self, e: ListComp, ctx) -> str:
        # ``[elt for x1 in iter1 [for x2 in iter2 ...]]``
        #
        # Emitted as a temporary ``std::vector<T> tmp;`` followed by
        # nested for-loops that ``push_back`` the element expression
        # into ``tmp``.  Returns the temp's name as the comprehension's
        # value.  The temp shape mirrors the cpp/ backend; future
        # polish may inline directly into an ``Assign`` target to skip
        # the temp + copy.
        result_ty = self._storage_for_expr(e)
        tmp = self._fresh_temp()
        self.writer.add_line(f'{result_ty.format()} {tmp};')

        for target, iterable in zip(e.targets, e.iterables):
            self._open_comp_loop(target, iterable, e, ctx)

        elt = self._visit_expr(e.elt, ctx)
        self.writer.add_line(f'{tmp}.push_back({elt});')

        for _ in e.targets:
            self.writer.dedent()
            self.writer.add_line('}')

        return tmp

    def _open_comp_loop(
        self,
        target,
        iterable: Expr,
        comp_site: ListComp,
        ctx,
    ) -> None:
        """Emit one ``for`` line for a single comprehension stage,
        leaving the writer indented inside the loop body.

        The target's storage class is determined by ``StorageInfer``
        exactly as for any other AssignDef.  We always declare-on-
        assign in the for header — the comprehension's target lives
        only inside the loop's lexical scope.  The comp target's
        SSA site is the ``ListComp`` node, not the target id itself
        (see ``define_use._visit_list_comp``).
        """
        if isinstance(target, NamedId):
            target_def = self.def_use.find_def_from_site(target, comp_site)
            target_name = self.storage.def_to_name[target_def]
            target_ty = self.storage.class_storage[
                self.storage.def_class[target_def]
            ].format()
            decl = f'{target_ty} {target_name}'
        elif isinstance(target, TupleBinding):
            # ``for (a, b) in xs`` — bind the tuple element to an
            # anonymous temp via ``auto`` and destructure inside the
            # loop body.  Range iterables can't pair with a tuple
            # binding; reject early.
            if isinstance(iterable, (Range1, Range2, Range3)):
                raise Cpp2EmitError(
                    'tuple-binding comprehension target requires a '
                    'non-range iterable'
                )
            tmp = self._fresh_temp()
            iter_str = self._visit_expr(iterable, ctx)
            self.writer.add_line(f'for (auto {tmp} : {iter_str}) {{')
            self.writer.indent()
            self._destructure(target, tmp, comp_site)
            return
        else:
            raise Cpp2EmitError(
                f'unsupported comprehension target {target!r}'
            )

        match iterable:
            case Range1():
                stop = self._visit_expr(iterable.arg, ctx)
                self.writer.add_line(
                    f'for ({decl} = 0; '
                    f'{target_name} < {stop}; ++{target_name}) {{'
                )
            case Range2():
                start = self._visit_expr(iterable.first, ctx)
                stop = self._visit_expr(iterable.second, ctx)
                self.writer.add_line(
                    f'for ({decl} = {start}; '
                    f'{target_name} < {stop}; ++{target_name}) {{'
                )
            case Range3():
                start = self._visit_expr(iterable.args[0], ctx)
                stop = self._visit_expr(iterable.args[1], ctx)
                step = self._visit_expr(iterable.args[2], ctx)
                self.writer.add_line(
                    f'for ({decl} = {start}; '
                    f'{target_name} < {stop}; {target_name} += {step}) {{'
                )
            case _:
                iter_str = self._visit_expr(iterable, ctx)
                self.writer.add_line(f'for ({decl} : {iter_str}) {{')
        self.writer.indent()

    def _visit_list_ref(self, e: ListRef, ctx) -> str:
        # ``xs[i]`` → ``xs[i]``.  C++ ``operator[]`` doesn't bounds-
        # check; FPy's interpreter does.  Strict bounds-checking is
        # deferred to a later phase along with slicing.
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        return f'{value}[{index}]'
    def _visit_list_slice(self, e: ListSlice, ctx) -> str:
        # ``xs[start:stop]`` →
        #   auto __cpp2_tmpN = <xs>;
        #   <result_ty>(__cpp2_tmpN.begin() + start,
        #               __cpp2_tmpN.begin() + stop)
        #
        # Binding the value to a temp avoids re-evaluating ``<xs>``
        # when it isn't a simple lvalue (and matches the interpreter,
        # which evaluates the value exactly once).  Indices are cast to
        # ``size_t`` to match the iterator-arithmetic API.  Strict
        # bounds-checking against the interpreter's behaviour is a TODO
        # (slice-out-of-range, negative-index handling, etc.).
        arr_tmp = self._fresh_temp()
        arr_str = self._visit_expr(e.value, ctx)
        self.writer.add_line(f'auto {arr_tmp} = {arr_str};')

        if e.start is None:
            start = '0'
        else:
            start = f'static_cast<size_t>({self._visit_expr(e.start, ctx)})'
        if e.stop is None:
            stop = f'{arr_tmp}.size()'
        else:
            stop = f'static_cast<size_t>({self._visit_expr(e.stop, ctx)})'

        result_ty = self._storage_for_expr(e)
        return (
            f'{result_ty.format()}('
            f'{arr_tmp}.begin() + {start}, '
            f'{arr_tmp}.begin() + {stop})'
        )
    def _visit_list_set(self, e, ctx):
        # ``ListSet`` is only produced by the ``FuncUpdate`` transform,
        # which the cpp2 pipeline does not run — C++ mutates vector
        # elements directly via ``IndexedAssign``.  Reaching this
        # visitor would be a pipeline bug.
        self._unsupported('ListSet')
    def _visit_if_expr(self, e, ctx): self._unsupported('IfExpr')
    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        # ``xs[i1]…[iN] = e`` is in-place mutation in C++.  The
        # post-mutation SSA def of ``xs`` shares a storage class with
        # its ``prev`` (see ``_is_in_place_assign`` in
        # ``storage_infer``), so the C++ name is the same on both
        # sides — emit a direct subscript-store.
        target_def = self.def_use.find_def_from_site(stmt.var, stmt)
        target_name = self.storage.def_to_name[target_def]
        idx_strs = [self._visit_expr(idx, ctx) for idx in stmt.indices]
        chain = target_name + ''.join(f'[{i}]' for i in idx_strs)
        rhs = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'{chain} = {rhs};')

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
        if isinstance(stmt.target, NamedId):
            self._emit_for_named_target(stmt, ctx)
        elif isinstance(stmt.target, TupleBinding):
            self._emit_for_tuple_target(stmt, ctx)
        else:
            raise Cpp2EmitError(
                f'unsupported for-loop target {stmt.target!r}'
            )

    def _emit_for_named_target(self, stmt: ForStmt, ctx):
        assert isinstance(stmt.target, NamedId)
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
                iter_str = self._visit_expr(stmt.iterable, ctx)
                header = f'for ({decl} : {iter_str})'
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_for_tuple_target(self, stmt: ForStmt, ctx):
        # ``for (a, b) in xs:`` — a tuple-binding target only makes
        # sense for non-``range`` iterables (range produces ints).
        # Iterate via a tuple-typed temp, then destructure into the
        # binding's named SSA defs at the top of the loop body.
        if isinstance(stmt.iterable, (Range1, Range2, Range3)):
            raise Cpp2EmitError(
                'tuple-binding for-loop target requires a non-range iterable'
            )
        iter_str = self._visit_expr(stmt.iterable, ctx)
        tmp = self._fresh_temp()
        self.writer.add_line(f'for (auto {tmp} : {iter_str}) {{')
        self.writer.indent()
        assert isinstance(stmt.target, TupleBinding)
        self._destructure(stmt.target, tmp, stmt)
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')
    def _visit_assert(self, stmt, ctx): self._unsupported('AssertStmt')
    def _visit_effect(self, stmt, ctx): self._unsupported('EffectStmt')
    def _visit_pass(self, stmt, ctx): self._unsupported('PassStmt')
