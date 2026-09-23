"""Unrolling proven-length sequences into one value per element."""

from ..analysis import ArraySizeAnalysis, ArraySizeInfer, DefineUse, SyntaxCheck
from ..analysis.array_size import ListSize
from ..ast import *
from ..utils import Gensym
from .utils import clone_block

__all__ = ['Scalarize']

_DEFAULT_CAP = 256
"""How many elements a sequence may have and still unroll.

The widest real MMA instruction is 64 (`nv.blackwell.nvfp4`), and `L + 1`
where an accumulator is appended, so this is four times the widest thing in
the corpus and still refuses an accidental 4096.  Over the cap a sequence is
*left alone*, not refused: a long comprehension compiles perfectly well as a
loop over a tile, and rejecting it would lose a program the backend handles.
"""


_MAX_ROUNDS = 8
"""How deep nesting may go.  Each round takes one level, and the deepest in
`examples/mmasim` is 2."""


def _clone(e: Expr) -> Expr:
    """A substituted expression cannot be shared between occurrences: the
    analyses key on node identity."""
    return DefaultTransformVisitor()._visit_expr(e, None)


class _Subst(DefaultTransformVisitor):
    """Replaces names by expressions, one fresh copy per occurrence."""

    def __init__(self, env: dict[NamedId, Expr]):
        super().__init__()
        self.env = env

    def _visit_var(self, e: Var, ctx):
        sub = self.env.get(e.name)
        return Var(e.name, e.loc) if sub is None else _clone(sub)

    def apply(self, e: Expr) -> Expr:
        return self._visit_expr(e, None)


def _int_of(e: Expr, env: dict[NamedId, int] | None = None) -> int | None:
    """*e* as a compile-time integer, or `None`."""
    if isinstance(e, Integer):
        return e.val
    if isinstance(e, RationalVal):
        v = e.as_rational()
        return v.numerator if v.denominator == 1 else None
    if env is not None:
        if isinstance(e, Var):
            return env.get(e.name)
        if isinstance(e, Add):
            a, b = (_int_of(x, env) for x in e.args)
            return None if a is None or b is None else a + b
        if isinstance(e, Sub):
            a, b = (_int_of(x, env) for x in e.args)
            return None if a is None or b is None else a - b
        if isinstance(e, Mul):
            a, b = (_int_of(x, env) for x in e.args)
            return None if a is None or b is None else a * b
    return None


class _Scalarize(DefaultTransformVisitor):
    """Unrolls each proven-length comprehension into its elements.

    The comprehension is replaced by a **list literal of the element names**
    rather than deleted, so every existing use keeps working -- `len`, a
    subscript, a fold, a call taking the whole list. What moves is where the
    element expressions are *evaluated*: into statement positions, where
    `FuncInline` can reach a call inside one.
    """

    def __init__(self, func: FuncDef, sizes: ArraySizeAnalysis, cap: int):
        super().__init__()
        self.sizes = sizes
        self.cap = cap
        self.gensym = Gensym(reserved=DefineUse.analyze(func).names())
        self.pending: list[Stmt] = []
        self.changed = False
        self.value_lists: dict[NamedId, list[Expr]] = {}
        """Names bound to a list of *values* -- a literal, with nothing in
        memory behind it.

        A loop over one has no runtime iteration to perform: there is no
        object to step through, only that many values.  A loop over something
        in *memory* is a different thing and stays a loop, which is why this
        tracks provenance rather than just a proven length.
        """
        self.consts: dict[NamedId, int] = {}
        """Names bound to an integer constant, so an index like `n + i`
        resolves.  `Simplify` would propagate these, but it runs *after* the
        normal form, and `optimize=False` skips it entirely."""
        self.lazy = False
        """Whether the expression being visited is only conditionally
        evaluated.  Hoisting out of an `IfExpr` arm would make it
        unconditional, which is the hazard `SimplifyIf` refuses over."""

    def _const(self, e: Expr) -> int | None:
        """*e* as a compile-time integer.

        Wider than :func:`_int_of` by `len(xs)`, which the size analysis
        answers.  Inlining binds a callee's `n = len(xs)` to a fresh name, so
        by the time a fill group is examined the length is a `Len` rather
        than the literal `ConstFold` would have produced.
        """
        if isinstance(e, Len):
            return self._size(e.arg)
        if isinstance(e, Var) and e.name in self.consts:
            return self.consts[e.name]
        if isinstance(e, (Add, Sub, Mul)) and len(e.args) == 2:
            a, b = (self._const(x) for x in e.args)
            if a is None or b is None:
                return None
            return a + b if isinstance(e, Add) else (
                a - b if isinstance(e, Sub) else a * b)
        return _int_of(e)

    def _size(self, e: Expr) -> int | None:
        bound = self.sizes.by_expr.get(e)
        if isinstance(bound, ListSize) and isinstance(bound.size, int):
            return bound.size
        return None

    def _elements(self, e: Expr) -> list[Expr] | None:
        """*e* as one expression per element, or `None`."""
        n = self._size(e)
        if n is None or n > self.cap:
            return None
        match e:
            case ListExpr():
                return list(e.elts)
            case Range1():
                return [Integer(i, e.loc) for i in range(n)]
            case Range3():
                lo, step = self._const(e.first), self._const(e.third)
                if lo is None or step is None:
                    return None
                return [Integer(lo + i * step, e.loc) for i in range(n)]
            case Var():
                # a named sequence: its elements are ordinary subscripts, and
                # the backend spells those however it holds the sequence
                return [ListRef(Var(e.name, e.loc), Integer(i, e.loc), e.loc)
                        for i in range(n)]
            case ListComp():
                return self._comp_elements(e, n)
        return None

    def _comp_elements(self, e: ListComp, n: int) -> list[Expr] | None:
        if len(e.iterables) != len(e.targets):
            return None
        srcs = []
        for it in e.iterables:
            elts = self._elements(it)
            if elts is None:
                return None
            srcs.append(elts)
        out: list[Expr] = []
        for i in range(n):
            env: dict[NamedId, Expr] = {}
            for target, src in zip(e.targets, srcs):
                if not isinstance(target, NamedId) or i >= len(src):
                    return None
                env[target] = src[i]
            out.append(_Subst(env).apply(_clone(e.elt)))
        return out

    def _visit_list_comp(self, e: ListComp, ctx):
        # the size analysis keys on node *identity*, so ask before rebuilding
        elems = None if self.lazy else self._elements(e)
        if elems is None:
            return super()._visit_list_comp(e, ctx)
        names: list[NamedId] = []
        for elt in elems:
            # visiting first, so anything it hoists lands above this element
            value = self._visit_expr(elt, ctx)
            name = self.gensym.fresh('e')
            self.pending.append(Assign(name, None, value, e.loc))
            names.append(name)
        self.changed = True
        return ListExpr([Var(n, e.loc) for n in names], e.loc)

    def _visit_if_expr(self, e: IfExpr, ctx):
        cond = self._visit_expr(e.cond, ctx)
        prev, self.lazy = self.lazy, True
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        self.lazy = prev
        return IfExpr(cond, ift, iff, e.loc)

    def _reads(self, e: Expr, name: NamedId) -> bool:
        """Whether *e* mentions *name* at all."""
        found = False

        class _R(DefaultVisitor):
            def _visit_var(self, v: Var, _c):
                nonlocal found
                if v.name == name:
                    found = True

        _R()._visit_expr(e, None)
        return found

    def _writes(self, stmt: Stmt, zs: NamedId) -> list[tuple[int, Expr]] | None:
        """*stmt* as the (index, value) pairs it stores into *zs*.

        An empty list means it stores nothing there; `None` means it is not
        a store this can account for, and the group has to stop.
        """
        match stmt:
            case IndexedAssign() if stmt.var == zs:
                if len(stmt.indices) != 1:
                    return None
                i = self._const(stmt.indices[0])
                if i is None or self._reads(stmt.expr, zs):
                    return None
                return [(i, stmt.expr)]
            case ForStmt():
                body = stmt.body.stmts
                if len(body) != 1 or not isinstance(body[0], IndexedAssign):
                    return None
                inner = body[0]
                if inner.var != zs or len(inner.indices) != 1:
                    return None
                if not isinstance(stmt.target, NamedId):
                    return None
                idxs = self._elements(stmt.iterable)
                if idxs is None:
                    return None
                out: list[tuple[int, Expr]] = []
                for v in idxs:
                    env = {stmt.target: v}
                    at = self._const(
                        _Subst(env).apply(_clone(inner.indices[0])))
                    value = _Subst(env).apply(_clone(inner.expr))
                    if at is None or self._reads(value, zs):
                        return None
                    out.append((at, value))
                return out
            case _:
                # anything else may read `zs`, so the group ends here
                return None if self._touches(stmt, zs) else []
        return None

    def _touches(self, stmt: Stmt, zs: NamedId) -> bool:
        found = False

        class _R(DefaultVisitor):
            def _visit_var(self, v: Var, _c):
                nonlocal found
                if v.name == zs:
                    found = True

        _R()._visit_statement(stmt, None)
        return found

    def _fill_group(
        self, stmts: list[Stmt], j: int,
    ) -> tuple[list[Stmt], int] | None:
        """`zs = empty(n)` and the stores that fill it, as plain values.

        Sound only when all four hold, so each is checked: the length is a
        constant within the cap, every store is at a constant index in range,
        each index is written exactly once, and nothing reads the list before
        the last store.  `join` in `examples/mmasim` satisfies them; a general
        rule cannot assume it.
        """
        alloc = stmts[j]
        if not isinstance(alloc, Assign) or not isinstance(alloc.target, NamedId):
            return None
        if not isinstance(alloc.expr, Empty) or len(alloc.expr.args) != 1:
            return None
        n = self._const(alloc.expr.args[0])
        if n is None or n <= 0 or n > self.cap:
            return None

        zs = alloc.target
        filled: dict[int, Expr] = {}
        k = j + 1
        while k < len(stmts) and len(filled) < n:
            pairs = self._writes(stmts[k], zs)
            if pairs is None or not pairs:
                return None
            for at, value in pairs:
                if not 0 <= at < n or at in filled:
                    return None  # out of range, or written twice
                filled[at] = value
            k += 1
        if len(filled) != n:
            return None

        names = [self.gensym.fresh(str(zs)) for _ in range(n)]
        out: list[Stmt] = [
            Assign(name, None, filled[i], alloc.loc)
            for i, name in enumerate(names)
        ]
        out.append(Assign(
            zs, None,
            ListExpr([Var(nm, alloc.loc) for nm in names], alloc.loc),
            alloc.loc,
        ))
        return out, k - j

    def _unroll_for(self, stmt: ForStmt) -> list[Stmt] | None:
        """A `for` over a list of values, as its iterations.

        Only over *values*: a loop over a range or over something in memory
        has an iteration to perform and stays a loop.  Unrolling those would
        rewrite every kernel that reduces over a tile, which is the shape
        this backend is built around.
        """
        if not isinstance(stmt.target, NamedId):
            return None  # a tuple binding is `ZipElim`'s business
        elts: list[Expr] | None = None
        if isinstance(stmt.iterable, ListExpr):
            elts = list(stmt.iterable.elts)
        elif isinstance(stmt.iterable, Var):
            elts = self.value_lists.get(stmt.iterable.name)
        if elts is None or len(elts) > self.cap:
            return None
        out: list[Stmt] = []
        for elt in elts:
            # cloned per copy: the analyses key on node identity
            out.append(Assign(stmt.target, None, _clone(elt), stmt.loc))
            out.extend(clone_block(stmt.body).stmts)
        return out

    def _visit_block(self, block: StmtBlock, ctx):
        outer, stmts = self.pending, []
        src, j = block.stmts, 0
        while j < len(src):
            here = src[j]
            if isinstance(here, ForStmt):
                unrolled = self._unroll_for(here)
                if unrolled is not None:
                    self.changed = True
                    src = list(src[:j]) + unrolled + list(src[j + 1:])
                    continue
            group = self._fill_group(list(src), j)
            if group is not None:
                rewritten, used = group
                self.changed = True
                src = list(src[:j]) + rewritten + list(src[j + used:])
                continue
            stmt = src[j]
            self.pending = []
            out, ctx = self._visit_statement(stmt, ctx)
            stmts.extend(self.pending)
            stmts.append(out)
            # an integer binding, so a later index like `n + i` resolves
            if isinstance(stmt, Assign) and isinstance(stmt.target, NamedId):
                v = self._const(stmt.expr)
                if v is not None:
                    self.consts[stmt.target] = v
                if isinstance(out, Assign):
                    # a literal, or a copy of one: inlining binds a callee's
                    # parameter to the caller's list by name, so the loop it
                    # came with sees the copy rather than the literal
                    if isinstance(out.expr, ListExpr):
                        self.value_lists[stmt.target] = list(out.expr.elts)
                    elif isinstance(out.expr, Var):
                        src_elts = self.value_lists.get(out.expr.name)
                        if src_elts is not None:
                            self.value_lists[stmt.target] = src_elts
            j += 1
        self.pending = outer
        return StmtBlock(stmts), ctx


class Scalarize:
    """Unrolls every proven-length comprehension into one value per element.

    See :func:`fpy2.strategies.unroll_seqs` for what this is for and what it
    leaves alone.
    """

    @staticmethod
    def apply(func: FuncDef, *, cap: int = _DEFAULT_CAP) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        if cap < 0:
            raise ValueError(f'cap must be non-negative, got {cap}')
        # to a fixpoint: a substituted element is a *clone*, so a
        # comprehension nested inside one has an identity the size analysis
        # has not seen.  Re-analyzing and going again reaches it.
        for _ in range(_MAX_ROUNDS):
            inst = _Scalarize(func, ArraySizeInfer.analyze(func), cap)
            out = inst._visit_function(func, None)
            if not inst.changed:
                break
            func = out
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
