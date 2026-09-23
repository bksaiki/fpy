"""Unrolling proven-length sequences into one value per element."""

from ..analysis import ArraySizeAnalysis, ArraySizeInfer, DefineUse, SyntaxCheck
from ..analysis.array_size import ListSize
from ..ast import *
from ..utils import Gensym

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


def _int_of(e: Expr) -> int | None:
    """*e* as a compile-time integer, or `None`."""
    if isinstance(e, Integer):
        return e.val
    if isinstance(e, RationalVal):
        v = e.as_rational()
        return v.numerator if v.denominator == 1 else None
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
        self.lazy = False
        """Whether the expression being visited is only conditionally
        evaluated.  Hoisting out of an `IfExpr` arm would make it
        unconditional, which is the hazard `SimplifyIf` refuses over."""

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
                lo, step = _int_of(e.first), _int_of(e.third)
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

    def _visit_block(self, block: StmtBlock, ctx):
        outer, stmts = self.pending, []
        for stmt in block.stmts:
            self.pending = []
            s, ctx = self._visit_statement(stmt, ctx)
            stmts.extend(self.pending)
            stmts.append(s)
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
