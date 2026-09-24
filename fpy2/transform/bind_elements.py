"""One name per constant-index element read."""

from ..analysis import (
    Alias,
    ArraySizeInfer,
    AssignDef,
    DefineUse,
    Definition,
    SyntaxCheck,
    TypeInfer,
)
from ..analysis.array_size import ListSize
from ..ast import *
from ..types import ListType, TupleType
from ..utils import Gensym

__all__ = ['BindElements']


class _Reads(DefaultVisitor):
    """Every `xs[k]` of a scalar at a constant `k` in range, by `(def, k)`."""

    def __init__(self, func: FuncDef):
        super().__init__()
        self.def_use = DefineUse.analyze(func)
        self.types = TypeInfer.check(func, def_use=self.def_use)
        self.sizes = ArraySizeInfer.analyze(func)
        self.reads: dict[tuple[Definition, int], list[ListRef]] = {}

    def _visit_list_ref(self, e: ListRef, ctx):
        super()._visit_list_ref(e, ctx)
        if not isinstance(e.value, Var) or not isinstance(e.index, Integer):
            return
        if isinstance(self.types.by_expr.get(e), ListType | TupleType):
            return
        size = self.sizes.by_expr.get(e.value)
        k = e.index.val
        if not isinstance(size, ListSize) or not isinstance(size.size, int):
            return
        if 0 <= k < size.size:
            d = self.def_use.find_def_from_use(e.value)
            self.reads.setdefault((d, k), []).append(e)


class _Bind(DefaultTransformVisitor):
    def __init__(self, func: FuncDef, names: dict[int, NamedId],
                 after: dict[int, list[Stmt]], first: list[Stmt]):
        super().__init__()
        self.func = func
        self.names = names
        self.after = after
        self.first = first

    def _visit_list_ref(self, e: ListRef, ctx):
        name = self.names.get(id(e))
        return Var(name, e.loc) if name is not None else super()._visit_list_ref(e, ctx)

    def _visit_block(self, block: StmtBlock, ctx):
        stmts: list[Stmt] = list(self.first) if block is self.func.body else []
        for stmt in block.stmts:
            out, ctx = self._visit_statement(stmt, ctx)
            stmts.append(out)
            stmts.extend(self.after.get(id(stmt), ()))
        return StmtBlock(stmts), ctx


class BindElements:
    """Binds each constant-index read of one list definition to one name.

    Two reads of `xs[0]` become two uses of `x = xs[0]`, bound right after
    `xs` is, so an analysis that knows a variable learns of one element
    everywhere it is read.  Only where the element cannot change: the list
    is never stored into, through any name, nor handed to a call.  Only a
    list bound by an assignment or a parameter, so the binding has a place.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected a 'FuncDef', got {func}")
        reads = _Reads(func)
        reads._visit_function(func, None)
        shared = {key: es for key, es in reads.reads.items() if len(es) > 1}
        if not shared:
            return func

        alias = Alias.analyze(func, def_use=reads.def_use, type_info=reads.types)
        gensym = Gensym(reserved=reads.def_use.names())
        names: dict[int, NamedId] = {}
        after: dict[int, list[Stmt]] = {}
        first: list[Stmt] = []
        for (d, k), es in sorted(shared.items(), key=lambda kv: kv[0][1]):
            if not isinstance(d, AssignDef) or alias.may_change(d):
                continue
            match d.site:
                case Assign():
                    at = after.setdefault(id(d.site), [])
                case Argument():
                    at = first
                case _:
                    continue
            name = gensym.fresh(f'{d.name.base}_{k}_')
            loc = es[0].loc
            at.append(Assign(name, None, ListRef(Var(d.name, loc), Integer(k, loc), loc), loc))
            for e in es:
                names[id(e)] = name
        if not names:
            return func

        out = _Bind(func, names, after, first)._visit_function(func, None)
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
