"""
Shared machinery for the loop transforms
(:class:`fpy2.transform.SplitLoop`, :class:`fpy2.transform.ForUnroll`).
"""

from ..analysis import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ListSize,
    concrete_size,
)
from ..ast.fpyast import (
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    Id,
    Location,
    Stmt,
    StmtBlock,
    TupleBinding,
    UnderscoreId,
)
from ..ast.visitor import DefaultTransformVisitor
from ..number import INTEGER


def infer_array_size(func: FuncDef) -> ArraySizeAnalysis | None:
    """Run the array-size analysis as an *auxiliary* input: a failure
    only disables a static optimization, so it never breaks the
    transformation."""
    try:
        return ArraySizeInfer.analyze(func)
    except Exception:  # noqa: BLE001 -- auxiliary analysis; failure only disables an optimization
        return None


def static_size(array_size: ArraySizeAnalysis | None, iterable: Expr) -> int | None:
    """The statically-known length of *iterable* (the original AST node,
    which is what the analysis indexes), or ``None`` if the analysis
    could not pin it down."""
    if array_size is None:
        return None
    bound = array_size.by_expr.get(iterable)
    if isinstance(bound, ListSize):
        return concrete_size(bound.size)
    return None


def integer_ctx(stmts: list[Stmt], loc: Location | None) -> ContextStmt:
    """A ``with fp.INTEGER:`` block: the exact integer context under
    which a loop transform's synthesized loop-control and index
    arithmetic must be evaluated (see the rounding-context-safety
    section of the transform's module docstring)."""
    return ContextStmt(UnderscoreId(), ForeignVal(INTEGER, None), StmtBlock(stmts), loc)


def clone_block(block: StmtBlock) -> StmtBlock:
    """A structurally-fresh copy of *block*, so each emitted copy of a
    loop body occupies distinct AST nodes (a plain transform visit
    rebuilds every node)."""
    block, _ = DefaultTransformVisitor()._visit_block(block, None)
    return block


def copy_target(target: Id | TupleBinding) -> Id | TupleBinding:
    """A fresh copy of a loop target with the *same* names.  ``Id``s are
    value-like and shared verbatim; a ``TupleBinding`` is rebuilt so no
    node is shared between the copies it appears in."""
    match target:
        case Id():
            return target
        case TupleBinding():
            return TupleBinding([copy_target(e) for e in target.elts], target.loc)
        case _:
            raise RuntimeError(f'Unexpected target {target}')
