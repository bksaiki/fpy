"""Transformation pass to give a function a single trailing return."""

from ..analysis import DefineUse, Reachability, SyntaxCheck
from ..ast import *
from ..utils import Gensym
from .error import TransformDeclined
from .utils import clone_block

_MAX_DUPLICATION = 8
"""How deep returning `if`s may nest before the copied continuation is refused.

Each level where both arms fall through doubles the tail; the deepest nest in
`examples/mmasim` is 3.
"""


def _has_return(stmts: list[Stmt]) -> bool:
    return bool(Reachability.analyze(StmtBlock(stmts)).ret_stmts)


def _falls_through(stmts: list[Stmt]) -> bool:
    return Reachability.analyze(StmtBlock(stmts)).has_fallthrough


def _sink(
    result: NamedId, stmts: list[Stmt], cont: list[Stmt], depth: int = 0,
) -> list[Stmt]:
    """*stmts* then *cont*, with every `return` rewritten to an assignment.

    `cont` is placed in each arm that falls through: where one does that moves
    it, where both do it is copied.  FPy requires the result be assigned on
    every path, and no single place is reachable from exactly the non-returning
    paths -- which is what the `return` was doing -- so each carries its own.
    """
    for i, stmt in enumerate(stmts):
        rest = list(stmts[i + 1:]) + cont
        match stmt:
            case ReturnStmt():
                # anything after a return is unreachable
                return list(stmts[:i]) + [
                    Assign(result, None, stmt.expr, stmt.loc)
                ]
            case IfStmt() | If1Stmt():
                if isinstance(stmt, IfStmt):
                    ift, iff = list(stmt.ift.stmts), list(stmt.iff.stmts)
                else:
                    ift, iff = list(stmt.body.stmts), []
                if not _has_return(ift) and not _has_return(iff):
                    continue
                ift_out, iff_out = _falls_through(ift), _falls_through(iff)
                iff_cont = rest
                if ift_out and iff_out:
                    depth += 1
                    if depth > _MAX_DUPLICATION:
                        raise TransformDeclined(
                            f'returning `if`s nested more than '
                            f'{_MAX_DUPLICATION} deep'
                        )
                    # the arms must not share nodes: analyses key on identity
                    iff_cont = list(clone_block(StmtBlock(rest)).stmts)
                return list(stmts[:i]) + [IfStmt(
                    stmt.cond,
                    StmtBlock(_sink(result, ift, rest if ift_out else [], depth)),
                    StmtBlock(_sink(result, iff, iff_cont if iff_out else [], depth)),
                    stmt.loc,
                )]
            case ContextStmt():
                body = list(stmt.body.stmts)
                if _falls_through(body):
                    continue
                # the assignment stays inside: its right-hand side rounds
                # under this context
                return list(stmts[:i]) + [ContextStmt(
                    stmt.target, stmt.ctx,
                    StmtBlock(_sink(result, body, [], depth)), stmt.loc,
                )]
    return list(stmts) + _sink(result, cont, [], depth)


class SingleExit:
    """Rewrites a function to a single trailing `return`::

        if c:               if c:
            return a   ⇝        r = a
        S ...               else:
        return b                S ...
                                r = b
                            return r

    Removes the shape `FPCoreCompiler`, `FuncInline` and `SimplifyIf` all
    decline; :func:`fpy2.strategies.single_exit` says why, and what is refused.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        if len(Reachability.analyze(func).ret_stmts) <= 1:
            return func

        result = Gensym(reserved=DefineUse.analyze(func).names()).fresh('r')
        stmts = _sink(result, list(func.body.stmts), [])
        stmts.append(ReturnStmt(Var(result, None), None))
        ast = FuncDef(func.name, func.args, StmtBlock(stmts), func.meta,
                      loc=func.loc)

        left = len(Reachability.analyze(ast).ret_stmts)
        if left != 1:
            raise TransformDeclined(
                f'cannot give `{func.name}` a single exit: {left} returns '
                'remain -- a `return` inside a loop (unroll it first), or '
                'under a `with` that only sometimes returns'
            )
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
