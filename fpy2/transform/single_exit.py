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
    *, gensym: Gensym | None = None, flag: NamedId | None = None,
) -> list[Stmt]:
    """*stmts* then *cont*, with every `return` rewritten to an assignment.

    `cont` is placed in each arm that falls through: where one does that moves
    it, where both do it is copied.  FPy requires the result be assigned on
    every path, and no single place is reachable from exactly the non-returning
    paths -- which is what the `return` was doing -- so each carries its own.

    *flag* is set inside a loop body, where a `return` cannot move the
    continuation -- it has to be recorded and tested after the loop instead.
    """
    for i, stmt in enumerate(stmts):
        rest = list(stmts[i + 1:]) + cont
        match stmt:
            case ReturnStmt():
                # anything after a return is unreachable
                assign = Assign(result, None, stmt.expr, stmt.loc)
                if flag is None:
                    return list(stmts[:i]) + [assign]
                return list(stmts[:i]) + [
                    Assign(flag, None, BoolVal(True, stmt.loc), stmt.loc),
                    assign,
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
                    StmtBlock(_sink(result, ift, rest if ift_out else [],
                                    depth, gensym=gensym, flag=flag)),
                    StmtBlock(_sink(result, iff, iff_cont if iff_out else [],
                                    depth, gensym=gensym, flag=flag)),
                    stmt.loc,
                )]
            case ForStmt():
                body = list(stmt.body.stmts)
                if not _has_return(body) or gensym is None:
                    continue
                # a `return` here cannot move the continuation the way one in
                # an `if` can: the statements after the loop are reachable
                # from the loop's *own* exit as well.  So it is recorded in a
                # flag and the continuation is tested against it.
                #
                # The guard wraps the **whole** body, not just the return:
                # these loops carry state across iterations, and guarding only
                # the return would keep mutating past the exit.
                done = gensym.fresh('done')
                # the result is loop-carried and assigned conditionally, so
                # it needs an incoming value the way a phi node does -- FPy
                # requires definite assignment and cannot see that the flag
                # makes every path cover it.  The seed is dead: `done` being
                # set implies the loop assigned it, and the tail assigns it
                # otherwise.
                seed = Assign(result, None, Decnum('0', stmt.loc), stmt.loc)
                inner = _sink(result, body, [], depth,
                              gensym=gensym, flag=done)
                tail = _sink(result, rest, [], depth,
                             gensym=gensym, flag=flag)
                unset = Not(Var(done, stmt.loc), stmt.loc)
                return list(stmts[:i]) + [
                    seed,
                    Assign(done, None, BoolVal(False, stmt.loc), stmt.loc),
                    ForStmt(
                        stmt.target, stmt.iterable,
                        StmtBlock([If1Stmt(
                            unset, StmtBlock(inner), stmt.loc,
                        )]),
                        stmt.loc,
                    ),
                    If1Stmt(
                        Not(Var(done, stmt.loc), stmt.loc),
                        StmtBlock(tail), stmt.loc,
                    ),
                ]
            case ContextStmt():
                body = list(stmt.body.stmts)
                if _falls_through(body):
                    continue
                # the assignment stays inside: its right-hand side rounds
                # under this context
                return list(stmts[:i]) + [ContextStmt(
                    stmt.target, stmt.ctx,
                    StmtBlock(_sink(result, body, [], depth,
                                    gensym=gensym, flag=flag)),
                    stmt.loc,
                )]
    if not cont:
        # nothing left to thread.  A function body always ends in a `return`,
        # so this only comes up for a loop body, which falls through.
        return list(stmts)
    return list(stmts) + _sink(result, cont, [], depth,
                               gensym=gensym, flag=flag)


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

        gensym = Gensym(reserved=DefineUse.analyze(func).names())
        result = gensym.fresh('r')
        stmts = _sink(result, list(func.body.stmts), [], gensym=gensym)
        stmts.append(ReturnStmt(Var(result, None), None))
        ast = FuncDef(func.name, func.args, StmtBlock(stmts), func.meta,
                      loc=func.loc)

        left = len(Reachability.analyze(ast).ret_stmts)
        if left != 1:
            raise TransformDeclined(
                f'cannot give `{func.name}` a single exit: {left} returns '
                'remain -- a `return` inside a `while` (whose early exit may '
                'be its only one), or under a `with` that only sometimes '
                'returns'
            )
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
