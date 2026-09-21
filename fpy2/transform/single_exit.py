"""Transformation pass to give a function a single trailing return."""

from ..analysis import DefineUse, Reachability, SyntaxCheck
from ..ast import *
from ..utils import Gensym
from .error import TransformDeclined
from .utils import clone_block


def _returns_in_loop(func: FuncDef) -> str | None:
    """Why *func* cannot be normalized, or `None`.

    A `return` under a loop would need a flag suppressing the rest of the body
    and every later iteration -- FPy has no `break`.  Refused instead: such a
    loop unrolls away where its trip count is known, which is the only shape
    the corpus has.  See `docs/todos/single-exit.md`.
    """
    found: str | None = None

    class _V(DefaultVisitor):
        depth = 0

        def _visit_return(self, stmt: ReturnStmt, ctx):
            nonlocal found
            if self.depth and found is None:
                found = 'a `return` inside a loop; unroll the loop first'

        def _visit_for(self, stmt: ForStmt, ctx):
            self.depth += 1
            super()._visit_for(stmt, ctx)
            self.depth -= 1

        def _visit_while(self, stmt: WhileStmt, ctx):
            self.depth += 1
            super()._visit_while(stmt, ctx)
            self.depth -= 1

    _V()._visit_function(func, None)
    return found


_MAX_DUPLICATION = 8
"""How deep returning `if`s may nest before the copied continuation is refused.

Each level where both arms fall through doubles the tail.  The deepest nest in
`examples/mmasim` is 3.
"""


def _has_return(stmts: list[Stmt]) -> bool:
    """Does *stmts* contain a `return` at any depth?"""
    return bool(Reachability.analyze(StmtBlock(stmts)).ret_stmts)


def _falls_through(stmts: list[Stmt]) -> bool:
    """Is there a path through *stmts* that does not return?"""
    return Reachability.analyze(StmtBlock(stmts)).has_fallthrough


class _SingleExitInstance:
    """Single-use instance of the SingleExit pass."""

    def __init__(self, func: FuncDef, result: NamedId):
        self.func = func
        self.result = result

    def apply(self) -> FuncDef:
        stmts = self._sink(self.func.body.stmts, [])
        stmts.append(ReturnStmt(Var(self.result, None), None))
        return FuncDef(
            self.func.name, self.func.args, StmtBlock(stmts),
            self.func._meta, loc=self.func.loc,
        )

    def _sink(self, stmts: list[Stmt], cont: list[Stmt], depth: int = 0) -> list[Stmt]:
        """*stmts* then *cont*, with every `return` rewritten to an assignment.

        `cont` is placed in each arm that falls through.  Where only one does,
        that moves it; where both do, it is copied, because there is no single
        place reachable from exactly the non-returning paths -- which is what
        the `return` was doing.  FPy requires the result be assigned on every
        path, so every surviving path carries its own continuation.
        """
        for i, stmt in enumerate(stmts):
            rest = list(stmts[i + 1:]) + cont
            match stmt:
                case ReturnStmt():
                    # anything after a return is unreachable
                    return list(stmts[:i]) + [
                        Assign(self.result, None, stmt.expr, stmt.loc)
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
                                f'nested returns {depth} deep would copy the '
                                f'continuation past the limit of '
                                f'{_MAX_DUPLICATION}'
                            )
                        # the arms must not share nodes: the analyses key by
                        # node identity
                        iff_cont = list(clone_block(StmtBlock(rest)).stmts)
                    return list(stmts[:i]) + [IfStmt(
                        stmt.cond,
                        StmtBlock(self._sink(ift, rest if ift_out else [], depth)),
                        StmtBlock(self._sink(iff, iff_cont if iff_out else [], depth)),
                        stmt.loc,
                    )]
                case ContextStmt():
                    body = list(stmt.body.stmts)
                    if _falls_through(body):
                        continue
                    # the assignment stays inside the `with`: its right-hand
                    # side rounds under that context
                    return list(stmts[:i]) + [ContextStmt(
                        stmt.target, stmt.ctx,
                        StmtBlock(self._sink(body, [], depth)), stmt.loc,
                    )]
        return list(stmts) + self._sink(cont, [], depth)

class SingleExit:
    """Rewrites a function to a single trailing `return`.

    An early return becomes an assignment to one result name, and the
    statements that would have followed move into the branch that falls
    through::

        if c:               if c:
            return a   ⇝        r = a
        S ...               else:
        return b                S ...
                                r = b
                            return r

    Nothing is duplicated: a branch that returns cannot reach what follows, so
    the continuation moves into the other one.

    Required by consumers that cannot express an early exit -- `FPCoreCompiler`
    rejects multiple returns outright, `FuncInline` refuses a callee with more
    than one, and `SimplifyIf` refuses a `return` inside a branch.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')
        why = _returns_in_loop(func)
        if why is not None:
            raise TransformDeclined(f'cannot give `{func.name}` a single exit: {why}')
        if len(Reachability.analyze(func).ret_stmts) <= 1:
            return func

        def_use = DefineUse.analyze(func)
        result = Gensym(reserved=def_use.names()).fresh('r')
        ast = _SingleExitInstance(func, result).apply()
        left = len(Reachability.analyze(ast).ret_stmts)
        if left != 1:
            # a shape `_sink` does not move -- a conditional return inside a
            # `with`, say.  Refuse rather than hand back a function that still
            # has several exits.
            raise TransformDeclined(
                f'cannot give `{func.name}` a single exit: {left} returns '
                'remain after rewriting'
            )
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
