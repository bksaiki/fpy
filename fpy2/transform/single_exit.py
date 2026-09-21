"""Transformation pass to give a function a single trailing return."""

from ..analysis import DefineUse, Reachability, SyntaxCheck
from ..ast import *
from ..utils import Gensym
from .error import TransformDeclined


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


def _falls_through(stmts: list[Stmt]) -> bool:
    """Is there a path through *stmts* that does not return?"""
    if not stmts:
        return True
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

    def _sink(self, stmts: list[Stmt], cont: list[Stmt]) -> list[Stmt]:
        """*stmts* then *cont*, with every `return` rewritten to an assignment.

        `cont` goes into whichever branch of an `if` falls through, so it is
        moved rather than copied: a branch that returns cannot reach it, and if
        both fall through the `if` needs no rewriting at all.
        """
        for i, stmt in enumerate(stmts):
            rest = list(stmts[i + 1:]) + cont
            match stmt:
                case ReturnStmt():
                    # anything after a return is unreachable
                    return list(stmts[:i]) + [
                        Assign(self.result, None, stmt.expr, None)
                    ]
                case IfStmt():
                    merged = self._merge(
                        stmt, list(stmt.ift.stmts), list(stmt.iff.stmts), rest,
                    )
                    if merged is None:
                        continue    # neither arm returns; nothing to move
                    return list(stmts[:i]) + [merged]
                case If1Stmt():
                    merged = self._merge(stmt, list(stmt.body.stmts), [], rest)
                    if merged is None:
                        continue
                    return list(stmts[:i]) + [merged]
                case ContextStmt():
                    body = list(stmt.body.stmts)
                    if _falls_through(body):
                        continue    # handled where it returns, or not at all
                    # the body always returns, so `rest` is unreachable.  The
                    # assignment stays inside the `with`: its right-hand side
                    # rounds under that context.
                    return list(stmts[:i]) + [ContextStmt(
                        stmt.target, stmt.ctx,
                        StmtBlock(self._sink(body, [])), stmt.loc,
                    )]
        if not cont:
            return list(stmts)
        return list(stmts) + self._sink(cont, [])

    def _merge(
        self, stmt: Stmt, ift: list[Stmt], iff: list[Stmt], rest: list[Stmt],
    ) -> 'IfStmt | None':
        """The `if` with *rest* moved into whichever arm falls through, or
        `None` where both do and there is nothing to move."""
        ift_out, iff_out = _falls_through(ift), _falls_through(iff)
        if ift_out and iff_out:
            return None
        assert isinstance(stmt, IfStmt | If1Stmt)
        return IfStmt(
            stmt.cond,
            StmtBlock(self._sink(ift, rest if ift_out else [])),
            StmtBlock(self._sink(iff, rest if iff_out else [])),
            stmt.loc,
        )

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
