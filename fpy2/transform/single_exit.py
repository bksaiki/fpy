"""Transformation pass to give a function a single trailing return."""

from collections.abc import Callable

from ..analysis import DefineUse, Reachability, SyntaxCheck, TypeInfer
from ..analysis.type_infer import TypeInferError
from ..ast import *
from ..types import BoolType, ListType, RealType, TupleType, Type, VarType
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


_Placeholder = Callable[[Location | None], Expr]
"""A fresh expression of the function's return type, at a location."""


def _placeholder_of(ty: Type) -> _Placeholder | None:
    """A value of type *ty*, or `None` where there is none to write."""
    match ty:
        case RealType() | VarType():
            return lambda loc: Decnum('0', loc)
        case BoolType():
            return lambda loc: BoolVal(False, loc)
        case ListType():
            return lambda loc: ListExpr([], loc)
        case TupleType():
            elts = [_placeholder_of(t) for t in ty.elts]
            if any(e is None for e in elts):
                return None
            return lambda loc: TupleExpr([e(loc) for e in elts if e is not None], loc)
    return None


def _sink(
    result: NamedId, stmts: list[Stmt], cont: list[Stmt], depth: int = 0,
    *, done: NamedId, placeholder: _Placeholder | None,
    flag: NamedId | None = None,
) -> list[Stmt]:
    """*stmts* then *cont*, with every `return` rewritten to an assignment.

    `cont` is placed in each arm that falls through: where one does that moves
    it, where both do it is copied.  FPy requires the result be assigned on
    every path, and no single place is reachable from exactly the non-returning
    paths -- which is what the `return` was doing -- so each carries its own.

    *flag* is set inside a loop body, where a `return` cannot move the
    continuation -- it has to be recorded and tested after the loop instead.
    *done* is the one flag name the whole function shares: a `return` in a
    nested loop has to stop every loop enclosing it, and a flag per loop stops
    only the innermost.
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
                                    depth, done=done, flag=flag,
                                    placeholder=placeholder)),
                    StmtBlock(_sink(result, iff, iff_cont if iff_out else [],
                                    depth, done=done, flag=flag,
                                    placeholder=placeholder)),
                    stmt.loc,
                )]
            case ForStmt() | WhileStmt():
                body = list(stmt.body.stmts)
                if not _has_return(body):
                    continue
                if placeholder is None:
                    raise TransformDeclined(
                        'a `return` inside a loop needs a placeholder of the '
                        'return type for the result, and this type has none'
                    )
                # the code after the loop is also reached from its own exit, so
                # a `return` sets a flag the continuation is tested against
                inner = _sink(result, body, [], depth, done=done, flag=done,
                              placeholder=placeholder)
                loop: Stmt
                if isinstance(stmt, WhileStmt):
                    # in the condition, so the loop stops rather than spins;
                    # `and` short-circuits, so `cond` is not evaluated after
                    loop = WhileStmt(
                        And([Not(Var(done, stmt.loc), stmt.loc), stmt.cond],
                            stmt.loc),
                        StmtBlock(inner), stmt.loc,
                    )
                else:
                    # the whole body is guarded, so a store after the
                    # returning `if` stops too
                    loop = ForStmt(
                        stmt.target, stmt.iterable,
                        StmtBlock([If1Stmt(
                            Not(Var(done, stmt.loc), stmt.loc),
                            StmtBlock(inner), stmt.loc,
                        )]),
                        stmt.loc,
                    )
                tail = _sink(result, rest, [], depth, done=done, flag=flag,
                             placeholder=placeholder)
                # FPy admits no `if` with an empty body
                guarded = [If1Stmt(
                    Not(Var(done, stmt.loc), stmt.loc),
                    StmtBlock(tail), stmt.loc,
                )] if tail else []
                return list(stmts[:i]) + [
                    # never read, but FPy requires definite assignment
                    Assign(result, None, placeholder(stmt.loc), stmt.loc),
                    # reached only while the shared flag is clear
                    Assign(done, None, BoolVal(False, stmt.loc), stmt.loc),
                    loop,
                    *guarded,
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
                                    done=done, flag=flag, placeholder=placeholder)),
                    stmt.loc,
                )]
    if not cont:
        # nothing left to thread.  A function body always ends in a `return`,
        # so this only comes up for a loop body, which falls through.
        return list(stmts)
    return list(stmts) + _sink(result, cont, [], depth,
                               done=done, flag=flag, placeholder=placeholder)


def _return_placeholder(func: FuncDef) -> _Placeholder | None:
    """A value of *func*'s return type, where it has one.  A function that
    does not type-check keeps the real placeholder it always had."""
    try:
        ty = TypeInfer.check(func).fn_type.return_type
    except TypeInferError:
        return _placeholder_of(RealType())
    return _placeholder_of(ty)


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
        stmts = _sink(result, list(func.body.stmts), [],
                      done=gensym.fresh('done'),
                      placeholder=_return_placeholder(func))
        stmts.append(ReturnStmt(Var(result, None), None))
        ast = FuncDef(func.name, func.args, StmtBlock(stmts), func.meta,
                      loc=func.loc)

        left = len(Reachability.analyze(ast).ret_stmts)
        if left != 1:
            raise TransformDeclined(
                f'cannot give `{func.name}` a single exit: {left} returns '
                'remain -- a `return` under a `with` that only sometimes '
                'returns'
            )
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
