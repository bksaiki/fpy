"""Transformation pass to rewrite if statements to if expressions."""

from ..gensym import Gensym
from ..fpyast import *
from ..visitor import DefaultTransformVisitor
from ..analysis import LiveVars, UniqueVars

class MergeIf(DefaultTransformVisitor):
    """
    Transforms if statements into if expressions.

    This transformation assumes SSA form and requires `DefUSe` and `LiveVars`.
    It transforms a block of the form:
    ```
    if <cond>:
        S1 ...
    else:
        S2 ...
    S3 ...
    ```
    to an equivalent block using if expressions:
    ```
    t = <cond>
    S1 ...
    S2 ...
    x_i = x_{i, S1} if t else x_{i, S2}
    ...
    S3 ...
    ```
    where {`x_i`} are the set of live variables at the start of `S3`
    that are defined or mutated within `S1` or `S2`. Additionally,
    any definition in `S1` and `S2` is renamed to some free variable to
    avoid namespace collisions. Likewise, `t` is some free variable.
    """

    def _if_phi_nodes(self, block: Block):
        """Collects all phi nodes corresponding to if statements."""
        if_to_phis: dict[IfStmt, list[Phi]] = {}
        for stmt in block.stmts:
            if isinstance(stmt, Phi):
                if stmt.branch in if_to_phis:
                    if_to_phis[stmt.branch].append(stmt)
                else:
                    if_to_phis[stmt.branch] = [stmt]
        return if_to_phis

    def _visit_block(self, block, ctx: Gensym):
        stmts: list[Stmt] = []
        if_to_phis = self._if_phi_nodes(block)
        for stmt in block.stmts:
            match stmt:
                case Assign():
                    stmts.append(self._visit(stmt, ctx))
                case TupleAssign():
                    stmts.append(self._visit(stmt, ctx))
                case Phi():
                    if not isinstance(stmt.branch, IfStmt):
                        stmts.append(Phi(stmt.name, stmt.lhs, stmt.rhs, stmt.branch))
                case Return():
                    stmts.append(self._visit(stmt, ctx))
                case _:
                    raise NotImplementedError('unreachable', stmt)


        return Block(stmts)

    def visit(self, e: Function | Block):
        if not isinstance(e, Function | Block):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
        # run live variables analysis (if needed)
        if LiveVars.analysis_name not in e.attribs:
            LiveVars().visit(e)

        unique_vars = UniqueVars().visit(e)
        return self._visit(e, Gensym(*unique_vars))
    