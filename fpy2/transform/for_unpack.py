"""
Transformation pass to push tuple unpacking in a for loop to the body.
"""

from ..analysis import DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast import *
from ..utils import Gensym

class _ForUnpackInstance(DefaultTransformVisitor):
    """Single-use instance of the ForUnpack pass."""
    func: FuncDef
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_for(self, stmt: ForStmt, ctx: None) -> tuple[ForStmt, None]:
        match stmt.target:
            case Id():
                return super()._visit_for(stmt, None)
            case TupleBinding():
                # compile iterable and body
                iterable = self._visit_expr(stmt.iterable, None)
                body, _ = self._visit_block(stmt.body, None)

                # create a fresh variable for the tuple
                t = self.gensym.fresh('t')
                binding = self._visit_tuple_binding(stmt.target, ctx)

                # insert tuple unpacking at the beginning of the body
                body.stmts.insert(0, Assign(binding, None, Var(t, None), None))

                # create the for statement with the fresh variable
                s = ForStmt(t, iterable, body, None)
                return s, None
            case _:
                raise RuntimeError('unreachable', stmt.target)


class ForUnpack:
    """
    Transformation pass to move any tuple unpacking in a for loop to its body::

        for x, y in iterable:
            ...

    becomes::

        for t in iterable:
            x, y = t
            ...

    where `t` is a fresh variable.
    """

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """
        Apply the transformation to the given function definition.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got `{func}`')
        def_use = DefineUse.analyze(func)
        inst = _ForUnpackInstance(func, def_use)
        func = inst.apply()
        SyntaxCheck.check(func, ignore_unknown=True)
        return func
