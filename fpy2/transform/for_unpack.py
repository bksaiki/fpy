"""
Transformation pass to push tuple unpacking in a for loop to the body.
"""

from ..analysis import DefineUse
from ..ast import *
from ..utils import Gensym

class _ForUnpackInstance(DefaultAstTransformVisitor):
    """Single-use instance of the ForUnpack pass."""
    func: FuncDef
    gensym: Gensym

    def __init__(self, func: FuncDef, names: set[NamedId]):
        self.func = func
        self.gensym = Gensym(reserved=names)

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_for(self, stmt: ForStmt, ctx: None) -> tuple[ForStmt, None]:
        match stmt.target:
            case Id():
                return super()._visit_for(stmt, None)
            case TupleBinding():
                t_id = self.gensym.fresh('t')
                iterable = self._visit_expr(stmt.iterable, None)
                body, _ = self._visit_block(stmt.body, None)
                phis, _ = self._visit_loop_phis(stmt.phis, ctx, None)

                binding = self._copy_tuple_binding(stmt.target)
                body.stmts.insert(0, TupleUnpack(binding, AnyType(), Var(t_id)))
                s = ForStmt(t_id, stmt.ty, iterable, body, phis)
                print(s.format())
                return s, None
            case _:
                raise RuntimeError('unreachable', stmt.target)


class ForUnpack:
    """
    Transformation pass to move any tuple unpacking in a for loop to its body.

    ```
    for x, y in iterable:
        ...
    ```
    becomes

    ```
    for t in iterable:
        x, y = t
        ...
    ```

    where `t` is a fresh variable.
    """

    @staticmethod
    def apply(func: FuncDef, names: Optional[set[NamedId]] = None) -> FuncDef:
        """
        Apply the transformation to the given function definition.
        """
        if names is None:
            def_use = DefineUse.analyze(func)
            names = set(def_use.defs.keys())
        inst = _ForUnpackInstance(func, names)
        func = inst.apply()
        SyntaxCheck.check(func)
        return func
