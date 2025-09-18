"""
Pure function analysis.
"""

from ..ast import *
from ..function import Function
from ..number import Context
from ..primitive import Primitive

from .define_use import DefineUse, DefineUseAnalysis, AssignDef

class _ImpureError(Exception):
    """
    Exception raised when an impure exception is detected.
    """
    pass


class _Purity(DefaultVisitor):
    """
    Purity analysis visitor.
    """

    func: FuncDef
    def_use: DefineUseAnalysis

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use

    def apply(self) -> bool:
        try:
            self._visit_function(self.func, None)
        except _ImpureError:
            return False
        return True

    def _visit_call(self, e: Call, ctx: None):
        super()._visit_call(e, ctx)
        match e.fn:
            case None:
                # unknown function -> impure by default
                raise _ImpureError(f'Impure: Unknown function call {e}')
            case Function():
                # user-defined function -> recursively check purity
                is_pure = Purity.analyze(e.fn.ast)
                if not is_pure:
                    raise _ImpureError(f'Impure: Call to impure function {e.fn.name}')
            case Primitive():
                # primitive function -> assume pure
                # TODO: how to specify unpure primitives?
                pass
            case type() if issubclass(e.fn, Context):
                # context constructor -> assume pure
                pass
            case _ if e.fn == print:
                # print function
                raise _ImpureError(f'Impure: call to print {e}')
            case _:
                raise NotImplementedError(f'unknown call {e}')

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        super()._visit_indexed_assign(stmt, ctx)
        d = self.def_use.find_def_from_use(stmt)
        if isinstance(d, AssignDef) and isinstance(d.site, Argument | FuncDef):
            # modifying an argument or a free variable
            raise _ImpureError(f'Impure: Indexed assignment {stmt}')


class Purity:
    """
    Pure function analysis.

    A function is considered pure if it has no side effects.
    """

    @staticmethod
    def analyze(func: FuncDef, def_use: DefineUseAnalysis | None = None) -> bool:
        """
        Analyze the given function and return True if it is pure, False otherwise.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')
        
        if def_use is None:
            def_use = DefineUse.analyze(func)
        return _Purity(func, def_use).apply()
