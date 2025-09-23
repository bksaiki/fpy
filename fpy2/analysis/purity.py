"""
Pure function analysis.
"""

from ..ast import *
from ..function import Function
from ..number import Context
from ..primitive import Primitive

from .define_use import AssignDef, DefineUse, DefineUseAnalysis

class _ImpureError(Exception):
    """
    Exception raised when an impure exception is detected.
    """
    pass


class _Purity(DefaultVisitor):
    """
    Purity analysis visitor.
    """

    ast: FuncDef | Expr
    def_use: DefineUseAnalysis

    def __init__(self, ast: FuncDef | Expr, def_use: DefineUseAnalysis):
        self.ast = ast
        self.def_use = def_use

    def apply(self) -> bool:
        try:
            match self.ast:
                case FuncDef():
                    self._visit_function(self.ast, None)
                case Expr():
                    self._visit_expr(self.ast, None)
                case _:
                    raise RuntimeError(f'Unexpected AST node `{self.ast}`')
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

    @staticmethod
    def analyze_expr(expr: Expr, def_use: DefineUseAnalysis) -> bool:
        """
        Analyze the given expression and return True if it is pure, False otherwise.
        """
        if not isinstance(expr, Expr):
            raise TypeError(f'Expected `Expr`, got {type(expr)} for {expr}')
        return _Purity(expr, def_use).apply()
