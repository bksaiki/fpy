"""Context inlining for FPy ASTs"""

from ..fpc_context import FPCoreContext
from ..number import Context
from ..env import ForeignEnv

from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor

class _ContextInlineInstance(DefaultTransformVisitor):
    """Per-IR instance of context inlining"""
    ast: FuncDef
    env: ForeignEnv

    def __init__(self, ast: FuncDef, env: ForeignEnv):
        self.ast = ast
        self.env = env

    def apply(self):
        return self._visit_function(self.ast, None)

    def _lookup(self, name: NamedId):
        if name.base not in self.env:
            raise NameError(f'free variable {name} not in environment')
        return self.env[name.base]

    def _eval_var(self, e: Var):
        if e.name.base not in self.env:
            raise NameError(f'free variable {e.name} not in environment')
        return self.env.get(e.name.base)

    def _eval_foreign_attr(self, e: ForeignAttribute):
        # lookup the root value (should be captured)
        val = self._lookup(e.name)
        # walk the attribute chain
        for attr_id in e.attrs:
            # need to manually lookup the attribute
            attr = str(attr_id)
            if isinstance(val, dict):
                if attr not in val:
                    raise RuntimeError(f'unknown attribute {attr} for {val}')
                val = val[attr]
            elif hasattr(val, attr):
                val = getattr(val, attr)
            else:
                raise RuntimeError(f'unknown attribute {attr} for {val}')
        return val

    def _eval_context_expr(self, e: ContextExpr):
        match e.ctor:
            case ForeignAttribute():
                ctor = self._eval_foreign_attr(e.ctor)
            case Var():
                ctor = self._eval_var(e.ctor)

        args: list[Any] = []
        for arg in e.args:
            match arg:
                case ForeignAttribute():
                    args.append(self._eval_foreign_attr(arg))
                case Integer():
                    args.append(arg.val)
                case Var():
                    args.append(self._eval_var(arg))
                case _:
                    # TODO: how to compute this
                    raise RuntimeError('cannot compute', arg)

        kwargs: dict[str, Any] = {}
        for k, v in e.kwargs:
            match v:
                case ForeignAttribute():
                    kwargs[k] = self._eval_foreign_attr(v)
                case Integer():
                    args.append(v.val)
                case Var():
                    args.append(self._eval_var(v))
                case _:
                    # TODO: how to compute this
                    raise RuntimeError('cannot compute', arg)

        return ctor(*args, **kwargs)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        match stmt.ctx:
            case ContextExpr():
                v = self._eval_context_expr(stmt.ctx)
                if not isinstance(v, Context | FPCoreContext):
                    raise TypeError(f'Expected `Context` or `FPCoreContext`, got {type(v)} for {v}')
                context = ForeignVal(v, None)
            case Var():
                v = self._eval_var(stmt.ctx)
                if not isinstance(v, Context | FPCoreContext):
                    raise TypeError(f'Expected `Context` or `FPCoreContext`, got {type(v)} for {v}')
                context = ForeignVal(v, None)
            case ForeignVal():
                context = stmt.ctx
            case _:
                raise RuntimeError('unreachable', stmt.ctx)

        body, _ = self._visit_block(stmt.body, None)
        s = ContextStmt(stmt.name, context, body, stmt.loc)
        return s, None


class ContextInline:
    """
    Context inliner.

    Contexts in FPy programs may be metaprogrammed.
    This pass resolves the context at each site.
    """

    @staticmethod
    def apply(ast: FuncDef, env: ForeignEnv) -> FuncDef:
        if not isinstance(ast, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(ast)} for {ast}')
        return _ContextInlineInstance(ast, env).apply()
