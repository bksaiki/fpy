"""
Definition analyzer for the FPy AST.
"""

from .fpyast import *
from .visitor import AstVisitor

_DefSet = set[NamedId]

class DefinitionAnalysisInstance(AstVisitor):
    """Single-use definition analyzer."""
    func: FunctionDef

    def __init__(self, func: FunctionDef):
        self.func = func

    def analyze(self):
        """Analyze the function."""
        self._visit_function(self.func, set())

    def _visit_var(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_bool(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_decnum(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_hexnum(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_integer(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_rational(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_digits(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_constant(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_unaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_binaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_ternaryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_naryop(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_compare(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_call(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_tuple_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_comp_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_ref_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError('should not be called')

    def _visit_var_assign(self, stmt: VarAssign, ctx: _DefSet):
        if isinstance(stmt.var, NamedId):
            ctx = ctx | { stmt.var }
        return ctx

    def _visit_tuple_assign(self, stmt: TupleAssign, ctx: _DefSet):
        return ctx | stmt.binding.names()

    def _visit_ref_assign(self, stmt, ctx):
        return ctx

    def _visit_if_stmt(self, stmt: IfStmt, ctx: _DefSet):
        ift_defs = self._visit_block(stmt.ift, ctx)
        if stmt.iff is None:
            return ctx | ift_defs
        else:
            iff_defs = self._visit_block(stmt.iff, ctx)
            return ctx | ift_defs | iff_defs

    def _visit_while_stmt(self, stmt: WhileStmt, ctx: _DefSet):
        block_defs = self._visit_block(stmt.body, ctx)
        return ctx | block_defs

    def _visit_for_stmt(self, stmt: ForStmt, ctx: _DefSet):
        if isinstance(stmt.var, NamedId):
            ctx = ctx | { stmt.var }
        block_defs = self._visit_block(stmt.body, ctx)
        return ctx | block_defs

    def _visit_context(self, stmt: ContextStmt, ctx: _DefSet):
        if stmt.name is not None and isinstance(stmt.name, NamedId):
            ctx = ctx | { stmt.name }
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: _DefSet):
        return ctx

    def _visit_return(self, stmt: Return, ctx: _DefSet):
        return set(ctx)

    def _visit_block(self, block: Block, ctx: _DefSet):
        def_in = ctx
        def_out: _DefSet = set()
        for stmt in block.stmts:
            def_out = self._visit_statement(stmt, def_out)
        block.attribs[DefinitionAnalysis.analysis_name] = (def_in, def_out)
        return def_out

    def _visit_function(self, func: FunctionDef, ctx: _DefSet):
        ctx = set(ctx)
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                ctx.add(arg.name)
        self._visit_block(func.body, ctx)

    # for typing hint
    def _visit_statement(self, stmt: Stmt, ctx: _DefSet) -> _DefSet:
        return super()._visit_statement(stmt, ctx)

class DefinitionAnalysis:
    """
    Definition analyzer on the AST.

    Associates to each statement block the set of variables defined in it.
    This is not the same as the set of variables in scope.
    """

    analysis_name = 'def_vars'

    @staticmethod
    def analyze(func: FunctionDef):
        """Analyze the defined variables in a function."""
        if not isinstance(func, FunctionDef):
            raise TypeError(f'expected a Function, got {func}')
        DefinitionAnalysisInstance(func).analyze()
