"""Transformation pass to rewrite if statements to if expressions."""

from ..analysis import ContextUse, DefineUse, DefineUseAnalysis, SyntaxCheck
from ..analysis.context_use import ContextUseAnalysis
from ..ast import *
from ..number import Context, OverflowMode
from ..utils import Gensym
from .copy_propagate import CopyPropagate
from .error import TransformDeclined
from .rename_target import RenameTarget

# Constructs that cannot be hoisted out of a branch under any mode: each can
# change whether, or which, value the function produces.  Scoped statements
# (`with`) are not listed -- the scan descends into them, since their contents
# become unconditional too.
_UNHOISTABLE: dict[type[Stmt], str] = {
    ReturnStmt: 'a `return` escapes the branch and has no expression form',
    AssertStmt: 'an `assert` would run unconditionally and can abort',
    EffectStmt: 'an effect would run unconditionally',
    IndexedAssign: 'a list write would run unconditionally',
    WhileStmt: 'a `while` would run unconditionally and may not terminate',
    ForStmt: 'a `for` would run unconditionally',
}


class _Unhoistable(DefaultVisitor):
    """Why a branch body cannot be hoisted, in two categories.

    ``aborts`` can change whether, or which, value the function produces, so no
    mode admits it.  ``unproven`` is preserved in value but not in observable
    effect, which is what ``strict`` governs.

    Descends to any depth: a branch's statements all become unconditional, and
    so do those of any block nested inside it.
    """

    def __init__(self, ctx_use: ContextUseAnalysis):
        super().__init__()
        self.ctx_use = ctx_use
        self.aborts: str | None = None
        self.unproven: str | None = None

    def _reject(self, node: Stmt) -> None:
        if self.aborts is None:
            self.aborts = _UNHOISTABLE[type(node)]

    def _cannot_prove(self, why: str) -> None:
        if self.unproven is None:
            self.unproven = why

    def _visit_return(self, stmt: ReturnStmt, ctx):
        self._reject(stmt)

    def _visit_assert(self, stmt: AssertStmt, ctx):
        self._reject(stmt)
        super()._visit_assert(stmt, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx):
        self._reject(stmt)
        super()._visit_effect(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        self._reject(stmt)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx):
        self._reject(stmt)
        super()._visit_while(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        self._reject(stmt)
        super()._visit_for(stmt, ctx)

    def _visit_list_ref(self, e: ListRef, ctx):
        self._cannot_prove('a subscript may be out of range outside its guard')
        super()._visit_list_ref(e, ctx)

    def _visit_list_slice(self, e: ListSlice, ctx):
        self._cannot_prove('a slice may be out of range outside its guard')
        super()._visit_list_slice(e, ctx)

    def _visit_unaryop(self, e: UnaryOp, ctx):
        if isinstance(e, Cast) and self.aborts is None:
            self.aborts = (
                '`fp.cast` asserts its result is exact, so hoisting it can abort'
            )
        elif isinstance(e, Round):
            self._check_round(e)
        super()._visit_unaryop(e, ctx)

    def _check_round(self, e: Round) -> None:
        """A rounding aborts where its context overflows by assertion.

        A context that does not resolve to a concrete one -- a function with no
        ``ctx=`` inherits its caller's -- cannot be shown either way, so it is
        ``unproven`` rather than an abort.  Refusing it outright would decline
        every rounding inside a branch of an unannotated function.
        """
        try:
            resolved = self.ctx_use.find_scope_from_use(e).ctx
        except KeyError:
            resolved = None
        if isinstance(resolved, Context):
            if getattr(resolved, 'overflow', None) is OverflowMode.ASSERT:
                if self.aborts is None:
                    self.aborts = (
                        'a rounding under an `ASSERT` overflow context can abort'
                    )
        else:
            self._cannot_prove(
                'a rounding under an unresolved context may abort on overflow'
            )


def _why_unhoistable(
    block: StmtBlock, ctx_use: ContextUseAnalysis, strict: bool,
) -> str | None:
    v = _Unhoistable(ctx_use)
    v._visit_block(block, None)
    return v.aborts or (v.unproven if strict else None)


class _SimplifyIfInstance(DefaultTransformVisitor):
    """Single-use instance of the SimplifyIf pass."""
    func: FuncDef
    def_use: DefineUseAnalysis
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        ctx_use: ContextUseAnalysis,
        strict: bool,
    ):
        self.func = func
        self.def_use = def_use
        self.ctx_use = ctx_use
        self.strict = strict
        self.gensym = Gensym(reserved=def_use.names())

    def apply(self):
        func = self._visit_function(self.func, None)
        return func, self.gensym.generated

    def _require_hoistable(self, block: StmtBlock) -> None:
        why = _why_unhoistable(block, self.ctx_use, self.strict)
        if why is not None:
            raise TransformDeclined(f'cannot rewrite `if` to `if` expression: {why}')

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._require_hoistable(stmt.body)
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = Assign(t, BoolTypeAnn(None), cond, None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the body
        body, _ = self._visit_block(stmt.body, ctx)

        # identify variables that were mutated in the body
        mutated = self.def_use.mutated_in(stmt.body)

        # rename mutated variables in the body
        rename = { var: self.gensym.refresh(var) for var in mutated }
        body = RenameTarget.apply_block(body, rename)

        # generate assignments and inline the body
        for var in mutated:
            t = rename[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(body.stmts)

        # make if expressions for each mutated variable
        for var in mutated:
            e = IfExpr(cond, Var(rename[var], None), Var(var, None), None)
            s = Assign(var, None, e, None)
            stmts.append(s)

        return StmtBlock(stmts)


    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._require_hoistable(stmt.ift)
        self._require_hoistable(stmt.iff)
        stmts: list[Stmt] = []

        # compile condition
        cond = self._visit_expr(stmt.cond, ctx)

        # generate temporary if needed
        if not isinstance(cond, Var):
            t = self.gensym.fresh('cond')
            s = Assign(t, BoolTypeAnn(None), cond, None)
            stmts.append(s)
            cond = Var(t, None)

        # compile the bodies
        ift, _ = self._visit_block(stmt.ift, ctx)
        iff, _ = self._visit_block(stmt.iff, ctx)

        # identify variables that were mutated in each body
        mutated_ift = self.def_use.mutated_in(stmt.ift)
        mutated_iff = self.def_use.mutated_in(stmt.iff)

        # identify variables that were introduced in the bodies
        # FPy semantics says they must be introduced in both branches
        intros_ift = self.def_use.introed_in(stmt.ift)
        intros_iff = self.def_use.introed_in(stmt.iff)
        intros = sorted(intros_ift & intros_iff) # intersection of fresh variables

        # combine sets
        mutated_or_new_ift = sorted(mutated_ift)
        mutated_or_new_iff = sorted(mutated_iff)
        mutated_or_new_ift.extend(intros)
        mutated_or_new_iff.extend(intros)

        # rename mutated variables in each body, generate assignments, and inline
        rename_ift = { var: self.gensym.refresh(var) for var in mutated_or_new_ift }
        rename_iff = { var: self.gensym.refresh(var) for var in mutated_or_new_iff }

        ift = RenameTarget.apply_block(ift, rename_ift)
        iff = RenameTarget.apply_block(iff, rename_iff)

        for var in mutated_ift:
            t = rename_ift[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(ift.stmts)

        for var in mutated_iff:
            t = rename_iff[var]
            s = Assign(t, None, Var(var, None), None)
            stmts.append(s)
        stmts.extend(iff.stmts)

        # make if expressions for each mutated or introduced variable
        unique: set[NamedId] = set()
        for var in mutated_or_new_ift:
            if var not in unique:
                ift_name = rename_ift.get(var, var)
                iff_name = rename_iff.get(var, var)
                e = IfExpr(cond, Var(ift_name, None), Var(iff_name, None), None)
                s = Assign(var, None, e, None)
                stmts.append(s)
                unique.add(var)

        return StmtBlock(stmts)


    def _visit_block(self, block: StmtBlock, ctx: None):
        stmts: list[Stmt] = []
        for stmt in block.stmts:
            match stmt:
                case If1Stmt():
                    if1_block = self._visit_if1(stmt, ctx)
                    stmts.extend(if1_block.stmts)
                case IfStmt():
                    if_block = self._visit_if(stmt, ctx)
                    stmts.extend(if_block.stmts)
                case _:
                    stmt, _ = self._visit_statement(stmt, ctx)
                    stmts.append(stmt)
        return StmtBlock(stmts), None


#
# This transformation rewrites a block of the form:
# ```
# if <cond>
#     S1 ...
# else:
#     S2 ...
# S3 ...
# ```
# to an equivalent block using if expressions:
# ```
# t = <cond>
# S1 ...
# S2 ...
# x_i = x_{i, S1} if t else x_{i, S2}
# S3 ...
# ```
# where `x_i` is a phi node merging `phi(x_{i, S1}` and `x_{i, S2})`
# that is associated with the if-statement and `t` is a free variable.

class SimplifyIf:
    """
    Control flow simplification:

    Transforms if statements into if expressions.
    The inner block is hoisted into the outer block and each
    phi variable is made explicit with an if expression.

    Hoisting makes a branch body unconditional, so a construct that could
    change whether -- or which -- value the function produces is declined under
    every mode: `return`, `assert`, an effect, a list write, `while`, `for`,
    `fp.cast`, and a rounding under an `ASSERT` overflow context.

    ``strict`` governs what is left: operations whose *value* is preserved but
    whose observable effects may differ.  ``strict=False`` (the default) hoists
    them, in the same spirit as `CppCompiler.unsafe_cast_int` -- an
    out-of-range subscript is behavior FPy already leaves undefined.
    ``strict=True`` declines them, making the rewrite observationally
    equivalent.

    A consumer that evaluates only the taken arm of an `IfExpr` -- C++ `?:`,
    the interpreter -- gets that equivalence for free and wants the default.
    """

    @staticmethod
    def apply(func: FuncDef, *, strict: bool = False):
        def_use = DefineUse.analyze(func)
        ctx_use = ContextUse.analyze(func, def_use=def_use)
        ast, new_ids = _SimplifyIfInstance(func, def_use, ctx_use, strict).apply()
        ast = CopyPropagate.apply(ast, names=new_ids)
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast
