"""
Unit tests for `Module` — phase 1 (registration + lazy public/private).
"""

import pytest

import titanfp.fpbench.fpcast as fpc

import fpy2 as fp

from fpy2 import CppCompiler, FPCoreCompiler, Module
from fpy2.types import RealType


# ----------------------------------------------------------------------
# Helpers: a small call graph — top -> {mid, leaf}, mid -> leaf.


def _funcs():
    @fp.fpy
    def leaf(x: fp.Real) -> fp.Real:
        return x + 1

    @fp.fpy
    def mid(x: fp.Real) -> fp.Real:
        return leaf(x) * 2

    @fp.fpy
    def top(x: fp.Real) -> fp.Real:
        return mid(x) + leaf(x)  # leaf reached via mid and directly

    return leaf, mid, top


class TestRegistration:
    def test_name_defaults_to_func_name(self):
        _, _, top = _funcs()
        m = Module()
        m.add(top)
        assert 'top' in m
        assert m.get('top').func is top
        assert len(m) == 1

    def test_explicit_name(self):
        _, _, top = _funcs()
        m = Module()
        m.add(top, name='entry')
        assert 'entry' in m and 'top' not in m
        assert m.get('entry').func is top

    def test_duplicate_name_raises(self):
        _, _, top = _funcs()
        m = Module()
        m.add(top)
        with pytest.raises(ValueError):
            m.add(top)

    def test_same_function_two_names_ok(self):
        _, _, top = _funcs()
        m = Module()
        m.add(top, name='a')
        m.add(top, name='b')
        assert len(m) == 2
        # public dedupes by identity
        assert m.public() == [top]

    def test_non_function_raises(self):
        m = Module()
        with pytest.raises(TypeError):
            m.add(object())  # type: ignore[arg-type]

    def test_bad_ctx_raises(self):
        _, _, top = _funcs()
        m = Module()
        with pytest.raises(TypeError):
            m.add(top, ctx='FP64')  # type: ignore[arg-type]

    def test_arg_types_length_mismatch_raises(self):
        _, _, top = _funcs()  # top has 1 arg
        m = Module()
        with pytest.raises(ValueError):
            m.add(top, arg_types=[RealType(fp.FP64), RealType(fp.FP64)])

    def test_monomorphization_spec_stored(self):
        _, _, top = _funcs()
        m = Module()
        m.add(top, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        e = m.get('top')
        assert e.ctx is fp.FP64
        assert e.arg_types == (RealType(fp.FP64),)

    def test_iteration_order(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        m.add(mid)
        assert [e.name for e in m] == ['top', 'mid']


class TestPublicPrivate:
    def test_private_derived_from_call_chain(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        assert m.public() == [top]
        assert {f.name for f in m.private()} == {'mid', 'leaf'}
        assert m.functions() == [top] + m.private()

    def test_private_is_leaves_first(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        order = [f.name for f in m.private()]
        # leaf (a leaf) precedes mid (its caller)
        assert order.index('leaf') < order.index('mid')

    def test_shared_callee_deduped(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        names = [f.name for f in m.private()]
        assert names.count('leaf') == 1

    def test_public_wins_over_private(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        m.add(mid)  # mid is also a callee of top, but it's registered
        assert {f.name for f in m.public()} == {'top', 'mid'}
        # mid is no longer private; only leaf is
        assert [f.name for f in m.private()] == ['leaf']

    def test_leaf_only_module_has_no_private(self):
        leaf, _, _ = _funcs()
        m = Module()
        m.add(leaf)
        assert m.public() == [leaf]
        assert m.private() == []

    def test_derivation_invalidated_on_add(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(mid)
        assert {f.name for f in m.functions()} == {'mid', 'leaf'}
        # adding `top` must surface it in the recomputed view
        m.add(top)
        assert {f.name for f in m.functions()} == {'top', 'mid', 'leaf'}


def _fpy_callees(func):
    """Callee `Function`s referenced in a function's body."""
    from fpy2.ast.visitor import DefaultVisitor
    from fpy2.function import Function

    out = []

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            if isinstance(e.fn, Function):
                out.append(e.fn)
            super()._visit_call(e, ctx)

    _C()._visit_function(func.ast, None)
    return out


class TestMap:
    def test_returns_new_module_original_unchanged(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        before = m.public() + m.private()
        m2 = m.map(lambda mod, fd: fd)
        assert m2 is not m
        # original module's functions are untouched (same objects)
        assert m.public() + m.private() == before
        assert all(a is not b for a, b in zip(before, m2.functions()))

    def test_transform_receives_the_module(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        seen_modules = []

        def T(mod, fd):
            seen_modules.append(mod)
            return fd

        m.map(T)
        # the module passed in is the one `.map` was called on, once per function
        assert seen_modules == [m] * len(m.functions())

    def test_preserves_public_entries(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top, name='entry', ctx=fp.FP64)
        m2 = m.map(lambda mod, fd: fd)
        assert [e.name for e in m2] == ['entry']
        assert m2.get('entry').ctx is fp.FP64
        assert {f.name for f in m2.private()} == {'mid', 'leaf'}

    def test_leaf_matches_direct_transform(self):
        # for a leaf (no calls), map's rebinding is a no-op, so the module
        # result is structurally identical to applying the transform directly.
        leaf, _, _ = _funcs()
        m = Module()
        m.add(leaf)
        T = lambda mod, fd: fp.transform.ConstFold.apply(fd, enable_op=False)
        direct = fp.transform.ConstFold.apply(leaf.ast, enable_op=False)
        through = m.map(T).get('leaf').func.ast
        assert through.is_equiv(direct)

    def test_rewires_calls_to_transformed_callees(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        m2 = m.map(lambda mod, fd: fd)
        new_ids = {id(f) for f in m2.functions()}
        old_ids = {id(f) for f in m.functions()}
        for caller in m2.functions():
            for callee in _fpy_callees(caller):
                assert id(callee) in new_ids        # points at a transformed func
                assert id(callee) not in old_ids    # not the stale original

    def test_preserves_semantics(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        T = lambda mod, fd: fp.transform.ConstFold.apply(fd, enable_op=False)
        new_top = m.map(T).get('top').func
        for xv in (0.0, 1.5, -3.25, 10.0):
            assert top(xv) == new_top(xv)

    def test_funcinline_through_module(self):
        # a transform that changes call structure: after inlining, the public
        # entry has no remaining FPy calls, and semantics are preserved.
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        m2 = m.map(lambda mod, fd: fp.transform.FuncInline.apply(fd))
        new_top = m2.get('top').func
        assert _fpy_callees(new_top) == []
        for xv in (0.0, 1.5, -3.25, 10.0):
            assert top(xv) == new_top(xv)


def _fpc_funcs():
    """Constant-free functions (FPCore rejects bare unrounded constants)."""
    @fp.fpy
    def helper(x: fp.Real) -> fp.Real:
        return x + x

    @fp.fpy
    def entry(x: fp.Real) -> fp.Real:
        return helper(x)

    return helper, entry


class TestCompileModule:
    def test_cpp_matches_manual_unit(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])

        through = CppCompiler().compile_module(m)
        unit = CppCompiler().unit()
        unit.add(top, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        assert through == unit.render()

    def test_cpp_includes_private_callees(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        out = CppCompiler().compile_module(m)
        # the entry and its (specialized) callees are all emitted
        assert 'top' in out and 'mid' in out and 'leaf' in out

    def test_cpp_rejects_non_module(self):
        with pytest.raises(TypeError):
            CppCompiler().compile_module(object())  # type: ignore[arg-type]

    def test_cpp_compiles_mapped_module(self):
        # map + compile compose: a mapped module still compiles
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        m2 = m.map(lambda mod, fd: fp.transform.FuncInline.apply(fd))
        out = CppCompiler().compile_module(m2)
        assert 'top' in out

    def test_fpc_returns_fpcore_per_entry(self):
        helper, entry = _fpc_funcs()
        m = Module()
        m.add(entry)
        m.add(helper, name='helper_pub')
        out = FPCoreCompiler().compile_module(m)
        assert sorted(out.keys()) == ['entry', 'helper_pub']
        assert all(isinstance(v, fpc.FPCore) for v in out.values())

    def test_fpc_ignores_arg_types(self):
        helper, entry = _fpc_funcs()
        m = Module()
        m.add(entry, arg_types=[RealType(fp.FP64)])  # fpc can't express this
        out = FPCoreCompiler().compile_module(m)
        assert isinstance(out['entry'], fpc.FPCore)

    def test_fpc_rejects_non_module(self):
        with pytest.raises(TypeError):
            FPCoreCompiler().compile_module(object())  # type: ignore[arg-type]


class TestCallGraph:
    def test_returns_module_call_graph(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        from fpy2 import ModuleCallGraph
        assert isinstance(cg, ModuleCallGraph)

    def test_publics_and_privates(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        assert cg.publics == [top]
        assert {f.name for f in cg.privates} == {'mid', 'leaf'}

    def test_callees_callers(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        assert cg.callees_of(top) == [mid, leaf]
        assert cg.callees_of(leaf) == []
        # leaf is called by mid (via the chain) and by top (directly)
        assert set(cg.callers_of(leaf)) == {mid, top}
        # top has no callers in the module
        assert cg.callers_of(top) == []

    def test_is_public(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        assert cg.is_public(top) is True
        assert cg.is_public(mid) is False
        assert cg.is_public(leaf) is False

    def test_iteration_leaves_first(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        order = list(cg)
        assert order.index(leaf) < order.index(mid)
        assert order.index(mid) < order.index(top)
        assert len(cg) == 3

    def test_contains(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        cg = m.call_graph()
        assert top in cg and mid in cg and leaf in cg

    def test_format_multi_root_shared_callee_marked(self):
        # two publics, both calling shared `helper -> leaf`
        @fp.fpy
        def leaf(x: fp.Real) -> fp.Real: return x + 1
        @fp.fpy
        def helper(x: fp.Real) -> fp.Real: return leaf(x) * 2
        @fp.fpy
        def top_a(x: fp.Real) -> fp.Real: return helper(x)
        @fp.fpy
        def top_b(x: fp.Real) -> fp.Real: return helper(x)

        m = Module(); m.add(top_a); m.add(top_b)
        out = m.call_graph().format()
        assert 'top_a' in out and 'top_b' in out
        # second occurrence of helper marked as a revisit
        assert 'helper (*)' in out

    def test_dot_styles_publics(self):
        leaf, mid, top = _funcs()
        m = Module()
        m.add(top)
        out = m.call_graph().dot()
        assert out.startswith('digraph module_call_graph {')
        # publics are bold; privates are not
        assert 'label="top", style=bold' in out
        assert 'label="mid"' in out and 'label="mid", style=bold' not in out
        # one edge per (caller, callee) pair
        assert out.count(' -> ') == 3   # top->mid, top->leaf, mid->leaf

    def test_call_graph_catches_cycle(self):
        from fpy2.analysis import CallGraphError
        from fpy2.ast.visitor import DefaultVisitor

        @fp.fpy
        def leaf(x: fp.Real) -> fp.Real: return x + 1
        @fp.fpy
        def m_(x: fp.Real) -> fp.Real: return leaf(x)

        # Patch m_'s only call to form a self-cycle m_ -> m_.
        calls = []
        class _C(DefaultVisitor):
            def _visit_call(self, e, ctx):
                calls.append(e); super()._visit_call(e, ctx)
        _C()._visit_function(m_.ast, None)
        calls[0].fn = m_

        mod = Module(); mod.add(m_)
        with pytest.raises(CallGraphError):
            mod.call_graph()
