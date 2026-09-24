"""
Unit tests for :class:`fpy2.utils.Gensym`.

The property that matters is **injectivity**: distinct inputs to `refresh`
must give distinct outputs.  Everything that renames -- inlining above all --
builds a substitution map from it, and a map that collapses two names silently
merges two variables.

`NamedId` caches its hash, so a `refresh` that *mutated* `count` filed the
result under the old name's bucket.  The new name then looked unused, and
refreshing *it* handed it out a second time.  That needs a name that was
generated earlier to come back round -- which is what inlining a cached callee
does -- so these tests refresh outputs, not just inputs.
"""

from fpy2.utils import Gensym, NamedId, SourceId
from fpy2.utils.location import Location

_LOC = Location('<test>', 1, 0, 1, 0)


class TestRefreshIsInjective:
    def test_refreshing_an_output_moves_it(self):
        """The minimal shape: `t8` -> `tN`, then `tN` must not stay `tN`."""
        g = Gensym([NamedId('t', 8)])
        a = g.refresh(NamedId('t', 8))
        b = g.refresh(a)
        assert str(a) != str(b), f'`{a}` refreshed to itself'

    def test_a_chain_of_refreshes_stays_distinct(self):
        g = Gensym([NamedId('t', 8)])
        seen = [NamedId('t', 8)]
        for _ in range(20):
            seen.append(g.refresh(seen[-1]))
        names = [str(n) for n in seen]
        assert len(set(names)) == len(names), f'collided: {names}'

    def test_two_names_never_meet(self):
        """Distinct inputs, interleaved with their own outputs."""
        g = Gensym([NamedId('t', 8), NamedId('x', 8)])
        out = []
        a, b = NamedId('t', 8), NamedId('x', 8)
        for _ in range(10):
            a, b = g.refresh(a), g.refresh(b)
            out += [str(a), str(b)]
        assert len(set(out)) == len(out), f'collided: {out}'

    def test_the_input_is_never_mutated(self):
        """`refresh` copies: a caller holding the old name keeps it."""
        name = NamedId('t', 8)
        Gensym([name]).refresh(name)
        assert str(name) == 't8'

    def test_a_source_id_stays_a_source_id(self):
        g = Gensym([SourceId('x', _LOC, 1)])
        out = g.refresh(SourceId('x', _LOC, 1))
        assert isinstance(out, SourceId)
        assert str(out) != 'x1'

    def test_an_unreserved_name_is_left_alone(self):
        assert str(Gensym([NamedId('a')]).refresh(NamedId('b'))) == 'b'


def test_fresh_names_do_not_collide():
    g = Gensym([NamedId('t', i) for i in range(5)])
    names = {str(g.fresh('t')) for _ in range(20)}
    assert len(names) == 20
