"""Unit tests for Unionfind data structure."""
import pytest
from hypothesis import given, strategies as st
from fpy2.utils.unionfind import Unionfind


class TestUnionfindBasic():
    """Basic unit tests for Unionfind."""

    def test_empty_initialization(self):
        """Test creating an empty unionfind."""
        uf: Unionfind[int] = Unionfind()
        assert len(uf) == 0
        assert list(uf) == []

    def test_initialization_with_elements(self):
        """Test creating unionfind with initial elements."""
        uf: Unionfind[int] = Unionfind([1, 2, 3, 4])
        assert len(uf) == 4
        assert uf.representatives() == {1, 2, 3, 4}

    def test_add_single_element(self):
        """Test adding a single element."""
        uf: Unionfind[int] = Unionfind()
        rep = uf.add(1)
        assert rep == 1
        assert len(uf) == 1
        assert 1 in uf

    def test_add_duplicate_element(self):
        """Test adding an element that already exists."""
        uf = Unionfind([1])
        rep = uf.add(1)
        assert rep == 1
        assert len(uf) == 1

    def test_find_existing_element(self):
        """Test finding an existing element."""
        uf = Unionfind([1, 2, 3])
        assert uf.find(1) == 1
        assert uf.find(2) == 2
        assert uf.find(3) == 3

    def test_find_nonexistent_element(self):
        """Test finding a non-existent element raises KeyError."""
        uf = Unionfind([1, 2, 3])
        with pytest.raises(KeyError):
            uf.find(4)

    def test_get_existing_element(self):
        """Test get with existing element."""
        uf = Unionfind([1, 2, 3])
        assert uf.get(1) == 1
        assert uf.get(2) == 2

    def test_get_nonexistent_element(self):
        """Test get with non-existent element returns default."""
        uf = Unionfind([1, 2, 3])
        assert uf.get(4) is None
        assert uf.get(4, "default") == "default"

    def test_union_two_elements(self):
        """Test union of two elements."""
        uf = Unionfind([1, 2, 3])
        rep = uf.union(1, 2)
        assert rep == 1
        assert uf.find(1) == uf.find(2)
        assert len(uf.representatives()) == 2

    def test_union_nonexistent_first(self):
        """Test union with non-existent first element."""
        uf = Unionfind([1, 2])
        with pytest.raises(KeyError):
            uf.union(3, 1)

    def test_union_nonexistent_second(self):
        """Test union with non-existent second element."""
        uf = Unionfind([1, 2])
        with pytest.raises(KeyError):
            uf.union(1, 3)

    def test_union_chain(self):
        """Test chaining multiple unions."""
        uf = Unionfind([1, 2, 3, 4, 5])
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        # All should have same representative
        rep = uf.find(1)
        assert uf.find(2) == rep
        assert uf.find(3) == rep
        assert uf.find(4) == rep
        # 5 should be separate
        assert uf.find(5) != rep
        assert len(uf.representatives()) == 2

    def test_contains(self):
        """Test membership checking."""
        uf = Unionfind([1, 2, 3])
        assert 1 in uf
        assert 2 in uf
        assert 4 not in uf

    def test_iter(self):
        """Test iteration over elements."""
        uf = Unionfind([1, 2, 3])
        uf.union(1, 2)
        reps = list(uf)
        assert len(reps) == 3
        assert uf.find(1) in reps
        assert uf.find(2) in reps
        assert uf.find(3) in reps

    def test_repr(self):
        """Test string representation."""
        uf = Unionfind([1, 2])
        repr_str = repr(uf)
        assert "Unionfind" in repr_str

    def test_representatives_after_unions(self):
        """Test representatives after performing unions."""
        uf = Unionfind(range(10))
        # Create two groups: {0,1,2,3,4} and {5,6,7,8,9}
        for i in range(4):
            uf.union(i, i + 1)
        for i in range(5, 9):
            uf.union(i, i + 1)
        
        reps = uf.representatives()
        assert len(reps) == 2

    def test_string_elements(self):
        """Test unionfind with string elements."""
        uf = Unionfind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("a") != uf.find("c")


class TestUnionfindModel():
    """Model-based test: the model maps each element to its block."""

    @given(
        st.lists(st.integers(-6, 6), unique=True),
        st.lists(st.tuples(st.booleans(), st.integers(-6, 6), st.integers(-6, 6)), max_size=30),
    )
    def test_matches_model(self, init: list[int], ops: list[tuple[bool, int, int]]):
        uf = Unionfind(init)
        model = {x: frozenset([x]) for x in init}
        assert uf.representatives() == set(init)
        for is_union, x, y in ops:
            if not is_union:
                expect = uf.find(x) if x in model else x
                assert uf.add(x) == expect
                model.setdefault(x, frozenset([x]))
            elif x in model and y in model:
                expect = uf.find(x)
                assert uf.union(x, y) == expect
                block = model[x] | model[y]
                model.update(dict.fromkeys(block, block))
            else:
                with pytest.raises(KeyError):
                    uf.union(x, y)
            assert len(uf) == len(model)
            assert set(uf) == set(model)
            assert len(uf.representatives()) == len(set(model.values()))
            assert uf.components() == set(model.values())
            for a, block_a in model.items():
                assert uf.find(a) in block_a
                assert uf.component(a) == block_a
                for b, block_b in model.items():
                    assert (uf.find(a) == uf.find(b)) == (block_a == block_b)
