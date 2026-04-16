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


class TestUnionfindProperties():
    """Property-based tests using hypothesis."""

    @given(st.lists(st.integers(), min_size=1, max_size=50, unique=True))
    def test_initialization_preserves_elements(self, elements: list[int]):
        """All elements should be findable after initialization."""
        uf = Unionfind(elements)
        for elem in elements:
            assert elem in uf
            assert uf.find(elem) == elem

    @given(st.lists(st.integers(), min_size=0, max_size=50, unique=True))
    def test_length_matches_elements(self, elements: list[int]):
        """Length should match number of elements."""
        uf = Unionfind(elements)
        assert len(uf) == len(elements)

    @given(st.lists(st.integers(), min_size=1, max_size=50, unique=True))
    def test_initial_representatives_equal_elements(self, elements: list[int]):
        """Initially, each element is its own representative."""
        uf = Unionfind(elements)
        assert uf.representatives() == set(elements)

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50, unique=True)
    )
    def test_union_reflexivity(self, elements: list[int]):
        """Union of an element with itself should not change structure."""
        uf = Unionfind(elements)
        x = elements[0]
        original_rep = uf.find(x)
        uf.union(x, x)
        assert uf.find(x) == original_rep
        assert len(uf.representatives()) == len(elements)

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=3, max_size=50, unique=True)
    )
    def test_union_transitivity(self, elements: list[int]):
        """If union(a,b) and union(b,c), then find(a) == find(c)."""
        uf = Unionfind(elements)
        a, b, c = elements[0], elements[1], elements[2]
        uf.union(a, b)
        uf.union(b, c)
        assert uf.find(a) == uf.find(c)

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50, unique=True)
    )
    def test_union_symmetry(self, elements: list[int]):
        """Union(x, y) and union(y, x) should produce same representative."""
        uf1 = Unionfind(elements)
        uf2 = Unionfind(elements)
        x, y = elements[0], elements[1]
        
        uf1.union(x, y)
        uf2.union(y, x)
        
        # Both should be in same set
        assert uf1.find(x) == uf1.find(y)
        assert uf2.find(x) == uf2.find(y)

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=50, unique=True)
    )
    def test_union_reduces_representatives(self, elements: list[int]):
        """Union should reduce number of representatives."""
        uf = Unionfind(elements)
        initial_reps = len(uf.representatives())
        
        x, y = elements[0], elements[1]
        if uf.find(x) != uf.find(y):
            uf.union(x, y)
            final_reps = len(uf.representatives())
            assert final_reps == initial_reps - 1

    @given(st.lists(st.integers(), min_size=1, max_size=50, unique=True))
    def test_add_then_find(self, elements: list[int]):
        """Elements added should be findable."""
        uf: Unionfind[int] = Unionfind()
        for elem in elements:
            rep = uf.add(elem)
            assert rep == elem
            assert uf.find(elem) == elem

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=30, unique=True),
        st.data()
    )
    def test_random_unions(self, elements: list[int], data: st.DataObject):
        """Random unions should maintain invariants."""
        uf = Unionfind(elements)
        
        # Perform random unions
        num_unions = data.draw(st.integers(min_value=0, max_value=len(elements) - 1))
        for _ in range(num_unions):
            x = data.draw(st.sampled_from(elements))
            y = data.draw(st.sampled_from(elements))
            uf.union(x, y)
        
        # All elements should still be findable
        for elem in elements:
            assert elem in uf
            rep = uf.find(elem)
            assert rep in elements
        
        # Number of representatives should be valid
        num_reps = len(uf.representatives())
        assert num_reps > 0
        assert num_reps <= len(elements)

    @given(st.integers(min_value=-1000, max_value=1000))
    def test_add_idempotent(self, value: int):
        """Adding same element multiple times is idempotent."""
        uf: Unionfind[int] = Unionfind()
        rep1 = uf.add(value)
        rep2 = uf.add(value)
        rep3 = uf.add(value)
        assert rep1 == rep2
        assert rep2 == rep3
        assert len(uf) == 1

    @given(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=4, max_size=50, unique=True)
    )
    def test_union_associativity(self, elements: list[int]):
        """Union operations should be associative in their effect."""
        a, b, c, d = elements[0], elements[1], elements[2], elements[3]
        
        # (a∪b)∪(c∪d)
        uf1 = Unionfind(elements)
        uf1.union(a, b)
        uf1.union(c, d)
        uf1.union(a, c)
        
        # ((a∪b)∪c)∪d
        uf2 = Unionfind(elements)
        uf2.union(a, b)
        uf2.union(a, c)
        uf2.union(a, d)
        
        # All four should be in same set in both cases
        assert uf1.find(a) == uf1.find(b)
        assert uf1.find(a) == uf1.find(c)
        assert uf1.find(a) == uf1.find(d)
        
        assert uf2.find(a) == uf2.find(b)
        assert uf2.find(a) == uf2.find(c)
        assert uf2.find(a) == uf2.find(d)

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=30, unique=True))
    def test_works_with_strings(self, string_elements: list[str]):
        """Unionfind should work with string elements."""
        uf = Unionfind(string_elements)
        assert len(uf) == len(string_elements)
        
        if len(string_elements) >= 2:
            x, y = string_elements[0], string_elements[1]
            uf.union(x, y)
            assert uf.find(x) == uf.find(y)

