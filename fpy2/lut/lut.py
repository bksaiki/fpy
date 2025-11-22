"""
Lookup table generator (LUT) module.
"""

from typing import Callable, Iterable, Iterator, TypeAlias

from ..number import EncodableContext, Float
from ..function import Function


_Prim: TypeAlias = Callable[..., Float]
_LutCallable: TypeAlias = _Prim | Function


class LUTIterator(Iterator[tuple[Float | tuple[Float, ...], Float]]):
    """
    An iterator over entries in a lookup table in encoding order.
    Returns (key, value) pairs like a dictionary.
    """

    lut: 'LUT'
    index: int

    def __init__(self, lut: 'LUT') -> None:
        self.lut = lut
        self.index = 0

    def __next__(self) -> tuple[tuple[Float, ...], Float]:
        if self.index >= len(self.lut):
            raise StopIteration

        # Convert flat index to encodings
        encodings = []
        remaining = self.index
        for ctx in reversed(self.lut.arg_ctxs):
            max_enc = ctx.max_encoding() + 1
            encodings.append(remaining % max_enc)
            remaining //= max_enc
        encodings.reverse()

        # Decode to get Float key(s)
        args = tuple(ctx.decode(enc) for ctx, enc in zip(self.lut.arg_ctxs, encodings))

        # Ensure table is built and access directly by flat index
        self.lut._ensure_table()
        value = self.lut._table[self.index]  # type: ignore

        self.index += 1
        return (args, value)


class LUT(Iterable):
    """
    A lookup table for a function with fixed argument contexts
    and rounding context. The table is constructed lazily.
    """
    fn: _LutCallable
    arg_ctxs: tuple[EncodableContext, ...]
    ctx: EncodableContext
    _table: list[Float] | None

    def __init__(self, 
        fn: _LutCallable,
        arg_ctxs: tuple[EncodableContext, ...],
        ctx: EncodableContext
    ) -> None:
        self.fn = fn
        self.arg_ctxs = arg_ctxs
        self.ctx = ctx
        self._table = None

    def _ensure_table(self) -> None:
        """
        Force construction of the lookup table if not already built.
        """
        if self._table is not None:
            return
        
        size = len(self)
        self._table = [None] * size  # type: ignore
        
        # Generate all entries
        for i in range(size):
            # Convert flat index to multi-dimensional encodings
            encodings = []
            remaining = i
            for ctx in reversed(self.arg_ctxs):
                max_enc = ctx.max_encoding() + 1
                encodings.append(remaining % max_enc)
                remaining //= max_enc
            encodings.reverse()
            
            # Decode and compute
            args = [ctx.decode(enc) for ctx, enc in zip(self.arg_ctxs, encodings)]
            if len(args) == 0:
                result = self.fn(ctx=self.ctx)
            else:
                result = self.fn(*args, ctx=self.ctx)
            
            self._table[i] = result

    def __iter__(self) -> LUTIterator:
        """
        Create an iterator for the lookup table in encoding order.

        Yields:
            (key, value) tuples where key is Float or tuple[Float, ...]
            and value is the computed Float result.
        """
        return LUTIterator(self)

    def force(self):
        """
        Force construction of the lookup table.
        """
        self._ensure_table()

    def keys(self) -> Iterator[tuple[Float, ...]]:
        """
        Returns an iterator over the keys (Float arguments) in the lookup table.

        Yields:
            Float values (for single argument functions) or
            tuple[Float, ...] (for multi-argument functions)
        """
        for i in range(len(self)):
            # Convert flat index to encodings
            encodings = []
            remaining = i
            for ctx in reversed(self.arg_ctxs):
                max_enc = ctx.max_encoding() + 1
                encodings.append(remaining % max_enc)
                remaining //= max_enc
            encodings.reverse()

            # decode to get key
            yield tuple(ctx.decode(enc) for ctx, enc in zip(self.arg_ctxs, encodings))


    def __len__(self) -> int:
        """
        Returns the total number of entries in the lookup table.
        """
        total = 1
        for ctx in self.arg_ctxs:
            total *= (ctx.max_encoding() + 1)
        return total

    def __getitem__(self, index: Float | tuple[Float, ...]) -> Float:
        """
        Index into the lookup table using Float values.

        Args:
            index: Either a single Float (for single argument functions) 
                   or a tuple of Float values (for multi-argument functions).

        Returns:
            The computed Float result at that index.
        """
        # Ensure table is built
        self._ensure_table()

        # Convert Float argument(s) to encoding indices
        if isinstance(index, tuple):
            # Multiple Float arguments
            if len(index) != len(self.arg_ctxs):
                raise IndexError(f"Expected {len(self.arg_ctxs)} Float arguments, got {len(index)}")

            # Encode each Float to get the encoding index
            encodings = []
            for i, (arg, ctx) in enumerate(zip(index, self.arg_ctxs)):
                if not isinstance(arg, Float):
                    raise TypeError(f"Argument {i} must be Float, got {type(arg)}")
                enc = ctx.encode(arg)
                encodings.append(enc)

            # Convert encoding tuple to flat index
            flat_index = 0
            multiplier = 1
            for enc, ctx in zip(reversed(encodings), reversed(self.arg_ctxs)):
                flat_index += enc * multiplier
                multiplier *= (ctx.max_encoding() + 1)

            return self._table[flat_index]  # type: ignore

        else:
            # Single Float argument
            if not isinstance(index, Float):
                raise TypeError(f"Index must be Float or tuple[Float, ...], got {type(index)}")
            
            if len(self.arg_ctxs) != 1:
                raise IndexError(f"Expected tuple of {len(self.arg_ctxs)} Float values for indexing")
            
            # Encode the Float to get the flat index
            flat_index = self.arg_ctxs[0].encode(index)
            
            if flat_index < 0 or flat_index >= len(self):
                raise IndexError(f"Encoded index {flat_index} out of range for LUT of size {len(self)}")
            
            return self._table[flat_index]  # type: ignore


class LUTGenerator:
    """
    A LUT generator 
    """

    @staticmethod
    def generate(
        fn: _LutCallable,
        *arg_ctxs: EncodableContext,
        ctx: EncodableContext
    ) -> LUT:
        """
        Generate a lookup table iterator for the given function and argument contexts.

        Args:
            fn: The function or callable to generate the LUT for.
            arg_ctxs: An iterable of argument contexts.
            ctx: The context for the output values.

        Returns:
            A LUTIterator for the generated lookup table.
        """
        return LUT(fn, arg_ctxs, ctx)
