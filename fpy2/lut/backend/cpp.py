"""
C++ backend for LUT compilation.

Compiles lookup tables into C++ functions that use array-based
or switch-based implementations for fast lookup.
"""

import math
from typing import Literal

from ...number import EncodableContext, Float
from ..lut import LUT


class _CppLUTCompiler:
    """
    Internal compiler state for C++ LUT code generation.
    """
    
    lut: LUT
    func_name: str
    output_type: str
    input_types: list[str]
    num_args: int
    dims: list[int]
    hex_width: int
    include_guards: bool
    indent_str: str
    
    def __init__(self, lut: LUT, func_name: str = "lut_lookup", include_guards: bool = True, indent_str: str = "    "):
        self.lut = lut
        self.func_name = func_name
        self.num_args = len(lut.arg_ctxs)
        self.include_guards = include_guards
        self.indent_str = indent_str
        
        if self.num_args == 0:
            raise ValueError("LUT must have at least one argument context")
        
        # Determine appropriate uint types for each input (check bounds)
        self.input_types = []
        for i, ctx in enumerate(lut.arg_ctxs):
            try:
                uint_type = self._determine_uint_type(ctx.max_encoding())
                self.input_types.append(uint_type)
            except ValueError as e:
                raise ValueError(f"Argument {i} context encoding exceeds 64-bit range: {e}") from e
        
        # Determine output type (check bounds)
        try:
            self.output_type = self._determine_uint_type(lut.ctx.max_encoding())
        except ValueError as e:
            raise ValueError(f"Output context encoding exceeds 64-bit range: {e}") from e
        
        # Calculate dimensions
        self.dims = [ctx.max_encoding() + 1 for ctx in lut.arg_ctxs]
        
        # Determine hex width for output type
        if self.output_type == "uint8_t":
            self.hex_width = 2
        elif self.output_type == "uint16_t":
            self.hex_width = 4
        elif self.output_type == "uint32_t":
            self.hex_width = 8
        else:  # uint64_t
            self.hex_width = 16
    
    @staticmethod
    def _determine_uint_type(max_encoding: int) -> str:
        """
        Determines the appropriate uint_t type based on max encoding value.
        
        Args:
            max_encoding: The maximum encoding value needed
            
        Returns:
            C++ type string (e.g., 'uint8_t', 'uint16_t', etc.)
            
        Raises:
            ValueError: If max_encoding exceeds 64-bit unsigned integer range
        """
        if max_encoding <= 0xFF:
            return "uint8_t"
        elif max_encoding <= 0xFFFF:
            return "uint16_t"
        elif max_encoding <= 0xFFFFFFFF:
            return "uint32_t"
        elif max_encoding <= 0xFFFFFFFFFFFFFFFF:
            return "uint64_t"
        else:
            raise ValueError(
                f"Encoding value {max_encoding} exceeds 64-bit range. "
                f"Cannot represent in C++ uint64_t (max: {0xFFFFFFFFFFFFFFFF})"
            )
    
    def _format_hex(self, value: int) -> str:
        """
        Formats an integer as a hexadecimal literal with the specified width.
        
        Args:
            value: The integer value to format
            
        Returns:
            Hex string like '0x1a' or '0x00ff'
        """
        return f"0x{value:0{self.hex_width}x}"
    
    def _build_signature(self) -> str:
        """
        Builds the C++ function signature.
        
        Returns:
            Function signature string
        """
        params = ", ".join(
            f"{self.input_types[i]} arg{i}"
            for i in range(self.num_args)
        )
        return f"{self.output_type} {self.func_name}({params})"
    
    def _build_headers(self) -> list[str]:
        """
        Builds the include headers if enabled.
        
        Returns:
            List of header lines
        """
        if self.include_guards:
            return ["#include <cstdint>", ""]
        return []
    
    def compile_array(self) -> list[str]:
        """
        Generates array-based lookup implementation.
        
        Returns:
            List of code lines (indented)
        """
        lines = []
        
        # Build array declaration with appropriate dimensions
        if self.num_args == 1:
            array_decl = f"{self.indent_str}static const {self.output_type} table[{self.dims[0]}]"
        else:
            dim_str = "][".join(str(d) for d in self.dims)
            array_decl = f"{self.indent_str}static const {self.output_type} table[{dim_str}]"
        
        lines.append(array_decl + " = {")
        
        # Generate array initialization
        if self.num_args == 1:
            # Simple 1D array
            for i in range(self.dims[0]):
                encoded = self.lut.ctx.encode(self.lut._table[i])  # type: ignore
                hex_val = self._format_hex(encoded)
                comma = "," if i < self.dims[0] - 1 else ""
                lines.append(f"{self.indent_str * 2}{hex_val}{comma}")
        else:
            # Multi-dimensional array - format with nesting
            lines.extend(self._format_nested_array(0, 0, indent=2))
        
        lines.append(f"{self.indent_str}}};")  
        lines.append("")
        
        # Generate index calculation and return
        if self.num_args == 1:
            lines.append(f"{self.indent_str}return table[arg0];")
        else:
            # Calculate flat index from multi-dimensional indices
            index_expr = "arg0"
            for i in range(1, self.num_args):
                index_expr = f"({index_expr} * {self.dims[i]} + arg{i})"
            lines.append(f"{self.indent_str}return table[{index_expr}];")
        
        return lines
    
    def _format_nested_array(self, depth: int, base_index: int, indent: int = 0) -> list[str]:
        """
        Recursively formats nested array initialization.
        """
        lines = []
        indent_str = self.indent_str * indent
        
        if depth == len(self.dims) - 1:
            # Innermost dimension - output values
            for i in range(self.dims[depth]):
                flat_idx = base_index + i
                encoded = self.lut.ctx.encode(self.lut._table[flat_idx])  # type: ignore
                hex_val = self._format_hex(encoded)
                comma = "," if i < self.dims[depth] - 1 else ""
                lines.append(f"{indent_str}{hex_val}{comma}")
        else:
            # Outer dimensions - recurse
            stride = 1
            for d in self.dims[depth + 1:]:
                stride *= d
            
            for i in range(self.dims[depth]):
                lines.append(f"{indent_str}{{")
                new_base = base_index + i * stride
                lines.extend(self._format_nested_array(depth + 1, new_base, indent + 1))
                comma = "," if i < self.dims[depth] - 1 else ""
                lines.append(f"{indent_str}}}{comma}")
        
        return lines
    
    def compile_switch(self) -> list[str]:
        """
        Generates switch-based lookup implementation.
        
        Returns:
            List of code lines (indented)
        """
        lines = []
        
        # Generate nested switch statements
        lines.extend(self._generate_nested_switch(0, 0, indent=1))
        
        # Default case (should never happen if inputs are valid)
        lines.append(f"{self.indent_str}return 0; // Should never reach here")
        
        return lines
    
    def _generate_nested_switch(self, depth: int, base_index: int, indent: int = 0) -> list[str]:
        """
        Recursively generates nested switch statements.
        """
        lines = []
        indent_str = self.indent_str * indent
        
        lines.append(f"{indent_str}switch (arg{depth}) {{")
        
        if depth == len(self.dims) - 1:
            # Innermost switch - return values directly
            for i in range(self.dims[depth]):
                flat_idx = base_index + i
                encoded = self.lut.ctx.encode(self.lut._table[flat_idx])  # type: ignore
                hex_val = self._format_hex(encoded)
                lines.append(f"{indent_str * 2}case {i}: return {hex_val};")
        else:
            # Outer switches - recurse
            stride = 1
            for d in self.dims[depth + 1:]:
                stride *= d
            
            for i in range(self.dims[depth]):
                lines.append(f"{indent_str * 2}case {i}:")
                new_base = base_index + i * stride
                lines.extend(self._generate_nested_switch(depth + 1, new_base, indent + 2))
        
        lines.append(f"{indent_str}}}")
        
        return lines


class CppLUT:
    """
    C++ code generator for lookup tables.
    """

    @staticmethod
    def compile_lut(
        lut: LUT,
        func_name: str = "lut_lookup",
        method: Literal["array", "switch"] = "array",
        include_guards: bool = True,
        indent_str: str = 4 * " "
    ) -> str:
        """
        Compiles a LUT instance into a C++ function.
        
        Args:
            lut: The lookup table to compile
            func_name: Name of the generated C++ function
            method: Either "array" for array-based lookup or "switch" for switch-case
            include_guards: Whether to include necessary headers
            indent_str: String to use for each indentation level (default: "    " - 4 spaces)
            
        Returns:
            C++ source code as a string
        """
        # Ensure the table is built
        lut.force()
        
        # Create compiler instance
        compiler = _CppLUTCompiler(lut, func_name, include_guards, indent_str)
        
        # Generate code
        lines = []
        
        # Add headers
        lines.extend(compiler._build_headers())
        
        # Add function signature
        lines.append(f"{compiler._build_signature()} {{")
        
        # Add method implementation
        if method == "array":
            lines.extend(compiler.compile_array())
        elif method == "switch":
            lines.extend(compiler.compile_switch())
        else:
            raise ValueError(f"Unknown method: {method}")
        
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
