"""Command-line interface for LUT compiler."""

import argparse
import importlib.util
import sys

from pathlib import Path
from typing import Callable, Literal

from .lut import LUTGenerator
from .backend import CppLUT
from .spec import parse_context_spec
from ..number import EncodableContext


def _load_function_from_file(path: Path, func_name: str) -> Callable:
    """Dynamically load a function from a Python file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    # Load module from file
    spec = importlib.util.spec_from_file_location("_lut_input_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["_lut_input_module"] = module
    spec.loader.exec_module(module)
    
    # Get function
    if not hasattr(module, func_name):
        raise AttributeError(f"Function '{func_name}' not found in {path}")
    
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"'{func_name}' is not a callable function")
    
    return func


def _log(verbose: bool, message: str):
    """Log message if verbose is enabled."""
    if verbose:
        print(message)

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="fpy-lut",
        description="C++ compiler of FPy functions into LUT-based implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Context Specification Format:
  ieee754(es, nbits, [rm])                    IEEE 754 floating-point
  efloat(es, nbits, inf, nan, off, [rm])      Extended floating-point
  fixed(signed, scale, nbits, [rm])           Fixed-point

Examples:
  ieee754(5, 32, RNE)                         32-bit IEEE 754 float (FP32)
  ieee754(8, 64)                              64-bit IEEE 754 float (FP64, default RNE)
  efloat(3, 8, true, ieee, 0, RNE)           8-bit float with IEEE semantics
  fixed(true, 8, 16, RTZ)                     16-bit signed fixed-point, scale 2^8, round toward zero

Rounding Modes:
  RNE - Round to nearest, ties to even (default)
  RNA - Round to nearest, ties away from zero
  RTZ - Round toward zero
  RTP - Round toward positive infinity
  RTN - Round toward negative infinity
  RAZ - Round away from zero
  RTO - Round toward odd
  RTE - Round toward even
"""
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Python file containing the function to compile"
    )
    
    parser.add_argument(
        "-n", "--name",
        required=True,
        metavar="FUNC",
        help="Name of the function to compile"
    )
    
    parser.add_argument(
        "-a", "--arg-spec",
        required=True,
        action="append",
        metavar="SPEC",
        help="Context specification for each argument (use multiple times for multiple args)"
    )
    
    parser.add_argument(
        "-r", "--return-spec",
        required=True,
        metavar="SPEC",
        help="Context specification for return value"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="FILE",
        help="Output C++ file path"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["array", "switch"],
        default="array",
        help="Code generation mode (default: array)"
    )
    
    parser.add_argument(
        "--indent",
        default="    ",
        metavar="STR",
        help="Indentation string (default: 4 spaces)"
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = _parse_args()
    
    # command-line arguments
    input_path: Path = Path(args.input).resolve()
    output_path: Path = Path(args.output).resolve()

    name: str = args.name
    arg_specs: list[str] = args.arg_spec
    return_spec: str = args.return_spec
    mode: Literal['array', 'switch'] = args.mode
    indent: str = args.indent
    verbose: bool = args.verbose
    
    try:
        # Parse context specifications
        _log(verbose, f"Parsing context specifications...")
        arg_contexts: list[EncodableContext] = []
        for spec in arg_specs:
            ctx = parse_context_spec(spec)
            arg_contexts.append(ctx)
            _log(verbose, f"  Argument context: {spec}")
        
        return_context = parse_context_spec(return_spec)
        _log(verbose, f"  Return context: {return_spec}")
        
        # Load function
        _log(verbose, f"\nLoading function '{name}' from {input_path}...")
        func = _load_function_from_file(input_path, name)
        
        # Generate LUT
        _log(verbose, f"\nGenerating lookup table...")
        lut = LUTGenerator.generate(func, *arg_contexts, ctx=return_context)
        
        # Force LUT construction to get size info
        lut.force()
        total_entries = len(lut._table) if lut._table else 0
        _log(verbose, f"  Total entries: {total_entries:,}")
        
        # Compile to C++
        _log(verbose, f"\nCompiling to C++ ({mode} mode)...")
        cpp_func_name = name
        
        cpp_code = CppLUT.compile_lut(
            lut=lut,
            func_name=cpp_func_name,
            method=mode,
            indent_str=indent,
        )

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cpp_code)
        
        _log(verbose, f"\nSuccessfully wrote C++ code to {output_path}")
        _log(verbose, f"Function signature: {cpp_func_name}(...)")
        
    except Exception as e:
        _log(verbose, f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
