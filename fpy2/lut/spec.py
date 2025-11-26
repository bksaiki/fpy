"""Context specification parser for LUT compiler."""

import re
from typing import Optional

from ..number import (
    EncodableContext,
    IEEEContext,
    EFloatContext,
    FixedContext,
    RoundingMode,
)
from ..number.context.efloat import EFloatNanKind


def parse_rounding_mode(rm_str: str) -> RoundingMode:
    """Parse rounding mode string to RoundingMode enum."""
    rm_map = {
        "RNE": RoundingMode.RNE,
        "RNA": RoundingMode.RNA,
        "RTZ": RoundingMode.RTZ,
        "RTP": RoundingMode.RTP,
        "RTN": RoundingMode.RTN,
        "RAZ": RoundingMode.RAZ,
        "RTO": RoundingMode.RTO,
        "RTE": RoundingMode.RTE,
    }
    
    if rm_str not in rm_map:
        valid = ", ".join(rm_map.keys())
        raise ValueError(f"Invalid rounding mode '{rm_str}'. Must be one of: {valid}")
    
    return rm_map[rm_str]


def parse_bool(s: str) -> bool:
    """Parse boolean string."""
    s_lower = s.lower()
    if s_lower in ("true", "t", "1", "yes", "y"):
        return True
    elif s_lower in ("false", "f", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Invalid boolean value '{s}'. Expected true/false.")


def parse_context_spec(spec: str) -> EncodableContext:
    """
    Parse a context specification string into an EncodableContext object.
    
    Supported formats:
    - ieee754(es, nbits, [rm])
    - efloat(es, nbits, enable_inf, nan_style, eoffset, [rm])
    - fixed(signed, scale, nbits, [rm])
    
    Examples:
        ieee754(5, 32, RNE)
        efloat(5, 8, true, ieee, 0, RNE)
        fixed(true, 8, 16, RTZ)
    """
    spec = spec.strip()
    
    # Match pattern: name(args)
    match = re.match(r'^(\w+)\((.*)\)$', spec)
    if not match:
        raise ValueError(f"Invalid context specification: '{spec}'")
    
    ctx_type = match.group(1)
    args_str = match.group(2).strip()
    
    # Parse arguments
    if args_str:
        # Simple comma-split (doesn't handle nested parens, but we don't need that)
        args = [arg.strip() for arg in args_str.split(',')]
    else:
        args = []
    
    # Dispatch based on context type
    if ctx_type == "ieee754":
        return _parse_ieee754(args)
    elif ctx_type == "efloat":
        return _parse_efloat(args)
    elif ctx_type == "fixed":
        return _parse_fixed(args)
    else:
        raise ValueError(f"Unknown context type '{ctx_type}'. Must be one of: ieee754, efloat, fixed")


def _parse_ieee754(args: list[str]) -> IEEEContext:
    """Parse ieee754(es, nbits, [rm])."""
    if len(args) < 2 or len(args) > 3:
        raise ValueError("ieee754 requires 2-3 arguments: ieee754(es, nbits, [rm])")
    
    try:
        es = int(args[0])
        nbits = int(args[1])
        rm = parse_rounding_mode(args[2]) if len(args) == 3 else RoundingMode.RNE
        
        return IEEEContext(es=es, nbits=nbits, rm=rm)
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"ieee754 arguments must be integers: {e}")
        raise


def _parse_efloat(args: list[str]) -> EFloatContext:
    """Parse efloat(es, nbits, enable_inf, nan_style, eoffset, [rm])."""
    if len(args) < 5 or len(args) > 6:
        raise ValueError("efloat requires 5-6 arguments: efloat(es, nbits, enable_inf, nan_style, eoffset, [rm])")
    
    try:
        es = int(args[0])
        nbits = int(args[1])
        enable_inf = parse_bool(args[2])
        nan_style = args[3].strip().lower()
        
        # Map nan_style string to EFloatNanKind
        nan_kind_map = {
            "ieee": EFloatNanKind.IEEE_754,
            "ieee754": EFloatNanKind.IEEE_754,
            "maxval": EFloatNanKind.MAX_VAL,
            "max_val": EFloatNanKind.MAX_VAL,
            "negzero": EFloatNanKind.NEG_ZERO,
            "neg_zero": EFloatNanKind.NEG_ZERO,
            "none": EFloatNanKind.NONE,
        }
        
        if nan_style not in nan_kind_map:
            valid = ", ".join(sorted(set(nan_kind_map.keys())))
            raise ValueError(f"nan_style must be one of {valid}, got '{nan_style}'")
        
        nan_kind = nan_kind_map[nan_style]
        eoffset = int(args[4])
        rm = parse_rounding_mode(args[5]) if len(args) == 6 else RoundingMode.RNE
        
        return EFloatContext(
            es=es,
            nbits=nbits,
            enable_inf=enable_inf,
            nan_kind=nan_kind,
            eoffset=eoffset,
            rm=rm,
        )
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"efloat numeric arguments must be integers: {e}")
        raise


def _parse_fixed(args: list[str]) -> FixedContext:
    """Parse fixed(signed, scale, nbits, [rm])."""
    if len(args) < 3 or len(args) > 4:
        raise ValueError("fixed requires 3-4 arguments: fixed(signed, scale, nbits, [rm])")
    
    try:
        signed = parse_bool(args[0])
        scale = int(args[1])
        nbits = int(args[2])
        rm = parse_rounding_mode(args[3]) if len(args) == 4 else RoundingMode.RNE
        
        return FixedContext(signed=signed, scale=scale, nbits=nbits, rm=rm)
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"fixed numeric arguments must be integers: {e}")
        raise
