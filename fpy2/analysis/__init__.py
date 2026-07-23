"""Program analyses for FPy programs"""

from .array_size import (
    ArraySize,
    ArraySizeAnalysis,
    ArraySizeBound,
    ArraySizeInfer,
    ListSize,
    TupleSize,
    concrete_size,
)
from .call_graph import CallGraph, CallGraphAnalysis, CallGraphError
from .context_use import (
    ContextScope,
    ContextScopeSite,
    ContextUse,
    ContextUseAnalysis,
    ContextUseSite,
)
from .define_use import DefCtx, DefineUse, DefineUseAnalysis, UseSite
from .defs import DefAnalysis
from .format_infer import FormatAnalysis, FormatInfer
from .live_vars import LiveVars
from .partial_eval import PartialEval, PartialEvalInfo
from .purity import Purity
from .reachability import Reachability
from .reaching_defs import (
    AssignDef,
    Definition,
    DefSite,
    PhiDef,
    PhiSite,
    ReachingDefs,
    ReachingDefsAnalysis,
)
from .syntax_check import FPySyntaxError, SyntaxCheck
from .type_infer import TypeAnalysis, TypeInfer, TypeInferError
