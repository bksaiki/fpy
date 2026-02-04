"""Program analyses for FPy programs"""

from .array_size import ArraySizeInfer, ArraySizeAnalysis
from .context_infer import ContextInfer, ContextAnalysis, ContextInferError
from .defs import DefAnalysis
from .define_use import (
    DefineUse, DefineUseAnalysis,
    DefCtx, UseSite
)
from .live_vars import LiveVars
from .partial_eval import PartialEval, PartialEvalInfo
from .purity import Purity
from .reachability import Reachability
from .reaching_defs import (
    ReachingDefs, ReachingDefsAnalysis,
    AssignDef, PhiDef, Definition,
    DefSite, PhiSite
)
from .syntax_check import SyntaxCheck, FPySyntaxError
from .type_infer import TypeInfer, TypeAnalysis, TypeInferError
