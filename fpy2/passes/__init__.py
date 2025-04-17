"""Analysis or transformation passes on the FPy IR."""

from ..analysis.define_use import DefineUse
from ..transform.for_bundling import ForBundling
from ..transform.func_update import FuncUpdate
from ..transform.ssa import SSA
from ..transform.simplify_if import SimplifyIf
from ..analysis.verify import VerifyIR
from ..transform.while_bundling import WhileBundling
