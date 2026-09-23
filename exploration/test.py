import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend import CppCompiler

@fp.fpy
def f(x: fp.Real):
    with fp.BF16:
        return fp.round(x)

cppc = CppCompiler()

f = st.unfold_overflow(f)
f = st.unfold_special(f)
f = st.float_to_fixed(f)
f = st.rescale_fixed(f)
print(f)

print(cppc.compile(f, ctx=fp.REAL, arg_types=[fp.types.RealType(fp.FP32)]))
