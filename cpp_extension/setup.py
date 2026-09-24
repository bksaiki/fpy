from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "fpy2_models" / "csrc"
SOURCES = [CSRC / "torch_ops.cpp"] + sorted(
    path for path in CSRC.glob("*.cpp") if path.name != "torch_ops.cpp"
)

setup(
    name="fpy2-cpp-extension",
    version="0.1.0",
    packages=["fpy2_models"],
    install_requires=["torch>=2.10"],
    ext_modules=[
        CppExtension(
            "fpy2_models._C",
            [str(path) for path in SOURCES],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-DPy_LIMITED_API=0x03090000",
                    "-DTORCH_TARGET_VERSION=0x020a000000000000",
                ]
            },
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
