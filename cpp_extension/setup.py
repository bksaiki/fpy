import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "fpy2_models" / "csrc"
GENERATED = ROOT / "build" / "generated_models"
COMPILE_MODELS = ROOT.parent / "examples" / "mmasim" / "compile.py"
MODELS = (
    ("nv.volta.f16.f32", "nv.volta.f16.f32.cpp"),
    ("nv.turing.f16.f32", "nv.turing.f16.f32.cpp"),
    ("nv.ampere.tf32.f32", "nv.ampere.tf32.f32.cpp"),
    ("nv.ampere.bf16.f32", "nv.ampere.bf16.f32.cpp"),
    ("nv.ada.e5m2.f32", "nv.ada.e5m2.f32.cpp"),
    ("nv.hopper.f16.f32", "nv.hopper.f16.f32.cpp"),
    ("nv.blackwell.mxfp8", "nv.blackwell.mxfp8.cpp"),
    ("nv.blackwell.nvfp4", "nv.blackwell.nvfp4.cpp"),
    ("amd.cdna2.bf16", "amd.cdna2.bf16.cpp"),
    ("amd.cdna2.f16", "amd.cdna2.f16.cpp"),
    ("amd.cdna3.f16", "amd.cdna3.f16.cpp"),
    ("amd.cdna3.bf16", "amd.cdna3.bf16.cpp"),
    ("amd.cdna3.bf8", "amd.cdna3.bf8.cpp"),
    ("fp64 (fma)", "fp64_fma.cpp"),
)
SOURCES = [CSRC / "torch_ops.cpp", *(GENERATED / filename for _, filename in MODELS)]


class GeneratedBuildExtension(BuildExtension):
    """Generate the FPy model translation units only when building the extension."""

    def run(self):
        GENERATED.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(COMPILE_MODELS),
                *(name for name, _ in MODELS),
                "--out",
                str(GENERATED),
            ],
            check=True,
        )
        super().run()


setup(
    name="fpy2-cpp-extension",
    version="0.1.0",
    packages=["fpy2_models"],
    install_requires=["torch>=2.10", "torchao>=0.18,<0.19"],
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
    cmdclass={"build_ext": GeneratedBuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
