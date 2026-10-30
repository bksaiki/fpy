import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "fpy2_models" / "csrc"
GENERATED = ROOT / "build" / "generated_models"
GENERATOR = ROOT / "generate_models.py"


class GeneratedBuildExtension(BuildExtension):
    """Generate the FPy model translation units only when building the extension.

    `generate_models.py` writes one translation unit per design, each
    registering its own operator; the extension builds all of them.
    """

    def run(self):
        subprocess.run([sys.executable, str(GENERATOR), str(GENERATED)], check=True)
        sources = [CSRC / "torch_ops.cpp", *sorted(GENERATED.glob("*.cpp"))]
        for extension in self.extensions:
            extension.sources = [str(path) for path in sources]
        super().run()


setup(
    name="fpy2-cpp-extension",
    version="0.1.0",
    packages=["fpy2_models"],
    install_requires=["torch>=2.10"],
    ext_modules=[
        CppExtension(
            "fpy2_models._C",
            [str(CSRC / "torch_ops.cpp")],
            include_dirs=[str(CSRC)],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    # The models set rounding modes with `fesetround` and must
                    # round every operation as written: no folding or moving
                    # arithmetic across a mode change, and no FMA contraction.
                    # GCC and Clang both accept these.
                    "-frounding-math",
                    "-ffp-contract=off",
                    # The torch::stable ABI version the operators target.
                    "-DTORCH_TARGET_VERSION=0x020a000000000000",
                ]
            },
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": GeneratedBuildExtension},
)
