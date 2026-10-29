import json
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

    Every design `generate_models.py` supports is compiled; its manifest lists
    the translation units, which the extension builds with its operators.
    """

    def run(self):
        subprocess.run([sys.executable, str(GENERATOR), str(GENERATED)], check=True)
        manifest = json.loads((GENERATED / "manifest.json").read_text())
        sources = [
            CSRC / "torch_ops.cpp",
            GENERATED / "torch_ops_generated.cpp",
            *(GENERATED / model["file"] for model in manifest["models"]),
        ]
        for extension in self.extensions:
            extension.sources = [str(path) for path in sources]
        super().run()


setup(
    name="fpy2-cpp-extension",
    version="0.1.0",
    packages=["fpy2_models"],
    install_requires=["torch>=2.10", "torchao>=0.18,<0.19"],
    ext_modules=[
        CppExtension(
            "fpy2_models._C",
            [str(CSRC / "torch_ops.cpp")],
            include_dirs=[str(CSRC)],
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
