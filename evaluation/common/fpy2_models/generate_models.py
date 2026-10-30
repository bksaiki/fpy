"""Generate the FPy model translation units and their Torch operators.

    python evaluation/common/fpy2_models/generate_models.py OUT_DIR

`examples/mmasim/compile.py -o` compiles every design to OUT_DIR/models/ and
lists their signatures; this wraps each one in a Torch operator, OUT_DIR/<op>.cpp.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # this file is in evaluation/common/fpy2_models
COMPILE = REPO_ROOT / "examples" / "mmasim" / "compile.py"


def _operator(op: str, model: dict) -> str:
    """The translation unit registering *model* as torch.ops.fpy2_models.<op>."""
    (a, _, _, *scales), t = model["params"], model["result"]["type"]
    fn = f"&{model['namespace']}::{model['entry']}"
    if not scales:
        params, runner = "", f"run_dpa<{t}, {a['size']}, {fn}>"
    else:
        s = scales[0].get("size", 1)  # MX models take scalar scales
        params = ", Tensor alphas, Tensor betas"
        runner = f"run_scaled_dpa<{t}, {a['size']}, {s}, {fn}>"
    return (
        f'#include "models/{model["file"]}"\n'
        '#include "runners.h"\n\n'
        "STABLE_TORCH_LIBRARY_FRAGMENT(fpy2_models, m) {\n"
        f'  m.def("{op}(Tensor a, Tensor b, Tensor c{params}) -> Tensor");\n'
        "}\n\n"
        "STABLE_TORCH_LIBRARY_IMPL(fpy2_models, CPU, m) {\n"
        f'  m.impl("{op}", TORCH_BOX((&fpy_models::runners::{runner})));\n'
        "}\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    # Exits 1 when any design is refused; its manifest lists them.
    jobs = str(os.cpu_count() or 1)
    subprocess.run([sys.executable, str(COMPILE), "-o", str(out / "models"), "-j", jobs], check=False)
    compiled = json.loads((out / "models" / "manifest.json").read_text())
    models = []
    for model in compiled["models"]:
        op = re.sub(r"\W+", "_", model["name"]).strip("_")  # fp64 (fma) -> fp64_fma
        (out / f"{op}.cpp").write_text(_operator(op, model))
        models.append({"name": model["name"], "op": op, "compute": model["result"]["type"],
                       "k": model["params"][0]["size"]})
    manifest = {"models": models, "unsupported": compiled["refused"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"generated {len(models)} models")
