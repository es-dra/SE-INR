"""Re-embed existing recovery/qualitative figures with TrueType fonts; no model calls.

Run on the evidence server:
  CUDA_VISIBLE_DEVICES='' python3 render_shared_figures.py --out NEW_RUN_DIR \
    --reference-numbers repository_numbers.json --reference-qual-meta repository_qual_meta.json
Original scripts and saved results are read only. PNG intermediates stay in /tmp.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--runs", default="/workspace/SE-INR/runs")
p.add_argument("--data", default="/workspace/SE-INR/Data/DIV2K_valid_HR")
p.add_argument("--out", required=True)
p.add_argument("--reference-numbers", required=True)
p.add_argument("--reference-qual-meta", required=True)
a = p.parse_args()
runs, out = Path(a.runs), Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out / "figure-validation.json").exists(), "Never overwrite a completed redraw"
for name in ("fig1_recovery.pdf", "fig3_qualitative.pdf"):
    assert not (out / name).exists(), name
assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "CPU-only rendering requires hidden GPUs"
oldfig = runs / "claude-20261020-paper-figures"
qual = runs / "claude-20261023-qualitative-v14"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

sources = [oldfig / "make_figs.py", oldfig / "numbers.json", oldfig / "fig1_recovery.pdf",
           qual / "qual.py", qual / "FROZEN.sha256", qual / "qual_meta.json",
           qual / "crops.npz", qual / "fig3_qualitative.pdf",
           Path(a.reference_numbers), Path(a.reference_qual_meta)]
sources += [runs / rel for rel in [
    "claude-20261019-two-coordinates-v2/formal/rows.json",
    "claude-20261019-encoder-v3/formal_gs/rows.json",
    "claude-20261019-prior-v4/early_liif/rows.json",
    "claude-20261019-prior-v4/early_lte/rows.json",
    "claude-20261019-prior-v4/early_sc-inr/rows.json",
    "claude-20261019-encoder-train-v5/eval_edsr/rows.json",
    "claude-20261019-encoder-train-v5/eval_rdn/rows.json",
    "claude-20261019-encoder-train-v5/eval_tinydec/rows.json",
    "claude-20261020-swinir-v7/eval200_swinir/rows.json",
    "claude-20261020-transfer-v11/transfer_v11.json"]]
images = [x for x in sorted(Path(a.data).glob("*.png")) if 869 <= int(x.stem) <= 900]
assert len(images) == 32
sources += images
before = {str(x): sha(x) for x in sources}
freeze_text = (qual / "FROZEN.sha256").read_text()
assert sha(qual / "qual.py") + "  qual.py" in freeze_text
meta = json.loads((qual / "qual_meta.json").read_text())
assert meta == json.loads(Path(a.reference_qual_meta).read_text())

# Run the original CPU-only figure script unchanged. Its disposable outputs stay in /tmp.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
cwd = Path.cwd()
with tempfile.TemporaryDirectory(prefix="se-inr-fonts-", dir="/tmp") as scratch:
    try:
        os.chdir(scratch)
        ns = runpy.run_path(str(oldfig / "make_figs.py"))
        generated = json.loads(Path("numbers.json").read_text())
        recovery_figure = plt.figure(1)
        handles, labels = recovery_figure.axes[2].get_legend_handles_labels()
        recovery_figure.axes[2].get_legend().remove()
        recovery_figure.set_size_inches(7.0, 2.3)
        recovery_figure.subplots_adjust(left=.078, right=.985, bottom=.27, top=.86, wspace=.45)
        recovery_figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01),
                               ncol=4, frameon=False, fontsize=6)
        recovery_figure.savefig(out / "fig1_recovery.pdf")
    finally:
        os.chdir(cwd)
plt.close("all")
assert plt.rcParams["pdf.fonttype"] == 42

def flatten(x):
    if isinstance(x, dict):
        return sum((flatten(x[k]) for k in sorted(x)), [])
    if isinstance(x, list):
        return sum((flatten(v) for v in x), [])
    return [float(x)]

server_reference = json.loads((oldfig / "numbers.json").read_text())
assert generated.keys() == server_reference.keys()
sv, gv = np.array(flatten(server_reference)), np.array(flatten(generated))
np.testing.assert_array_equal(gv, sv)

# The repository retains an earlier numbers schema: training profiles at all x2/3/4.
repository_reference = json.loads(Path(a.reference_numbers).read_text())
repository_generated = {}
for key, expected in repository_reference.items():
    if key.startswith("dR_training_x"):
        s = int(key.rsplit("x", 1)[1])
        values = {}
        for text in expected:
            b = float(text)
            values[text] = float(np.mean([
                ns["R"](ns["A1"], ns["T1"], (m, 1, s, b))
                - (1 - ns["EA"][m][0][(m, s, b)]["1.0"] / ns["EA"][m][1][(m, s, b)])
                for m in ("liif", "lte", "sc-inr")]))
        repository_generated[key] = values
    else:
        repository_generated[key] = generated[key]
rv, nv = np.array(flatten(repository_reference)), np.array(flatten(repository_generated))
np.testing.assert_array_equal(nv, rv)

# Figure 3 uses only already selected and saved windows, never selects a new image.
with np.load(qual / "crops.npz", allow_pickle=False) as saved:
    crops = {k: saved[k].copy() for k in saved.files}
order = ["HR", "Bicubic", "EDSR-LIIF", "no cell", "decoder 2x64", "RDN-LIIF", "SwinIR-LIIF"]
assert set(crops) == set(order)
assert all(v.shape == (48, 48, 3) for v in crops.values())
def y(x):
    return (16 + 65.481*x[..., 0] + 128.553*x[..., 1] + 24.966*x[..., 2])/255
recomputed = {}
for k in order[1:]:
    value = float(-10*np.log10(((y(crops[k])-y(crops["HR"]))**2).mean()))
    assert round(value, 2) == meta["psnr_crop"][k], (k, value, meta["psnr_crop"][k])
    recomputed[k] = value
plt.rcdefaults()
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(2, 7, figsize=(7.0, 2.35))
for c, k in enumerate(order):
    ax[0, c].imshow(crops[k], interpolation="nearest")
    ax[0, c].set_title(k if k == "HR" else f'{k}\n{meta["psnr_crop"][k]:.2f} dB', fontsize=6.5)
    if k == "HR":
        ax[1, c].text(.5, .5, "|error| (Y)\n0 to 0.25", ha="center", va="center",
                      fontsize=6.5, transform=ax[1, c].transAxes)
    else:
        ax[1, c].imshow(np.abs(y(crops[k])-y(crops["HR"])), cmap="magma", vmin=0, vmax=.25,
                        interpolation="nearest")
    for row in (0, 1):
        ax[row, c].set_xticks([])
        ax[row, c].set_yticks([])
ax[1, 0].axis("off")
fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.01, wspace=.04, hspace=.04)
fig.savefig(out / "fig3_qualitative.pdf")
plt.close(fig)
after = {str(x): sha(x) for x in sources}
assert before == after, "Source evidence changed during rendering"
fonts = {}
for name in ("fig1_recovery.pdf", "fig3_qualitative.pdf"):
    report = subprocess.check_output(["pdffonts", str(out / name)], text=True)
    assert "Type 3" not in report and "TrueType" in report, report
    fonts[name] = report
receipt = {
    "operation": "TrueType font re-embedding of existing figures, unchanged scientific data",
    "model_calls": 0, "training_runs": 0, "cpu_bicubic_only": True,
    "layout_change": "Figure 1(c) legend moved below all panels, preserving all curves and labels; 7.0 by 2.3 inches, minimum legend font 6 pt. Figure 3 geometry unchanged.",
    "source_files_sha256": before, "sources_unchanged_after_rendering": True,
    "render_script_sha256": sha(__file__),
    "recovery": {
        "server_reference_numeric_values": len(sv), "server_reference_max_abs_difference": float(np.max(abs(gv-sv))),
        "repository_reference_numeric_values": len(rv), "repository_reference_max_abs_difference": float(np.max(abs(nv-rv))),
        "comparison": "Exact float64 array equality, no tolerance; all keys and numeric values checked"},
    "qualitative": {
        "image_and_crop_metadata": meta, "saved_crops_sha256": sha(qual / "crops.npz"),
        "crop_psnr_recomputed": recomputed, "all_displayed_psnr_equal_at_original_two_decimal_precision": True,
        "selection_repeated": False,
        "original_freeze_boundary": "The historical FROZEN.sha256 records qual.py only, not crops.npz or metadata. The final script hash matches; metadata matches the repository. Output hashes are established before this redraw, not claimed as historical pre-registration.",
        "full_image_psnr": "Preserved original metadata; cannot recompute full-image PSNR from saved windows"},
    "pdf_fonts": fonts, "outputs_sha256": {k: sha(out / k) for k in fonts},
    "visual_inspection": "Pending parent inspection of rendered PDFs"
}
(out / "figure-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: receipt[k] for k in ("recovery", "outputs_sha256")}, indent=2))
