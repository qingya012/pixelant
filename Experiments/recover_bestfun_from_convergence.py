#!/usr/bin/env python3
"""
Recover the (rough) final bestfun of old BPSO runs directly from the saved
convergence PNGs, without re-running the optimisation.

WHY THIS EXISTS
---------------
The old grid-repeat runs were produced *before* the feed-pixel bug was fixed.
In those runs the fitness stored in bpso_grid_results.csv (`bestfun`) was
re-evaluated from `gbest` WITHOUT forcing the feed pixels (6:7,1)=1, so some
rows are badly under-reported (e.g. 654 or 755 instead of ~806-808).

The convergence curve that was plotted (`ffmin`) was tracked *inside* the loop
WITH the feed override, so its LAST point is the value the optimiser actually
achieved. That curve was only ever saved as a PNG (never numerically), so this
script digitises the PNG to recover that final value.

The convergence curve is monotonic non-decreasing (best-so-far), so its final
value == its highest point == the topmost pixel of the plotted line.

HOW CALIBRATION WORKS
---------------------
MATLAB auto-scales the y-axis per figure, so we need each plot's y-limits.
They are listed in YLIM_MAP below (read off the plots once). The axis box edges
are detected automatically (outermost full-width lines) and mapped linearly to
[ymin, ymax]; the curve's highest pixel is then converted to a fitness value.

DEPENDENCIES: numpy + Pillow only (no OCR, no pandas).
USAGE:        python3 recover_bestfun_from_convergence.py
OUTPUT:       recovered_bestfun.csv written inside each results folder,
              plus a stored-vs-recovered comparison printed to the console.
"""

import os
import re
import csv
import glob
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Per-plot y-axis limits (ymin, ymax), keyed by PNG file name.
# Read directly off each convergence figure. If you add new plots, add their
# limits here; any plot missing from this map is skipped with a warning.
# ---------------------------------------------------------------------------
YLIM_MAP = {
    # bpso_grid_repeats_results (n=500, maxite=50)
    "convergence_n500_maxite50_rep1.png":  (790, 808),
    "convergence_n500_maxite50_rep2.png":  (786, 806),
    "convergence_n500_maxite50_rep3.png":  (770, 810),
    "convergence_n500_maxite50_rep4.png":  (790, 808),
    "convergence_n500_maxite50_rep5.png":  (798, 808),
    "convergence_n500_maxite50_rep6.png":  (785, 810),
    "convergence_n500_maxite50_rep7.png":  (799, 808),
    "convergence_n500_maxite50_rep8.png":  (788, 808),
    "convergence_n500_maxite50_rep9.png":  (792, 808),
    "convergence_n500_maxite50_rep10.png": (792, 804),
    # bpso_grid_repeats_large_results
    "convergence_n1000_maxite100_rep1.png": (792, 810),
    "convergence_n1000_maxite100_rep2.png": (796, 810),
    "convergence_n1000_maxite100_rep3.png": (785, 810),
    "convergence_n2000_maxite50_rep1.png":  (798, 810),
    "convergence_n2000_maxite50_rep2.png":  (792, 810),
    "convergence_n2000_maxite50_rep3.png":  (794, 808),
    # warm-start-experiment/bpso_grid_repeats_good_results (n=1000, maxite=50)
    "convergence_n1000_maxite50_rep1.png":  (792, 808),
    "convergence_n1000_maxite50_rep2.png":  (797, 807),
    "convergence_n1000_maxite50_rep3.png":  (785, 810),
    "convergence_n1000_maxite50_rep4.png":  (780, 810),
    "convergence_n1000_maxite50_rep5.png":  (785, 810),
    "convergence_n1000_maxite50_rep6.png":  (785, 810),
    "convergence_n1000_maxite50_rep7.png":  (800, 808),
    "convergence_n1000_maxite50_rep8.png":  (796, 808),
    "convergence_n1000_maxite50_rep9.png":  (794, 808),
    "convergence_n1000_maxite50_rep10.png": (794, 808),
}

# Results folders to process (relative to this script). Each entry is
# (folder_path, stored_results_filename); the stored file may be a plain CSV or
# an xlsx-with-.csv-extension and is used only for the stored-vs-recovered
# comparison (set to None if there is no stored file).
FOLDERS = [
    ("bpso_grid_repeats_results", "bpso_grid_results.csv"),
    ("bpso_grid_repeats_large_results", "bpso_grid_results.csv"),
    ("warm-start-experiment/bpso_grid_repeats_good_results", "bpso_grid_results_1000.csv"),
]

# Flag a stored value as "likely corrupted by the feed-pixel bug" if it differs
# from the recovered value by more than this many fitness points.
CORRUPTION_TOL = 1.0


def _cluster(positions, gap=4):
    """Group nearby pixel indices and return the mean of each group."""
    if not positions:
        return []
    positions = sorted(positions)
    groups, current = [], [positions[0]]
    for p in positions[1:]:
        if p - current[-1] <= gap:
            current.append(p)
        else:
            groups.append(float(np.mean(current)))
            current = [p]
    groups.append(float(np.mean(current)))
    return groups


def recover_final_bestfun(png_path, ymin, ymax):
    """Digitise the final (highest) point of a monotonic convergence curve.

    Returns (value, diagnostics_dict).
    """
    img = np.asarray(Image.open(png_path).convert("RGB"))
    H, W, _ = img.shape
    r = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    b = img[..., 2].astype(int)
    bright = (r + g + b) / 3.0

    # Axis box + gridlines are near-gray, full-span lines. Outermost horizontal
    # ones = top/bottom axis edges; outermost vertical ones = left/right edges.
    grayish = (bright < 232) & (abs(r - g) < 18) & (abs(g - b) < 18)
    h_lines = _cluster(list(np.where(grayish.sum(1) > 0.6 * W)[0]))
    v_lines = _cluster(list(np.where(grayish.sum(0) > 0.6 * H)[0]))
    if len(h_lines) < 2 or len(v_lines) < 2:
        raise RuntimeError(f"could not detect axis box in {os.path.basename(png_path)}")
    top, bot = h_lines[0], h_lines[-1]
    left, right = v_lines[0], v_lines[-1]

    # The plotted curve ('-k') is a thin dark line (darker than the gray grid,
    # lighter-cored than the very dark axis-tick corners). Restrict to the plot
    # interior and drop a few px near the right edge (tick/gridline artifacts).
    curve = bright < 140
    L = int(round(left)) + 3
    R = int(round(right)) - 8
    T = int(round(top)) + 3
    B = int(round(bot)) - 3
    mask = np.zeros_like(curve)
    mask[T:B, L:R] = True
    curve &= mask

    # Look only in the rightmost 15% of the plot (the converged plateau) and
    # take the highest pixel there = final best-so-far value.
    x_right = int(R - 0.15 * (R - L))
    region = curve.copy()
    region[:, :x_right] = False
    ys = np.where(region.any(1))[0]
    if ys.size == 0:  # fallback: use whole interior
        ys = np.where(curve.any(1))[0]
        if ys.size == 0:
            raise RuntimeError(f"could not detect curve in {os.path.basename(png_path)}")
    y_final = ys.min()

    value = ymax - (y_final - top) / (bot - top) * (ymax - ymin)
    diag = dict(top=round(top, 1), bot=round(bot, 1), left=round(left, 1),
                right=round(right, 1), y_final=int(y_final))
    return float(value), diag


def parse_run(fname):
    """Extract (n, maxite, repeat_id) from convergence_nXXX_maxiteYY_repZZ.png."""
    m = re.search(r"n(\d+)_maxite(\d+)_rep(\d+)", fname)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _read_xlsx(path):
    """Minimal, dependency-free reader for an xlsx file (first worksheet)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{ns}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
    sheet = sorted(n for n in z.namelist()
                   if re.match(r"xl/worksheets/sheet\d+\.xml", n))[0]
    root = ET.fromstring(z.read(sheet))
    rows = []
    for row in root.iter(f"{ns}row"):
        vals = []
        for c in row.findall(f"{ns}c"):
            v = c.find(f"{ns}v")
            x = "" if v is None else v.text
            if c.get("t") == "s":
                x = shared[int(x)]
            vals.append(x)
        rows.append(vals)
    return rows


def load_stored_bestfun(path):
    """Return {(n, maxite, repeat_id): stored_bestfun} from a CSV or xlsx file."""
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as fh:
        is_xlsx = fh.read(2) == b"PK"
    if is_xlsx:
        rows = _read_xlsx(path)
    else:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
    if not rows:
        return {}
    header = [h.strip() for h in rows[0]]
    idx = {name: header.index(name) for name in header}
    out = {}
    for row in rows[1:]:
        if not row or len(row) < len(header):
            continue
        try:
            key = (int(float(row[idx["n"]])),
                   int(float(row[idx["maxite"]])),
                   int(float(row[idx["repeat_id"]])))
            out[key] = float(row[idx["bestfun"]])
        except (ValueError, KeyError):
            continue
    return out


def process_folder(base_dir, folder, stored_name):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"[skip] folder not found: {folder_path}")
        return

    pngs = sorted(glob.glob(os.path.join(folder_path, "convergence_*.png")))
    if not pngs:
        print(f"[skip] no convergence PNGs in {folder}")
        return

    stored = {} if not stored_name else \
        load_stored_bestfun(os.path.join(folder_path, stored_name))

    print(f"\n=== {folder} ===")
    header = f"{'file':<38}{'n':>6}{'maxite':>8}{'rep':>5}" \
             f"{'stored':>11}{'recovered':>11}{'delta':>9}  flag"
    print(header)
    print("-" * len(header))

    out_rows = []
    for png in pngs:
        fname = os.path.basename(png)
        if fname not in YLIM_MAP:
            print(f"{fname:<38}  [no YLIM entry - add (ymin,ymax) to YLIM_MAP to include]")
            continue
        ymin, ymax = YLIM_MAP[fname]
        try:
            value, _ = recover_final_bestfun(png, ymin, ymax)
        except RuntimeError as err:
            print(f"{fname:<38}  [ERROR] {err}")
            continue

        n, maxite, rep = parse_run(fname)
        stored_val = stored.get((n, maxite, rep))
        if stored_val is None:
            delta_str, flag, stored_str = "", "", ""
        else:
            delta = stored_val - value
            delta_str = f"{delta:+.2f}"
            stored_str = f"{stored_val:.2f}"
            flag = "CORRUPTED" if abs(delta) > CORRUPTION_TOL else "ok"

        print(f"{fname:<38}{str(n):>6}{str(maxite):>8}{str(rep):>5}"
              f"{stored_str:>11}{value:>11.2f}{delta_str:>9}  {flag}")

        out_rows.append({
            "file": fname, "n": n, "maxite": maxite, "repeat_id": rep,
            "ymin": ymin, "ymax": ymax,
            "stored_bestfun": "" if stored_val is None else round(stored_val, 4),
            "recovered_final_bestfun": round(value, 4),
            "delta_stored_minus_recovered": "" if stored_val is None else round(stored_val - value, 4),
            "likely_corrupted": "" if stored_val is None else int(abs(stored_val - value) > CORRUPTION_TOL),
        })

    if out_rows:
        out_csv = os.path.join(folder_path, "recovered_bestfun.csv")
        fields = ["file", "n", "maxite", "repeat_id", "ymin", "ymax",
                  "stored_bestfun", "recovered_final_bestfun",
                  "delta_stored_minus_recovered", "likely_corrupted"]
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"-> wrote {out_csv}  ({len(out_rows)} rows)")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder, stored_name in FOLDERS:
        process_folder(base_dir, folder, stored_name)
    print("\nNote: recovered values are digitised from the plots and are")
    print("accurate to ~0.1 fitness points (fine for a rough correction).")


if __name__ == "__main__":
    main()
