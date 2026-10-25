"""
upload.py - give the phantom file while the code is running
===========================================================

Three ways to hand over a phantom without editing any code:

    Notebook (VS Code / Jupyter / Colab)    uploader = show_upload_widget()     # shows an "Upload" button
                                            up = collect_upload(uploader)       # reads what you chose
    Terminal                                python run_pipeline.py --upload      # opens a file dialog
                                            python run_pipeline.py               # asks what to do
    Anywhere                                resolve_inputs(["phantom.png", "map.json"])

What can be uploaded (select the phantom and, for an image, its mapping file together):
    image  .png .jpg .jpeg .bmp .tif .tiff   + a mapping .json (label / intensity / colour mapping)
    points .csv                              x_mm,z_mm,amp   or   x_mm,z_mm,tissue
    array  .npy                              (N, 3) array  x_mm, z_mm, amp   (converted to a CSV)

Uploaded files are saved in the folder ``uploads/`` (created next to where you run the code).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
POINTS_EXT = (".csv", ".txt")
ARRAY_EXT = (".npy",)
ACCEPT = ",".join(IMAGE_EXT + POINTS_EXT + ARRAY_EXT + (".json",))


@dataclass
class PhantomInput:
    source: str                          # path of the image / CSV (arrays are converted to CSV)
    mapping: Optional[str] = None        # mapping .json for an image
    files: List[str] = field(default_factory=list)
    is_image: bool = False

    def __str__(self):
        m = f" + mapping {os.path.basename(self.mapping)}" if self.mapping else ""
        return f"{os.path.basename(self.source)}{m}"


def in_colab() -> bool:
    return "google.colab" in sys.modules


# --------------------------------------------------------------------------- resolve
def _ext(p: str) -> str:
    return os.path.splitext(p)[1].lower()


def _sibling_mapping(img: str) -> Optional[str]:
    stem, d = os.path.splitext(img)[0], os.path.dirname(img)
    base = os.path.basename(stem)
    for cand in (stem + "_mapping.json", stem + ".json", stem + "_map.json",
                 os.path.join(d, base.replace("_labels", "") + "_mapping.json")):
        if os.path.isfile(cand):
            return cand
    return None


def resolve_inputs(paths: Iterable[str]) -> PhantomInput:
    """Sort the chosen files into phantom + mapping and check they fit together."""
    paths = [os.path.abspath(str(p)) for p in paths if p]
    if not paths:
        raise ValueError("no file selected")
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    maps = [p for p in paths if _ext(p) == ".json"]
    srcs = [p for p in paths if _ext(p) != ".json"]
    bad = [p for p in srcs if _ext(p) not in IMAGE_EXT + POINTS_EXT + ARRAY_EXT]
    if bad:
        raise ValueError(f"unsupported file type: {os.path.basename(bad[0])} (use {', '.join(IMAGE_EXT[:2] + POINTS_EXT[:1] + ARRAY_EXT)} and a mapping .json)")
    if len(srcs) != 1:
        raise ValueError("choose exactly one phantom file (image, .csv or .npy) and, for an image, its mapping .json; "
                         f"got {len(srcs)} phantom files")
    src = srcs[0]
    mapping = None
    if maps:
        mapping = next((m for m in maps if "map" in os.path.basename(m).lower()), maps[0])
    if _ext(src) in ARRAY_EXT:
        arr = np.load(src)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f".npy must have shape (N, 3) = x_mm, z_mm, amp; got {arr.shape}")
        csv = os.path.splitext(src)[0] + "_points.csv"
        np.savetxt(csv, arr, delimiter=",", header="x_mm,z_mm,amp", comments="")
        src = csv
    is_image = _ext(src) in IMAGE_EXT
    if is_image and mapping is None:
        mapping = _sibling_mapping(src)
        if mapping is None:
            raise ValueError("an image phantom needs its mapping file: select the mapping .json together with the image "
                             "(or keep it next to the image as <name>_mapping.json)")
    return PhantomInput(src, mapping, paths, is_image)


# --------------------------------------------------------------------------- notebook
def save_uploaded(value, upload_dir: str = "uploads") -> List[str]:
    """Write what an upload widget (ipywidgets 7 or 8) or Colab's files.upload() returned to ``upload_dir``."""
    os.makedirs(upload_dir, exist_ok=True)
    items: Dict[str, bytes] = {}
    if isinstance(value, dict):                       # ipywidgets 7: name -> {metadata, content};  Colab: name -> bytes
        for name, v in value.items():
            items[name] = bytes(v["content"]) if isinstance(v, dict) else bytes(v)
    else:                                             # ipywidgets 8: tuple of {name, content, ...}
        for v in value:
            items[v["name"]] = bytes(v["content"])
    out = []
    for name, data in items.items():
        path = os.path.join(upload_dir, os.path.basename(name))
        with open(path, "wb") as f:
            f.write(data)
        out.append(os.path.abspath(path))
    return out


class _Done:
    """Stands in for a widget after Colab's blocking upload."""
    def __init__(self, value):
        self.value = value


class _Paths:
    """Stands in for a widget after a native file dialog: files already on disk."""
    def __init__(self, paths):
        self.paths = list(paths)
        self.value = bool(self.paths)


def native_dialog(title="Select the phantom (image / .csv / .npy) and its mapping .json"):
    """Open the operating system's own file dialog (blocking). Returns a list of paths, [] when cancelled,
    or None when no dialog could be shown (then the Upload button is used instead)."""
    import subprocess
    try:
        if sys.platform == "darwin":                  # macOS: AppleScript dialog, works from a notebook kernel
            script = ('set fs to choose file with prompt "%s" with multiple selections allowed\n'
                      'set out to ""\nrepeat with f in fs\nset out to out & POSIX path of f & linefeed\n'
                      'end repeat\nreturn out' % title.replace('"', "'"))
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if r.returncode != 0:
                return [] if "User canceled" in r.stderr else None
            return [x for x in r.stdout.splitlines() if x.strip()]
        code = ("import tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk();r.withdraw()\n"
                "r.attributes('-topmost',True)\n"
                "print('\\n'.join(filedialog.askopenfilenames(title=%r)))" % title)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        return [x for x in r.stdout.splitlines() if x.strip()]
    except Exception:
        return None


def show_upload_widget():
    """Show an Upload button (Colab: open the upload dialog right away). Returns an object for ``collect_upload``."""
    if in_colab():
        from google.colab import files
        print("Choose your phantom file (image / .csv / .npy) and its mapping .json (Cancel = use the default).")
        return _Done(files.upload())
    print("A file dialog opens now: select the phantom (image / .csv / .npy) and its mapping .json together "
          "(Cancel = use the default phantom).")
    picked = native_dialog()
    if picked is not None:
        print("selected:", [os.path.basename(x) for x in picked] or "nothing (default phantom will be used)")
        return _Paths(picked)
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        print("ipywidgets is not installed (pip install ipywidgets); set SOURCE / MAPPING by hand instead.")
        return None
    up = w.FileUpload(accept=ACCEPT, multiple=True, description="Upload phantom (+ mapping)",
                      layout=w.Layout(width="320px"))
    display(up)
    print("Click the button, select the phantom (image / .csv / .npy) and its mapping .json together, then run the next cell.")
    return up


def collect_upload(uploader, upload_dir: str = "uploads") -> Optional[PhantomInput]:
    """Read the files chosen in the widget. Returns None when nothing was uploaded."""
    if uploader is None or not getattr(uploader, "value", None):
        return None
    if isinstance(uploader, _Paths):
        return resolve_inputs(uploader.paths)
    return resolve_inputs(save_uploaded(uploader.value, upload_dir))


# --------------------------------------------------------------------------- terminal
def pick_files_dialog(title: str = "Select the phantom (image / .csv / .npy) and its mapping .json") -> List[str]:
    """Native file dialog (tkinter). Falls back to typing the paths when no display is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        files = filedialog.askopenfilenames(title=title, filetypes=[
            ("Phantom + mapping", "*" + " *".join(IMAGE_EXT + POINTS_EXT + ARRAY_EXT + (".json",))), ("All files", "*.*")])
        root.destroy()
        return list(files)
    except Exception:
        raw = input("Path(s) of the phantom and its mapping .json, separated by a space (drag the files here): ")
        import shlex
        return shlex.split(raw)


SAMPLES = {"sample_abdomen": "abdomen", "sample_thyroid": "thyroid"}


def sample_phantom(name: str) -> PhantomInput:
    """The label-map phantoms shipped with T4 (``sample_abdomen`` / ``sample_thyroid``)."""
    import _paths  # noqa: F401
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    d = os.path.join(root, "teams", "T4_Tissue_Interaction", "sample_data")
    k = SAMPLES[name]
    return resolve_inputs([os.path.join(d, f"phantom_{k}_labels.png"), os.path.join(d, f"phantom_{k}_mapping.json")])


def ask_interactively() -> PhantomInput | str:
    """Terminal menu used when run_pipeline.py is started without --source. Returns a PhantomInput or a preset/points name."""
    if not sys.stdin.isatty():
        raise SystemExit("no input given: use --source <preset|file> [--mapping map.json], or --upload to choose a file")
    print("\nWhich phantom do you want to run?\n"
          "  [1] Upload / choose my own file (a file dialog opens)\n"
          "  [2] Sample abdomen phantom (about 30 s)\n"
          "  [3] Nine-wire demo (about 5 s)\n"
          "  [4] Type a preset name (abdomen, kidney, thyroid, breast, ...)")
    while True:
        c = input("Choice 1-4: ").strip()
        if c == "1":
            return resolve_inputs(pick_files_dialog())
        if c == "2":
            return sample_phantom("sample_abdomen")
        if c == "3":
            return "points"
        if c == "4":
            return input("Preset name: ").strip()
        print("please type 1, 2, 3 or 4")
