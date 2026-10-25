"""Builds notebooks/04_tissue.ipynb (Colab friendly)."""
import nbformat as nbf, os
nb = nbf.v4.new_notebook()
C = lambda s: nbf.v4.new_code_cell(s.strip("\n"))
M = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
nb.cells = [
M("""# 04_tissue - Tissue Interaction (Subtask 04)
Phantom (supplied by the user) -> acoustic maps -> scatterers -> **reflected pressure per scatterer** + 2-D grayscale images.

Hand-off: `tissue_scatterers.npz` -> Acoustic Propagation `forward_propagation()`; `reflected_pressure_input` -> `return_propagation()`.
See `docs/INTERFACE_CONTRACT.md`."""),
C("""
# --- setup (Colab): clone + install; skipped automatically if the package is already importable
try:
    import tissue_interaction
except ImportError:
    !git clone https://github.com/<your-username>/NITK-UsoundSim-Tissue-Interaction.git
    %cd NITK-UsoundSim-Tissue-Interaction
    !pip install -e . -q
    import tissue_interaction
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
import tissue_interaction as ti
print("tissue_interaction", ti.__version__)
"""),
M("## 1. Choose body part and phantom source"),
C("""
BODY_PART      = "thyroid"      # see ti.list_body_parts()
USER_POINTS    = None           # phantom as POINTS: CSV x_mm,z_mm,tissue  or  x_mm,z_mm,amp  (preferred input)
USER_UNITS     = "mm"           # unit of the x/z columns
USER_PHANTOM   = None           # or an image, e.g. "my_phantom.png"  (None + no points -> built-in synthetic phantom)
USER_MAPPING   = None           # e.g. "my_mapping.json" (labels / colors / ranges / classes)
SPACING_MM     = None           # mm per pixel of YOUR phantom, e.g. 0.2   (mandatory for a user phantom)
MODE           = "auto"         # auto | labels | color | intensity
MAX_SCATTERERS = 15000          # diffuse-scatterer budget (raise for finer speckle; costs memory)

print(ti.list_body_parts())
cfg = ti.config_for_body_part(BODY_PART, max_scatterers=MAX_SCATTERERS)
# cfg.f0_hz = 5e6; cfg.probe.geometry = "sector"; cfg.probe.apex_z_m = -0.05   # <- overwrite with the Transducer team's values
if USER_POINTS:
    pc = ti.load_points(USER_POINTS, units=USER_UNITS); print(pc.summary())
    ph = ti.tissue_cloud_to_phantom(pc) if pc.mode == "tissue_cloud" else None
elif USER_PHANTOM:
    ph = ti.load_phantom(USER_PHANTOM, mode=MODE, mapping=USER_MAPPING, spacing_mm=SPACING_MM, body_part=BODY_PART)
else:
    ph = ti.make_phantom(BODY_PART)
if ph is not None: print(ph.summary())
"""),
C("""
if ph is not None:
    plt.figure(figsize=(5, 5))
    plt.imshow(ti.pipeline.property_panels(ph, cfg)["tissue_labels"][0], cmap="gray",
               extent=[v * 1e3 for v in ph.extent_m()]); plt.title("tissue labels (grayscale)")
    plt.xlabel("x [mm]"); plt.ylabel("depth [mm]"); plt.show()
"""),
M("## 2. Stage 1 - scatterer list (give `tissue_scatterers.npz` to Acoustic Propagation)"),
C("""
if ph is not None:
    scat, files = ti.run_stage1(ph, cfg, outdir="outputs/handoff")
else:                                   # scatterer-list input
    scat = ti.scatterers_from_points(pc, cfg); print("scatterer list used as given")
print(scat.summary())
print("scatterer_xyz[:3] (m):\\n", scat.xyz[:3])
"""),
M("""## 3. Stage 2 - reflected pressure
Set `AP_INCIDENT_FILE` to the Acoustic Propagation output (`incident_pressure` `(n_scatterers, n_samples)` + `t_axis`, rows in stage-1 order).
With `None` a mock incident pulse is used so the module can be tested alone."""),
C("""
AP_INCIDENT_FILE = None
if AP_INCIDENT_FILE:
    from tissue_interaction.outputs import load_incident
    incident, t_axis = load_incident(AP_INCIDENT_FILE)
else:
    incident = t_axis = None
if USER_POINTS:
    res = ti.run_points(pc, cfg, incident=incident, t_axis=t_axis, outdir="outputs/tissue_run")
else:
    res = ti.run(ph, cfg, incident, t_axis, outdir="outputs/tissue_run")
ph = res.phantom
print("reflected_pressure_input:", res.reflected.shape, "| channel data for Receiver BF:", res.channel_rf.shape)
"""),
M("## 4. 2-D grayscale outputs"),
C("""
n = len(res.panels); cols = 4; rows = int(np.ceil(n / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows)); axs = np.atleast_2d(axs)
ext = [v * 1e3 for v in ph.extent_m()]
for ax in axs.ravel(): ax.axis("off")
for ax, (k, (img, lab)) in zip(axs.ravel(), res.panels.items()):
    ax.axis("on"); ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext); ax.set_title(lab, fontsize=8)
plt.tight_layout(); plt.show()
"""),
M("## 4b. Channel data = the output for the Receiver Beamforming team"),
C("""
plt.figure(figsize=(6, 5))
plt.imshow(ti.channel_image(res), cmap="gray", aspect="auto",
           extent=[res.element_xz[0, 0] * 1e3, res.element_xz[-1, 0] * 1e3, res.t_rx[-1] * 1e6, res.t_rx[0] * 1e6])
plt.xlabel("element x [mm]"); plt.ylabel("time [us]"); plt.title("received_pressure envelope (n_elements x n_samples)"); plt.show()
"""),
M("## 5. Inspect individual reflected waveforms (one interface, one speckle scatterer)"),
C("""
kind = res.scatterers.kind
i_if = int(np.where(kind == 1)[0][np.argmax(np.abs(res.scatterers.amp[kind == 1]))])
i_df = int(np.where(kind == 0)[0][0])
fig, axs = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
for ax, i in zip(axs, (i_if, i_df)):
    ax.plot(res.t_axis * 1e6, res.reflected[i]); ax.set_title(res.scatterers.row_labels()[i], fontsize=9)
axs[1].set_xlabel("time [us]"); plt.tight_layout(); plt.show()
"""),
M("""## 6. Files to hand over
* `outputs/tissue_run/receiver_bf_input.npz` -> `received_pressure (n_elements, n_samples)` for **Receiver Beamforming**
* `outputs/tissue_run/tissue_interaction_output.npz` -> `reflected_pressure_input` (only if Acoustic Propagation's `return_propagation()` stays in the chain)
* `outputs/handoff/tissue_scatterers.npz`, `tissue_property_maps.npz` -> Acoustic Propagation
* `outputs/tissue_run/images/*.png`, `tissue_interaction_montage.png` -> report / slides"""),
C("""
# Colab: download everything as one zip
import shutil
shutil.make_archive("tissue_interaction_results", "zip", "outputs")
try:
    from google.colab import files; files.download("tissue_interaction_results.zip")
except ImportError:
    print("zip written: tissue_interaction_results.zip")
"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "notebooks"), exist_ok=True)
nbf.write(nb, os.path.join(os.path.dirname(__file__), "..", "notebooks", "04_tissue.ipynb"))
print("notebook written")
