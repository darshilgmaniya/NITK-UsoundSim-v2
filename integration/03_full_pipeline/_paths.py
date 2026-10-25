"""Makes every module and the earlier integration steps importable without installing anything (repo-relative)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(ROOT, "teams", "T3_Acoustic_Propagation", "code"),
          os.path.join(ROOT, "teams", "T4_Tissue_Interaction", "src"),
          os.path.join(ROOT, "teams", "T5_RX_Beamforming", "src"),
          os.path.join(ROOT, "teams", "T6_Image_Reconstruction", "src"),
          os.path.join(ROOT, "teams", "T7_B_mode", "src"),
          os.path.join(ROOT, "teams", "T8_Post_Image_Processing", "src"),
          os.path.join(ROOT, "integration", "01_ap_ti_relay"),
          os.path.join(ROOT, "integration", "02_full_relay"),
          os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
