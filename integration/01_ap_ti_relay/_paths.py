"""Makes the two modules importable without installing anything (repo-relative paths)."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (os.path.join(ROOT, "teams", "T3_Acoustic_Propagation", "code"),
          os.path.join(ROOT, "teams", "T4_Tissue_Interaction", "src"),
          os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
