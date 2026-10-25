"""Runs the Receiver BF team's ORIGINAL ``liver_echo_simulation.py`` (unchanged) against this module's
``das_beamform``.  The team's script does its own scatterer/echo model with 48 elements; it is kept as the
team's stand-alone demo.  The integrated pipeline uses the relay's echo data instead (see integration/02).

    python demos/run_liver_demo.py            # opens the figure
    python demos/run_liver_demo.py --save out.png   # headless
"""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))      # makes `receive_beamforming` importable
sys.path.insert(0, HERE)

if "--save" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = sys.argv[sys.argv.index("--save") + 1]
    plt.show = lambda *a, **k: plt.gcf().savefig(out, dpi=130, bbox_inches="tight") or print("saved", out)

runpy.run_path(os.path.join(HERE, "liver_echo_simulation.py"), run_name="__main__")
