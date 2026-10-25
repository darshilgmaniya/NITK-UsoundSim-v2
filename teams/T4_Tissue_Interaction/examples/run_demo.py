"""Minimal end-to-end example:  python examples/run_demo.py [body_part]"""
import sys
import tissue_interaction as ti

bp = sys.argv[1] if len(sys.argv) > 1 else "thyroid"
cfg = ti.config_for_body_part(bp, max_scatterers=15000)
ph = ti.make_phantom(bp)
print(ph.summary())
res = ti.run(ph, cfg, outdir=f"outputs/{bp}")
print("reflected_pressure_input:", res.reflected.shape, "->", res.files["reflected"])
