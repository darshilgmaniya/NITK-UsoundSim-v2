"""Reproduces every number in docs/08_VERIFICATION_REPORT.md and docs/09_PERFORMANCE.md.
    python verification/run_verification.py            -> docs/verification/results.json + figures
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _paths  # noqa: F401
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nitk_usoundsim import acoustic_propagation as ap, config
from usoundsim_relay import run_relay, RelayConfig
from usoundsim_relay.verify import das_image, locate

OUT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "docs", "verification"))
os.makedirs(OUT, exist_ok=True)
R = {}
GRID = [[x, z, 1.0] for z in (10, 20, 30) for x in (-6.0, 0.0, 6.0)]


def fwhm(profile, axis, level=0.5):
    p = profile / profile.max(); above = np.where(p >= level)[0]
    i0, i1 = above[0], above[-1]
    def cross(a, b):
        return axis[a] + (level - p[a]) / (p[b] - p[a]) * (axis[b] - axis[a])
    lo = cross(i0 - 1, i0) if i0 > 0 else axis[i0]
    hi = cross(i1, i1 + 1) if i1 < p.size - 1 else axis[i1]
    return hi - lo


# ---- V1 batch equivalence ------------------------------------------------------------------
rng = np.random.default_rng(1)
n = 40; sx = rng.uniform(-8e-3, 8e-3, n); sz = rng.uniform(5e-3, 35e-3, n); amp = rng.uniform(0.2, 1.5, n)
ex = ap.create_linear_array()
pt, _ = ap.generate_pulse()
dmax = np.sqrt((np.abs(sx)[:, None] + 10e-3) ** 2 + sz[:, None] ** 2).max()
t = np.arange(-pt[-1], 2 * dmax / config.C * 1.15 + pt[-1], 1 / config.FS)
ref_f = np.stack([ap.forward_propagation(dict(x=sx[i], z=sz[i], amp=1), ex, t)[0] for i in range(n)])
new_f = ap.forward_propagation_many(sx, sz, ex, t, dtype=np.float64)
inc = ref_f * amp[:, None]
ref_r = sum(ap.return_propagation(inc[i], t, dict(x=sx[i], z=sz[i], amp=1), ex)[0] for i in range(n))
new_r = ap.return_propagation_many(inc, t, sx, sz, ex)
R["V1_batch_equivalence"] = dict(
    n_scatterers=n, n_elements=int(ex.size),
    forward_max_abs_error=float(np.abs(new_f - ref_f).max()), forward_peak=float(np.abs(ref_f).max()),
    forward_rel_error=float(np.abs(new_f - ref_f).max() / np.abs(ref_f).max()),
    return_max_abs_error=float(np.abs(new_r - ref_r).max()), return_peak=float(np.abs(ref_r).max()),
    return_rel_error=float(np.abs(new_r - ref_r).max() / np.abs(ref_r).max()))

# ---- V2 focus accuracy + V3 resolution, 64 vs 128 --------------------------------------------
res_by_n = {}
for ne in (64, 128):
    r = run_relay(np.array(GRID), RelayConfig(n_elements=ne, also_ti_channel=False), units="mm", verbose=False)
    res_by_n[ne] = r
    err = []
    for x, z, _ in GRID:
        xb, zb = locate(r.received_pressure, r.t_axis, r.element_x, x * 1e-3, z * 1e-3)
        err.append((xb - x * 1e-3, zb - z * 1e-3))
    err = np.array(err) * 1e6
    R[f"V2_focus_error_{ne}el"] = dict(
        x_err_um=err[:, 0].round(1).tolist(), z_err_um=err[:, 1].round(1).tolist(),
        max_abs_x_um=float(np.abs(err[:, 0]).max()), max_abs_z_um=float(np.abs(err[:, 1]).max()),
        rms_total_um=float(np.sqrt((err ** 2).sum(1).mean())))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
R["V3_resolution"] = {}
for k, z_mm in enumerate((10, 20, 30)):
    for ne, col in ((64, "tab:orange"), (128, "tab:blue")):
        r = res_by_n[ne]
        xs = np.arange(-1.5e-3, 1.5e-3, 0.01e-3); zs = np.arange(z_mm * 1e-3 - 1.5e-3, z_mm * 1e-3 + 1.5e-3, 0.01e-3)
        img = das_image(r.received_pressure, r.t_axis, r.element_x, xs, zs)
        iz, ix = np.unravel_index(img.argmax(), img.shape)
        lat = fwhm(img[iz], xs); axl = fwhm(img[:, ix], zs)
        R["V3_resolution"].setdefault(f"z{z_mm}mm", {})[f"{ne}el"] = dict(lateral_fwhm_mm=lat * 1e3, axial_fwhm_mm=axl * 1e3, peak_rel=float(img.max()))
        ax[k].plot(xs * 1e3, img[iz] / img.max(), c=col, label=f"{ne} el  (FWHM {lat*1e3:.2f} mm)")
    ax[k].axhline(.5, c="gray", ls=":"); ax[k].set_title(f"lateral profile, target at z = {z_mm} mm"); ax[k].set_xlabel("x [mm]"); ax[k].legend()
ax[0].set_ylabel("normalised envelope"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "resolution_64_vs_128.png"), dpi=130); plt.close(fig)
for z_mm in (10, 20, 30):
    d = R["V3_resolution"][f"z{z_mm}mm"]
    d["lateral_ratio_64_over_128"] = d["64el"]["lateral_fwhm_mm"] / d["128el"]["lateral_fwhm_mm"]

# ---- V4 linearity ----------------------------------------------------------------------------
base = run_relay(np.array(GRID), RelayConfig(also_ti_channel=False), units="mm", verbose=False).received_pressure
lin = {}
for g in (0.1, 10.0):
    p = np.array(GRID); p[:, 2] *= g
    y = run_relay(p, RelayConfig(also_ti_channel=False), units="mm", verbose=False).received_pressure
    lin[str(g)] = float(np.abs(y - g * base).max() / np.abs(g * base).max())
R["V4_linearity_max_rel_dev"] = lin

# ---- V5 attenuation ownership ---------------------------------------------------------------
pts1 = np.array([[0.0, 40.0, 1.0]])
def peak(owner, a0):
    return float(np.abs(run_relay(pts1, RelayConfig(attenuation_owner=owner, alpha0=a0, also_ti_channel=False), units="mm", verbose=False).received_pressure).max())
R["V5_attenuation_ownership"] = dict(
    owner_ap_ratio_alpha0_0p5_over_0=peak("ap", 0.5) / peak("ap", 0.0),
    owner_tissue_ratio_alpha0_0p5_over_0=peak("tissue", 0.5) / peak("tissue", 0.0),
    expected_round_trip_20dB_ratio=10 ** (-2 * 0.5 * 5 * 4 / 20))

# ---- V6 AP baton vs TI stage-3 alternative ---------------------------------------------------
r = run_relay(np.array(GRID), RelayConfig(), units="mm", verbose=False)
xs = np.linspace(-9e-3, 9e-3, 220); zs = np.linspace(6e-3, 34e-3, 220)
a = das_image(r.received_pressure, r.t_axis, r.element_x, xs, zs); b = das_image(r.ti_channel, r.ti_channel_t, r.element_x, xs, zs)
a /= a.max(); b /= b.max()
corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
dpos = []
for x, z, _ in GRID:
    pa = locate(r.received_pressure, r.t_axis, r.element_x, x * 1e-3, z * 1e-3); pb = locate(r.ti_channel, r.ti_channel_t, r.element_x, x * 1e-3, z * 1e-3)
    dpos.append(float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]) * 1e6))
R["V6_ap_vs_ti_stage3"] = dict(image_correlation=corr, max_peak_position_difference_um=max(dpos))

# ---- V7 performance (each in a fresh process for clean peak memory) ----------------------------
code = ("import sys,resource,json;sys.path.insert(0,%r);import _paths;from usoundsim_relay import run_relay,RelayConfig;"
        "r=run_relay('generic',RelayConfig(max_scatterers=%d,n_elements=%d,also_ti_channel=False),verbose=False);"
        "print(json.dumps(dict(n=int(r.scatterers.n),T=int(r.t_axis.size),t=r.timings,rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))")
perf = []
for ms, ne in ((500, 128), (2000, 128), (8000, 128), (2000, 64)):
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code % (os.path.dirname(HERE), ms, ne)], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    d = json.loads(out); d.update(max_scatterers=ms, n_elements=ne); perf.append(d)
R["V7_performance"] = perf
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
p128 = [d for d in perf if d["n_elements"] == 128]
ns = [d["n"] for d in p128]
for key, lab in (("runner2_ap_part1", "AP part 1"), ("runner3_ti_stage2", "TI stage 2"), ("runner4_ap_part2", "AP part 2")):
    ax[0].plot(ns, [d["t"][key] for d in p128], "o-", label=lab)
ax[0].set_xlabel("number of scatterers N"); ax[0].set_ylabel("seconds"); ax[0].set_title("time per runner (128 elements)"); ax[0].legend()
ax[1].plot(ns, [d["rss"] for d in p128], "s-", c="k"); ax[1].set_xlabel("number of scatterers N"); ax[1].set_ylabel("peak RSS [MB]"); ax[1].set_title("peak memory")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "performance_scaling.png"), dpi=130); plt.close(fig)

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(json.dumps(R, indent=1)[:6000])
