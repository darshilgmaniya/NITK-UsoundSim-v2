"""
run_team3.py  -  Team 3: Acoustic Propagation  (NITK-UsoundSim)
================================================================

ONE command runs the whole Team 3 demo:

    python3 run_team3.py        (macOS / Linux)
    python  run_team3.py        (Windows)

What it does (the team's code in code/nitk_usoundsim/ is NOT changed;
this script only imports it and calls its functions):

  STEP 1  Run the team's 22 unit tests (test_acoustic_propagation.py).
  STEP 2  Run the team's own demo (128 elements, 5 MHz, 3 scatterers)
          and save its two figures + two .npz data files into output/.
  STEP 3  Second run with the project's final probe (Philips L12-4 model:
          128 elements, 0.30 mm pitch, 8 MHz) and one point scatterer at
          (0, 20) mm, done only by passing arguments to the team's functions.
  STEP 4  Attenuation vs depth at 4, 8 and 12 MHz (team's calculate_attenuation).

Everything is saved in output/ (plots, data, console log, console screenshot).
"""

import io
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True          # keep code/ clean (no __pycache__)

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(HERE, "code")
INPUT_FILE = os.path.join(HERE, "input", "team3_run_parameters.json")
OUT_DIR = os.path.join(HERE, "output")
sys.path.insert(0, CODE_DIR)            # makes the team's package importable

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402

from nitk_usoundsim import config                          # noqa: E402  (team code)
from nitk_usoundsim import acoustic_propagation as ap      # noqa: E402  (team code)
from nitk_usoundsim import demo_acoustic_propagation as demo  # noqa: E402  (team code)


# ----------------------------------------------------------------------
# Small helpers (printing to screen AND to a log file)
# ----------------------------------------------------------------------
class Tee:
    """Send printed text to the real console and keep a copy for the log."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, text):
        self.stream.write(text)
        self.buffer.write(text)

    def flush(self):
        self.stream.flush()


def banner(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def rel(path):
    return os.path.relpath(path, HERE)


def console_to_png(text, out_png, title="Terminal", width_chars=100, max_lines=150):
    """Draw the captured console text as a terminal-style screenshot PNG."""
    lines = []
    for line in text.rstrip("\n").expandtabs(4).splitlines() or [""]:
        while len(line) > width_chars:
            lines.append(line[:width_chars])
            line = "  -> " + line[width_chars:]
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [f"... ({len(lines) - max_lines + 1} more lines, see console_log.txt)"]
    h = 0.62 + 0.165 * len(lines)
    fig = plt.figure(figsize=(0.083 * width_chars + 0.4, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.012, 1 - 0.28 / h, "● ● ●   " + title, color="#9a9a9a",
             fontsize=9, family="DejaVu Sans Mono", va="center")
    for i, ln in enumerate(lines):
        fig.text(0.012, 1 - (0.55 + 0.165 * i) / h, ln, color="#e6e6e6",
                 fontsize=8.6, family="DejaVu Sans Mono", va="top")
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------
# STEP 1: the team's unit tests
# ----------------------------------------------------------------------
def step1_unit_tests():
    banner("STEP 1: Team 3 unit tests (test_acoustic_propagation.py)")
    print("Command used: python -m unittest nitk_usoundsim.test_acoustic_propagation -v")
    print("(run inside the code/ folder)\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "nitk_usoundsim.test_acoustic_propagation", "-v"],
        cwd=CODE_DIR, env=env, capture_output=True, text=True)
    text = (proc.stdout + proc.stderr).strip()
    with open(os.path.join(OUT_DIR, "unit_tests_full_output.txt"), "w") as fh:
        fh.write(text + "\n")
    # Print a short line per test: "ok  Class.test_name" (full text is saved)
    for line in text.splitlines():
        if " ... " in line and "(" in line:
            name, _, verdict = line.partition(" ... ")
            full = name[name.find("(") + 1:name.rfind(")")]
            short = ".".join(full.split(".")[-2:])
            print(f"  {verdict.strip():5s} {short}")
        elif line.startswith("Ran ") or line.strip() == "OK" or line.startswith("FAILED"):
            print(line)
    print("  (full unittest text saved in output/unit_tests_full_output.txt)")
    n_ran = 0
    for line in text.splitlines():
        if line.startswith("Ran ") and " test" in line:
            n_ran = int(line.split()[1])
    n_ok = sum(1 for line in text.splitlines() if line.rstrip().endswith("... ok"))
    status = "ALL PASSED" if proc.returncode == 0 else "SOME TESTS FAILED"
    print(f"\nRESULT: {n_ok} of {n_ran} tests passed  ->  {status}")
    return n_ok, n_ran


# ----------------------------------------------------------------------
# STEP 2: the team's own demo (128 elements, 5 MHz, 3 scatterers)
# ----------------------------------------------------------------------
def step2_team_demo(params):
    banner("STEP 2: Team demo - 128 elements, 5 MHz, lambda/2 pitch (config.py)")
    p1 = params["run1_team_design"]
    same = (config.N_ELEMENTS == p1["n_elements"] and config.F0 == p1["f0_hz"]
            and config.FS == p1["fs_hz"] and config.C == p1["c_m_per_s"]
            and config.ALPHA_0 == p1["alpha0_db_per_mhz_per_cm"]
            and config.PHANTOM_POINTS == p1["scatterers"])
    print(f"Input check: values in input JSON match config.py -> {same}")
    print(f"  Elements      : {config.N_ELEMENTS}")
    print(f"  Centre freq   : {config.F0 / 1e6:.1f} MHz")
    print(f"  Sampling freq : {config.FS / 1e6:.1f} MHz")
    print(f"  Speed of sound: {config.C:.0f} m/s")
    print(f"  Wavelength    : {config.LAMBDA * 1e3:.4f} mm")
    print(f"  Pitch (lam/2) : {config.PITCH * 1e3:.4f} mm")
    print(f"  Attenuation   : {config.ALPHA_0} dB/MHz/cm")
    print(f"  Scatterers    : " + ", ".join(
        f"({p['x'] * 1e3:.0f}, {p['z'] * 1e3:.0f}) mm" for p in config.PHANTOM_POINTS))

    # The team's demo normally saves next to its own source file. Here we pass
    # out_dir=output/ to its functions so the original folder stays untouched.
    f1, f2, result = demo.make_figures(out_dir=OUT_DIR)
    part1, part2 = demo.save_part_outputs(result, out_dir=OUT_DIR)
    print()
    print("Saved figure:", rel(f1))
    print("Saved figure:", rel(f2))
    print("Saved PART 1 (forward propagation) data:", rel(part1))
    print("Saved PART 2 (return propagation) data: ", rel(part2))

    t = result["t_axis"]
    print()
    print(f"Time axis     : {t.size} samples, {t[0] * 1e6:.3f} to {t[-1] * 1e6:.3f} us")
    print(f"PART 1 incident_pressure shape : {result['incident_pressure'].shape}"
          "  (scatterers, time)")
    print(f"PART 2 received_pressure shape : {result['received_pressure'].shape}"
          "  (RX elements, time)")
    print("Per scatterer (on-axis, round trip = min t_TX + min t_RX):")
    for e in result["per_scatterer"]:
        s = e["scatterer"]
        rt = (e["t_tx"].min() + e["t_rx"].min()) * 1e6
        a = ap.calculate_attenuation(e["d_tx"].min()) ** 2
        print(f"  z = {s['z'] * 1e3:2.0f} mm: one-way t = {e['t_tx'].min() * 1e6:6.3f}-"
              f"{e['t_tx'].max() * 1e6:6.3f} us | round trip = {rt:6.3f} us "
              f"(2z/c = {2 * s['z'] / config.C * 1e6:6.3f}) | 2-way amp = {a:.3f}")
    return result


# ----------------------------------------------------------------------
# STEP 3: project's final probe (Philips L12-4 model), by arguments only
# ----------------------------------------------------------------------
def step3_final_probe(params):
    p = params["run2_final_probe"]
    banner("STEP 3: Final probe - Philips L12-4 model (128 el, 0.30 mm, 8 MHz)")
    print("Done ONLY by passing arguments to the team's functions (code unchanged).")
    n_el, pitch, f0, fs = p["n_elements"], p["pitch_m"], p["f0_hz"], p["fs_hz"]
    scat = dict(p["scatterer"])

    element_x = ap.create_linear_array(n_el, pitch)
    pulse_t, pulse_p = ap.generate_pulse(fs=fs, f0=f0)
    half_width = pulse_t[-1]

    # Build the shared time axis the same way the team's simulator does:
    # start at minus the pulse half-width, end after the longest round trip.
    t_tx_all = ap.calculate_propagation_delay(
        ap.calculate_tx_distance(element_x, scat["x"], scat["z"]))
    t_rx_all = ap.calculate_propagation_delay(
        ap.calculate_rx_distance(element_x, scat["x"], scat["z"]))
    max_total = float(t_tx_all.max() + t_rx_all.max())
    t_end = max_total + p["time_margin_fraction"] * max_total + half_width
    t_axis = np.arange(-half_width, t_end, 1.0 / fs)

    incident, d_tx, t_tx = ap.forward_propagation(scat, element_x, t_axis, f0=f0)
    reflected = ap.mock_tissue_interaction(incident, scat)
    pressure, d_rx, t_rx = ap.return_propagation(reflected, t_axis, scat, element_x, f0=f0)

    print(f"  Elements        : {n_el}   pitch = {pitch * 1e3:.2f} mm   "
          f"aperture = {(element_x[-1] - element_x[0]) * 1e3:.2f} mm")
    print(f"  Element x range : {element_x[0] * 1e3:.2f} to {element_x[-1] * 1e3:.2f} mm")
    print(f"  Pulse           : {f0 / 1e6:.0f} MHz, {ap.N_CYCLES} cycles, sigma = "
          f"{ap.N_CYCLES / (2 * f0) * 1e6:.4f} us, half-width = {half_width * 1e6:.4f} us, "
          f"{pulse_t.size} samples")
    print(f"  Scatterer       : ({scat['x'] * 1e3:.0f}, {scat['z'] * 1e3:.0f}) mm, amp = {scat['amp']}")
    print(f"  Time axis       : {t_axis.size} samples at {fs / 1e6:.0f} MHz, "
          f"{t_axis[0] * 1e6:.3f} to {t_axis[-1] * 1e6:.3f} us")
    print(f"  One-way distance: {d_tx.min() * 1e3:.3f} to {d_tx.max() * 1e3:.3f} mm")
    print(f"  One-way time    : {t_tx.min() * 1e6:.2f} to {t_tx.max() * 1e6:.2f} us "
          f"(centre to edge elements)")
    print(f"  Round trip time : {(t_tx.min() + t_rx.min()) * 1e6:.2f} us (centre element) to "
          f"{(t_tx.max() + t_rx.max()) * 1e6:.2f} us (edge element)")
    a = ap.calculate_attenuation(d_tx, f0=f0)
    print(f"  One-way amplitude factor at 8 MHz: {a.max():.4f} (centre) to {a.min():.4f} (edge)"
          f"  = {-20 * np.log10(a.max()):.2f} to {-20 * np.log10(a.min()):.2f} dB")
    i_pk = int(np.argmax(np.abs(incident)))
    print(f"  Incident wave peak |p| = {abs(incident[i_pk]):.3f} a.u. at t = {t_axis[i_pk] * 1e6:.2f} us"
          f"  (sum of {n_el} attenuated pulses)")
    print(f"  Received pressure array shape: {pressure.shape}  (RX elements, time)")
    m_c = n_el // 2
    j_pk = int(np.argmax(np.abs(pressure[m_c])))
    print(f"  Element #{m_c}: peak |p| = {abs(pressure[m_c, j_pk]):.3f} a.u. at t = {t_axis[j_pk] * 1e6:.2f} us")
    print(f"  Element #0  : peak |p| = {np.abs(pressure[0]).max():.3f} a.u. at t = "
          f"{t_axis[np.argmax(np.abs(pressure[0]))] * 1e6:.2f} us")

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    ax = axs[0, 0]
    ax.plot(pulse_t * 1e6, pulse_p, color="tab:blue")
    ax.set_title(f"(a) Transmitted pulse: {f0 / 1e6:.0f} MHz, {ap.N_CYCLES}-cycle Gaussian")
    ax.set_xlabel("Time [µs]"); ax.set_ylabel("Pressure (a.u.)"); ax.grid(alpha=0.3)

    ax = axs[0, 1]
    ax.plot(t_axis * 1e6, incident, color="tab:orange", lw=0.9)
    ax.axvspan(t_tx.min() * 1e6, t_tx.max() * 1e6, color="grey", alpha=0.15,
               label="range of one-way travel times")
    ax.set_xlim((t_tx.min() - 1.5e-6) * 1e6, (t_tx.max() + 1.5e-6) * 1e6)
    ax.set_title("(b) Incident wave at the scatterer (0, 20) mm")
    ax.set_xlabel("Time [µs]"); ax.set_ylabel("Pressure (a.u.)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1, 0]
    idx = np.arange(n_el)
    ax.plot(idx, t_tx * 1e6, color="tab:green", label="t_TX (element -> scatterer)")
    ax.plot(idx, t_rx * 1e6, "k:", label="t_RX (scatterer -> element)")
    ax.set_title("(c) One-way travel time vs element")
    ax.set_xlabel("Element index"); ax.set_ylabel("Time [µs]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1, 1]
    vmax = np.max(np.abs(pressure)) or 1.0
    im = ax.imshow(pressure, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax,
                   extent=[t_axis[0] * 1e6, t_axis[-1] * 1e6, n_el - 1, 0])
    ax.set_xlim(24, 38)
    ax.set_title(f"(d) Echo received on all {n_el} elements")
    ax.set_xlabel("Time [µs]"); ax.set_ylabel("RX element index")
    fig.colorbar(im, ax=ax, label="Pressure (a.u.)")
    fig.suptitle("Final probe run: Philips L12-4 model, point scatterer at (0, 20) mm", fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "L12-4_final_probe_propagation.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    npz_path = os.path.join(OUT_DIR, "L12-4_final_probe_outputs.npz")
    np.savez(npz_path, t_axis=t_axis, element_x=element_x, pulse_t=pulse_t, pulse_p=pulse_p,
             incident_wave=incident, reflected_wave=reflected, received_pressure=pressure,
             d_tx=d_tx, t_tx=t_tx, d_rx=d_rx, t_rx=t_rx,
             description=np.array("Final probe run (128 el, 0.30 mm pitch, 8 MHz), scatterer (0,20) mm. "
                                  "received_pressure shape = (128 RX elements, time). Units: m, s, a.u."))
    print()
    print("Saved figure:", rel(fig_path))
    print("Saved data  :", rel(npz_path))


# ----------------------------------------------------------------------
# STEP 4: attenuation at 4 / 8 / 12 MHz
# ----------------------------------------------------------------------
def step4_attenuation(params):
    p = params["attenuation_plot"]
    banner("STEP 4: Attenuation vs depth at 4, 8, 12 MHz (calculate_attenuation)")
    depth = np.linspace(0.0, p["max_depth_m"], 301)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    print(f"alpha0 = {config.ALPHA_0} dB/MHz/cm.  One-way and two-way (round trip) loss:")
    print("  f0       depth 20 mm: one-way dB  amp factor | round-trip dB  amp factor")
    for f in p["frequencies_hz"]:
        a = ap.calculate_attenuation(depth, f0=f)
        axs[0].plot(depth * 1e3, -20 * np.log10(a), label=f"{f / 1e6:.0f} MHz")
        axs[1].plot(depth * 1e3, a ** 2, label=f"{f / 1e6:.0f} MHz")
        a20 = float(ap.calculate_attenuation(0.020, f0=f))
        print(f"  {f / 1e6:4.0f} MHz               {-20 * np.log10(a20):6.2f}      {a20:.4f}   |"
              f"     {-40 * np.log10(a20):6.2f}      {a20 ** 2:.4f}")
    axs[0].set_title("One-way attenuation (dB)")
    axs[0].set_xlabel("Depth [mm]"); axs[0].set_ylabel("Loss [dB]")
    axs[1].set_title("Round-trip amplitude factor  (linear, = A_TX x A_RX)")
    axs[1].set_xlabel("Depth [mm]"); axs[1].set_ylabel("Amplitude factor")
    for ax in axs:
        ax.axvline(20, color="grey", ls="--", alpha=0.6)
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "attenuation_vs_depth_4_8_12MHz.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("Saved figure:", rel(path))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        print("NITK-UsoundSim  -  Team 3: Acoustic Propagation  -  demo run")
        print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, matplotlib {matplotlib.__version__}")
        print(f"Input file : {rel(INPUT_FILE)}")
        print(f"Output dir : {rel(OUT_DIR)}/")
        with open(INPUT_FILE) as fh:
            params = json.load(fh)
        n_ok, n_ran = step1_unit_tests()
        step2_team_demo(params)
        step3_final_probe(params)
        step4_attenuation(params)
        banner("SUMMARY")
        print(f"Unit tests : {n_ok}/{n_ran} passed")
        print("Output files in output/:")
        for name in sorted(os.listdir(OUT_DIR)):
            if name not in ("console_log.txt", "console_screenshot.png", "unit_tests_full_output.txt"):
                print("  -", name)
        print("  - unit_tests_full_output.txt")
        print("  - console_log.txt")
        print("  - console_screenshot.png")
        print(f"Total run time: {time.time() - t0:.1f} s")
    finally:
        sys.stdout = tee.stream
    log = tee.buffer.getvalue()
    with open(os.path.join(OUT_DIR, "console_log.txt"), "w") as fh:
        fh.write(log)
    console_to_png(log, os.path.join(OUT_DIR, "console_screenshot.png"),
                   title="python3 run_team3.py  (Team 3: Acoustic Propagation)", width_chars=106)
    print("Console log and screenshot saved in output/. Done.")


if __name__ == "__main__":
    main()
