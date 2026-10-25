"""
demo_acoustic_propagation.py
==============================

Small end-to-end demo of the Phase-1 acoustic propagation pipeline:

    N_ELEMENTS (128) TX elements -> forward_propagation() -> mock_tissue_interaction()
                    -> return_propagation()  -> pressure[N_ELEMENTS, time]

run on the three synthetic PHANTOM_POINTS scatterers from config.py.
Saves two PNG figures illustrating every stage of the pipeline.

Run with (from the usound_sim folder, so the package is importable):
    python -m nitk_usoundsim.demo_acoustic_propagation
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from nitk_usoundsim import config
    from nitk_usoundsim import acoustic_propagation as ap
except ImportError:  # allow running directly from inside the package dir
    import config
    import acoustic_propagation as ap

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def make_figures(out_dir: str = OUT_DIR):
    result = ap.simulate_acoustic_propagation()
    x_el = result["element_x"]
    t_axis = result["t_axis"]
    pressure = result["pressure"]
    pulse_t, pulse_p = result["pulse_t"], result["pulse_p"]

    # ======================================================================
    # Figure 1: array geometry, TX pulse, delay-vs-depth, attenuation-vs-depth
    # ======================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) array geometry + phantom scatterers
    ax = axes[0, 0]
    ax.scatter(x_el * 1e3, np.zeros_like(x_el), marker="s", s=18,
               color="tab:blue", label="TX/RX elements")
    for pt, c in zip(config.PHANTOM_POINTS, COLORS):
        ax.scatter(pt["x"] * 1e3, pt["z"] * 1e3, marker="*", s=180, color=c, zorder=3)
        ax.annotate(f"z={pt['z']*1e3:.0f} mm", (pt["x"] * 1e3, pt["z"] * 1e3),
                    textcoords="offset points", xytext=(8, 0), fontsize=9)
    ax.set_xlabel("Lateral x [mm]")
    ax.set_ylabel("Depth z [mm]")
    ax.set_title(f"{config.N_ELEMENTS}-element linear array + phantom scatterers")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # (b) transmitted pulse
    ax = axes[0, 1]
    ax.plot(pulse_t * 1e6, pulse_p, color="tab:blue")
    ax.set_xlabel("Time [\u00b5s]")
    ax.set_ylabel("Pressure (a.u.)")
    ax.set_title(f"Transmitted pulse ({config.F0/1e6:.0f} MHz, "
                 f"{ap.N_CYCLES:.0f}-cycle Gaussian envelope)")
    ax.grid(alpha=0.3)

    # (c) propagation delay vs depth (on-axis, near-central element)
    x_center = float(x_el[config.N_ELEMENTS // 2])
    z_sweep = np.linspace(config.Z_MIN, config.Z_MAX, 200)
    d_sweep = ap.calculate_tx_distance(x_center, 0.0, z_sweep)
    t_sweep = ap.calculate_propagation_delay(d_sweep, c=config.C)

    ax = axes[1, 0]
    ax.plot(z_sweep * 1e3, t_sweep * 1e6, color="tab:green")
    for pt, c in zip(config.PHANTOM_POINTS, COLORS):
        ax.axvline(pt["z"] * 1e3, color=c, ls="--", alpha=0.6)
    ax.set_xlabel("Depth z [mm]")
    ax.set_ylabel("One-way delay [\u00b5s]")
    ax.set_title("Propagation delay vs depth (on-axis)")
    ax.grid(alpha=0.3)

    # (d) attenuation vs depth
    a_sweep = ap.calculate_attenuation(d_sweep, f0=config.F0, alpha0=config.ALPHA_0)
    a_sweep_db = -20.0 * np.log10(a_sweep)

    ax = axes[1, 1]
    ax.plot(z_sweep * 1e3, a_sweep_db, color="tab:purple")
    for pt, c in zip(config.PHANTOM_POINTS, COLORS):
        ax.axvline(pt["z"] * 1e3, color=c, ls="--", alpha=0.6)
    ax.set_xlabel("Depth z [mm]")
    ax.set_ylabel("One-way attenuation [dB]")
    ax.set_title(f"Attenuation vs depth (at F0 = {config.F0/1e6:.0f} MHz)")
    ax.grid(alpha=0.3)

    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "demo_geometry_pulse_delay_attenuation.png")
    fig1.savefig(fig1_path, dpi=140)
    plt.close(fig1)

    # ======================================================================
    # Figure 2: pipeline signals (incident / reflected / one RX / RX image)
    # ======================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8))

    # (a) incident wave AT each scatterer (forward_propagation output)
    ax = axes2[0, 0]
    incidents = {}
    for entry, c in zip(result["per_scatterer"], COLORS):
        scatterer = entry["scatterer"]
        incident, _, _ = ap.forward_propagation(scatterer, x_el, t_axis)
        incidents[id(scatterer)] = incident
        ax.plot(t_axis * 1e6, incident, color=c, label=f"z={scatterer['z']*1e3:.0f} mm")
    ax.set_xlabel("Time [\u00b5s]")
    ax.set_ylabel("Incident pressure (a.u.)")
    ax.set_title("Forward propagation: incident wave at each scatterer")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) synthetic reflected/scattered waves (mock_tissue_interaction output)
    ax = axes2[0, 1]
    for entry, c in zip(result["per_scatterer"], COLORS):
        scatterer = entry["scatterer"]
        reflected = ap.mock_tissue_interaction(incidents[id(scatterer)], scatterer)
        ax.plot(t_axis * 1e6, reflected, color=c,
                label=f"z={scatterer['z']*1e3:.0f} mm, amp={scatterer['amp']:.1f}")
    ax.set_xlabel("Time [\u00b5s]")
    ax.set_ylabel("Reflected pressure (a.u.)")
    ax.set_title("Synthetic (mock) reflected/scattered waves")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) one receive-element pressure waveform (central element)
    ax = axes2[1, 0]
    m_center = config.N_ELEMENTS // 2
    ax.plot(t_axis * 1e6, pressure[m_center, :], color="black")
    for pt, c in zip(config.PHANTOM_POINTS, COLORS):
        expected_rt = 2 * pt["z"] / config.C
        ax.axvline(expected_rt * 1e6, color=c, ls=":", alpha=0.6)
    ax.set_xlabel("Time [\u00b5s]")
    ax.set_ylabel("Pressure (a.u.)")
    ax.set_title(f"Return propagation: RX element #{m_center} waveform")
    ax.text(0.02, 0.02, "dotted = expected same-path\nround-trip time per target",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left", alpha=0.8)
    ax.grid(alpha=0.3)

    # (d) receive-element x time image of the full returning pressure field
    ax = axes2[1, 1]
    extent = [t_axis[0] * 1e6, t_axis[-1] * 1e6, config.N_ELEMENTS - 1, 0]
    vmax = np.max(np.abs(pressure)) or 1.0
    im = ax.imshow(pressure, aspect="auto", extent=extent, cmap="seismic",
                    vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Time [\u00b5s]")
    ax.set_ylabel("RX element index")
    ax.set_title(f"Returning mechanical pressure field (all {config.N_ELEMENTS} RX elements)")
    fig2.colorbar(im, ax=ax, label="Pressure (a.u.)")

    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "demo_pipeline_signals.png")
    fig2.savefig(fig2_path, dpi=140)
    plt.close(fig2)

    return fig1_path, fig2_path, result


def save_part_outputs(result, out_dir: str = OUT_DIR):
    """
    Save this module's two deliverables into SEPARATE, self-describing
    .npz files -- one per project part -- so each can be shown/inspected
    independently:

      PART 1  forward_propagation()   TX array -> scatterer
              -> part1_forward_propagation_outputs.npz

      PART 2  return_propagation()    scatterer -> RX array (FINAL output)
              -> part2_return_propagation_outputs.npz

    Every numeric array is paired with a "*_units" and/or "*_row_labels"
    entry, and each file has a top-level "description" string, so the
    contents are understandable on their own when printed -- no need to
    have this source file open alongside them.
    """
    scatterer_labels = np.array([
        f"scatterer_{i}: x={e['scatterer']['x']*1e3:.1f}mm, "
        f"z={e['scatterer']['z']*1e3:.1f}mm, amp={e['scatterer']['amp']:.2f}"
        for i, e in enumerate(result["per_scatterer"])
    ])
    scatterer_xyz = np.array(
        [[e["scatterer"]["x"], e["scatterer"]["z"], e["scatterer"]["amp"]]
         for e in result["per_scatterer"]]
    )
    rx_labels = np.array(
        [f"rx_element_{m:02d}" for m in range(result["received_pressure"].shape[0])]
    )

    # ---------------- PART 1: forward propagation (TX array -> scatterer) ----
    part1_path = os.path.join(out_dir, "part1_forward_propagation_outputs.npz")
    np.savez(
        part1_path,
        description=np.array(
            "PART 1 - forward_propagation(): incident mechanical pressure wave "
            f"arriving AT EACH SCATTERER from the {config.N_ELEMENTS}-element TX array, vs time. "
            "Row i of incident_pressure corresponds to scatterer_xyz[i] / "
            "incident_pressure_row_labels[i]. Also included: the TX pulse "
            "(pulse_t, pulse_p) and TX element geometry (element_x) used to "
            "compute it."
        ),
        t_axis=result["t_axis"],
        t_axis_units=np.array("seconds"),
        incident_pressure=result["incident_pressure"],
        incident_pressure_shape_meaning=np.array("(n_scatterers, n_time_samples)"),
        incident_pressure_units=np.array("arbitrary pressure units (a.u.)"),
        incident_pressure_row_labels=scatterer_labels,
        scatterer_xyz=scatterer_xyz,
        scatterer_xyz_columns=np.array(["x [m]", "z [m]", "amp"]),
        element_x=result["element_x"],
        element_x_units=np.array("meters (TX element lateral position)"),
        pulse_t=result["pulse_t"],
        pulse_p=result["pulse_p"],
    )

    # ---------------- PART 2: return propagation (scatterer -> RX array) -----
    part2_path = os.path.join(out_dir, "part2_return_propagation_outputs.npz")
    np.savez(
        part2_path,
        description=np.array(
            "PART 2 - return_propagation(): FINAL module output. Mechanical "
            f"pressure received at each of the {config.N_ELEMENTS} RX elements vs time "
            "(received_pressure), all scatterers combined by linear "
            "superposition. reflected_pressure_input is the reflected/"
            "scattered wave this stage propagates to the RX array -- it is "
            "INCLUDED HERE ONLY FOR CONTEXT (currently produced by the "
            "temporary mock tissue-interaction stand-in; return_propagation() "
            "itself does not care how it was produced, so the real tissue-"
            "interaction module's output can replace it later unchanged)."
        ),
        t_axis=result["t_axis"],
        t_axis_units=np.array("seconds"),
        received_pressure=result["received_pressure"],
        received_pressure_shape_meaning=np.array(f"({config.N_ELEMENTS} rx_elements, n_time_samples)"),
        received_pressure_units=np.array("arbitrary pressure units (a.u.)"),
        received_pressure_row_labels=rx_labels,
        reflected_pressure_input=result["reflected_pressure"],
        reflected_pressure_input_row_labels=scatterer_labels,
        scatterer_xyz=scatterer_xyz,
        scatterer_xyz_columns=np.array(["x [m]", "z [m]", "amp"]),
        element_x=result["element_x"],
        element_x_units=np.array("meters (RX element lateral position)"),
    )

    return part1_path, part2_path


if __name__ == "__main__":
    f1, f2, result = make_figures()
    part1_path, part2_path = save_part_outputs(result)

    print("Saved figure:", f1)
    print("Saved figure:", f2)
    print("Saved PART 1 (forward propagation) data:", part1_path)
    print("Saved PART 2 (return propagation) data:  ", part2_path)
    print()
    print("PART 1 - incident_pressure  :", result["incident_pressure"].shape,
          " <- pressure AT each scatterer, per phantom point")
    print("PART 2 - received_pressure  :", result["received_pressure"].shape,
          " <- FINAL output, pressure[rx_element, time]")
    print()
    print("To load and inspect later, e.g.:")
    print("  import numpy as np")
    print("  d1 = np.load('part1_forward_propagation_outputs.npz')")
    print("  print(d1['description'])")
    print("  print(d1.files)                       # lists every labelled array")
    print("  d1['incident_pressure']                # (n_scatterers, n_time)")
    print("  d1['incident_pressure_row_labels']      # which row is which scatterer")
    print()
    print("  d2 = np.load('part2_return_propagation_outputs.npz')")
    print("  print(d2['description'])")
    print("  d2['received_pressure']                # (N_ELEMENTS, n_time)")
