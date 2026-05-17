import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


print("THIS IS THE REAL CODE15")


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class WaveformConfig:
    fs_hz: int = 1000
    n_beats: int = 4

    spont_cycle_mean_ms: float = 300.0
    spont_cycle_std_ms: float = 25.0
    paced_cycle_ms: float = 300.0

    tail_ms: float = 220.0
    activation_anchor_ms: float = 55.0

    pre_zoom_ms: float = 55.0
    post_zoom_ms: float = 110.0

    baseline_wander_amp_sp: float = 0.006
    baseline_wander_amp_pc: float = 0.003
    noise_std_sp: float = 0.006
    noise_std_pc: float = 0.004

    local_width_base_ms: float = 4.5
    local_width_fi_gain_ms: float = 6.0
    local_width_block_gain_ms: float = 8.0
    local_width_vuln_gain_ms: float = 5.0

    second_delay_base_ms: float = 18.0
    second_delay_block_gain_ms: float = 26.0
    second_delay_vuln_gain_ms: float = 16.0

    third_delay_base_ms: float = 34.0
    third_delay_vuln_gain_ms: float = 18.0

    amplitude_floor: float = 0.045
    amplitude_gain: float = 0.26

    spont_cleaning_factor: float = 1.00
    paced_cleaning_factor: float = 0.72

    rng_seed: int = 2026


# =========================================================
# 2) File loading
# =========================================================

def resolve_input_file(script_dir: Path, filename: str) -> Path:
    candidates = [
        script_dir / "manual_check_ablation_utility" / filename,
        script_dir / filename,
        Path(filename),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find required input file: {filename}")


def load_code13_exports(script_dir: Path):
    sp_path = resolve_input_file(script_dir, "spontaneous_electrode_features.csv")
    pc_path = resolve_input_file(script_dir, "paced_electrode_features.csv")
    pl_path = resolve_input_file(script_dir, "placement_table.csv")

    sp = pd.read_csv(sp_path)
    pc = pd.read_csv(pc_path)
    pl = pd.read_csv(pl_path)

    return sp, pc, pl, sp_path, pc_path, pl_path


# =========================================================
# 3) Selection of representative placements/electrodes
# =========================================================

def minmax_series(s: pd.Series, eps: float = 1e-8) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + eps)


def build_placement_summary(sp: pd.DataFrame) -> pd.DataFrame:
    summary = (
        sp.groupby("placement_id")
        .agg(
            time_index=("time_index", "first"),
            time_min=("time_min", "first"),
            center_x=("center_x", "first"),
            center_y=("center_y", "first"),
            angle_deg=("angle_deg", "first"),
            LAT_mean=("LAT", "mean"),
            A_mean=("A", "mean"),
            FI_mean=("FI", "mean"),
            block_mean=("block_field", "mean"),
            vuln_mean=("vuln_field", "mean"),
            interaction_mean=("interaction", "mean"),
            Disp_mean=("Disp", "mean"),
        )
        .reset_index()
    )

    summary["FI_norm"] = minmax_series(summary["FI_mean"])
    summary["block_norm"] = minmax_series(summary["block_mean"])
    summary["vuln_norm"] = minmax_series(summary["vuln_mean"])

    summary["normal_burden"] = (
        0.55 * summary["block_norm"]
        + 0.30 * summary["vuln_norm"]
        + 0.15 * summary["FI_norm"]
    )

    summary["vuln_frac_score"] = (
        0.65 * summary["FI_norm"] + 0.35 * summary["vuln_norm"]
    )

    return summary


def choose_representative_placements(sp: pd.DataFrame):
    summary = build_placement_summary(sp)

    normal_id = int(summary.sort_values(["normal_burden", "FI_mean", "block_mean"]).iloc[0]["placement_id"])

    remaining = summary[summary["placement_id"] != normal_id].copy()
    block_id = int(remaining.sort_values(["block_mean", "FI_mean"], ascending=False).iloc[0]["placement_id"])

    remaining = remaining[remaining["placement_id"] != block_id].copy()
    vuln_id = int(remaining.sort_values(["vuln_frac_score", "FI_mean"], ascending=False).iloc[0]["placement_id"])

    selected = {
        "normal": normal_id,
        "block": block_id,
        "vuln_fractionated": vuln_id,
    }
    return selected, summary


def choose_representative_electrodes(sp: pd.DataFrame, placement_id: int):
    patch = sp[sp["placement_id"] == placement_id].copy()

    patch["fi_plus_vuln"] = patch["FI"] + 2.0 * patch["vuln_field"]

    elec_low_fi = int(patch.sort_values(["FI", "block_field", "vuln_field"]).iloc[0]["electrode_id"])
    elec_high_fiv = int(patch.sort_values(["fi_plus_vuln", "FI"], ascending=False).iloc[0]["electrode_id"])
    elec_high_block = int(patch.sort_values(["block_field", "FI"], ascending=False).iloc[0]["electrode_id"])

    return {
        "lowest_fi": elec_low_fi,
        "highest_fi_vuln": elec_high_fiv,
        "highest_block": elec_high_block,
    }


# =========================================================
# 4) Waveform generation helpers
# =========================================================

def make_branch_time_axis(cfg: WaveformConfig, branch: str, rng: np.random.Generator):
    if branch == "spontaneous":
        cls = rng.normal(cfg.spont_cycle_mean_ms, cfg.spont_cycle_std_ms, cfg.n_beats)
        cls = np.clip(cls, 230.0, 380.0)
    else:
        cls = np.full(cfg.n_beats, cfg.paced_cycle_ms)

    beat_starts = np.zeros(cfg.n_beats)
    for k in range(1, cfg.n_beats):
        beat_starts[k] = beat_starts[k - 1] + cls[k - 1]

    total_ms = int(np.ceil(beat_starts[-1] + cls[-1] + cfg.tail_ms))
    t_ms = np.arange(total_ms, dtype=float)
    return t_ms, beat_starts, cls


def soft_amplitude_map(a_value: float, a_ref: float, cfg: WaveformConfig) -> float:
    return cfg.amplitude_floor + cfg.amplitude_gain * np.tanh(max(a_value, 0.0) / max(a_ref, 1e-6))


def biphasic_kernel(t_rel: np.ndarray, amp: float, width_ms: float) -> np.ndarray:
    pos = 0.85 * amp * np.exp(-((t_rel + 0.34 * width_ms) ** 2) / (2.0 * (0.20 * width_ms) ** 2 + 1e-8))
    neg = 1.20 * amp * np.exp(-(t_rel ** 2) / (2.0 * (0.24 * width_ms) ** 2 + 1e-8))
    tail = 0.22 * amp * np.exp(-((t_rel - 0.55 * width_ms) ** 2) / (2.0 * (0.30 * width_ms) ** 2 + 1e-8))
    return pos - neg + 0.35 * tail


def infer_n_deflections(fi_norm: float, block_norm: float, vuln_norm: float, branch: str) -> int:
    score = 0.95 * fi_norm + 1.10 * block_norm + 0.90 * vuln_norm
    if branch == "paced":
        if score > 0.95:
            return 2
        return 1

    if score > 1.35:
        return 3
    if score > 0.60:
        return 2
    return 1


def generate_electrode_waveform(
    row: pd.Series,
    branch: str,
    t_ms: np.ndarray,
    beat_starts: np.ndarray,
    a_ref: float,
    fi_ref: float,
    block_ref: float,
    vuln_ref: float,
    cfg: WaveformConfig,
    rng: np.random.Generator,
):
    fi_norm = float(row["FI"] / max(fi_ref, 1e-8))
    block_norm = float(row["block_field"] / max(block_ref, 1e-8))
    vuln_norm = float(row["vuln_field"] / max(vuln_ref, 1e-8))

    fi_norm = min(max(fi_norm, 0.0), 1.25)
    block_norm = min(max(block_norm, 0.0), 1.25)
    vuln_norm = min(max(vuln_norm, 0.0), 1.25)

    amp = soft_amplitude_map(float(row["A"]), a_ref, cfg)
    if branch == "paced":
        amp *= cfg.paced_cleaning_factor
    else:
        amp *= cfg.spont_cleaning_factor

    width_ms = (
        cfg.local_width_base_ms
        + cfg.local_width_fi_gain_ms * fi_norm
        + cfg.local_width_block_gain_ms * block_norm
        + cfg.local_width_vuln_gain_ms * vuln_norm
    )

    n_defl = infer_n_deflections(fi_norm, block_norm, vuln_norm, branch)

    latency_rel = float(row["LAT"])
    signal = np.zeros_like(t_ms, dtype=float)

    for beat_start in beat_starts:
        t0 = beat_start + cfg.activation_anchor_ms + latency_rel

        signal += biphasic_kernel(t_ms - t0, amp=amp, width_ms=width_ms)

        if n_defl >= 2:
            d2 = cfg.second_delay_base_ms + cfg.second_delay_block_gain_ms * block_norm + cfg.second_delay_vuln_gain_ms * vuln_norm
            signal += 0.72 * biphasic_kernel(t_ms - (t0 + d2), amp=amp, width_ms=0.92 * width_ms)

        if n_defl >= 3:
            d3 = cfg.third_delay_base_ms + cfg.third_delay_vuln_gain_ms * vuln_norm
            signal += 0.46 * biphasic_kernel(t_ms - (t0 + d3), amp=amp, width_ms=0.88 * width_ms)

    phase = rng.uniform(0, 2 * np.pi)
    if branch == "spontaneous":
        wander = cfg.baseline_wander_amp_sp * np.sin(2 * np.pi * t_ms / 900.0 + phase)
        noise = rng.normal(0.0, cfg.noise_std_sp, size=t_ms.shape[0])
    else:
        wander = cfg.baseline_wander_amp_pc * np.sin(2 * np.pi * t_ms / 1200.0 + phase)
        noise = rng.normal(0.0, cfg.noise_std_pc, size=t_ms.shape[0])

    signal = signal + wander + noise

    first_event_time = beat_starts[0] + cfg.activation_anchor_ms + latency_rel

    return signal, first_event_time, n_defl


def generate_branch_waveforms(branch_df: pd.DataFrame, branch: str, cfg: WaveformConfig, base_seed: int):
    rng_master = np.random.default_rng(base_seed)

    a_ref = float(branch_df["A"].quantile(0.80))
    fi_ref = float(max(branch_df["FI"].quantile(0.90), 1e-6))
    block_ref = float(max(branch_df["block_field"].quantile(0.95), 1e-6))
    vuln_ref = float(max(branch_df["vuln_field"].quantile(0.95), 1e-6))

    placements = {}
    meta_rows = []

    for placement_id, patch in branch_df.groupby("placement_id"):
        patch = patch.sort_values(["row", "col", "electrode_id"]).reset_index(drop=True)

        rng_axis = np.random.default_rng(rng_master.integers(0, 10_000_000))
        t_ms, beat_starts, cls = make_branch_time_axis(cfg, branch, rng_axis)

        patch_signals = {}
        patch_events = {}
        patch_ndefl = {}

        for _, row in patch.iterrows():
            rng_e = np.random.default_rng(rng_master.integers(0, 10_000_000))
            sig, first_event_time, n_defl = generate_electrode_waveform(
                row=row,
                branch=branch,
                t_ms=t_ms,
                beat_starts=beat_starts,
                a_ref=a_ref,
                fi_ref=fi_ref,
                block_ref=block_ref,
                vuln_ref=vuln_ref,
                cfg=cfg,
                rng=rng_e,
            )
            electrode_id = int(row["electrode_id"])
            patch_signals[electrode_id] = sig
            patch_events[electrode_id] = first_event_time
            patch_ndefl[electrode_id] = n_defl

            meta_rows.append({
                "branch": branch,
                "placement_id": int(placement_id),
                "electrode_id": electrode_id,
                "row": int(row["row"]),
                "col": int(row["col"]),
                "LAT": float(row["LAT"]),
                "A": float(row["A"]),
                "FI": float(row["FI"]),
                "block_field": float(row["block_field"]),
                "vuln_field": float(row["vuln_field"]),
                "first_event_time_ms": float(first_event_time),
                "n_deflections": int(n_defl),
            })

        placements[int(placement_id)] = {
            "t_ms": t_ms,
            "beat_starts": beat_starts,
            "cycle_lengths_ms": cls,
            "signals": patch_signals,
            "first_events": patch_events,
            "n_deflections": patch_ndefl,
        }

    meta_df = pd.DataFrame(meta_rows)
    return placements, meta_df


# =========================================================
# 5) Bipolar construction
# =========================================================

def build_adjacent_pairs():
    pairs = []
    for r in range(4):
        for c in range(3):
            e1 = r * 4 + c
            e2 = r * 4 + (c + 1)
            pairs.append((f"h_{e1}_{e2}", e1, e2))
    for r in range(3):
        for c in range(4):
            e1 = r * 4 + c
            e2 = (r + 1) * 4 + c
            pairs.append((f"v_{e1}_{e2}", e1, e2))
    return pairs


def build_bipolar_waveforms(unipolar_store: dict):
    pair_defs = build_adjacent_pairs()
    bipolar_store = {}

    for placement_id, payload in unipolar_store.items():
        t_ms = payload["t_ms"]
        signals = payload["signals"]

        pair_signals = {}
        pair_peaks = {}
        for pair_name, e1, e2 in pair_defs:
            b = signals[e1] - signals[e2]
            pair_signals[pair_name] = b
            pair_peaks[pair_name] = float(t_ms[np.argmax(np.abs(b))])

        bipolar_store[placement_id] = {
            "t_ms": t_ms,
            "signals": pair_signals,
            "first_peak_ms": pair_peaks,
        }

    return bipolar_store


# =========================================================
# 6) Plotting helpers
# =========================================================

def crop_window(t_ms, y, center_ms, pre_ms, post_ms):
    mask = (t_ms >= center_ms - pre_ms) & (t_ms <= center_ms + post_ms)
    return t_ms[mask] - center_ms, y[mask]


def plot_feature_maps(sp: pd.DataFrame, selected: dict, out_path: Path):
    fields = ["LAT", "A", "FI", "block_field", "vuln_field"]
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 10))

    row_labels = ["normal", "block", "vuln_fractionated"]

    for r, label in enumerate(row_labels):
        placement_id = selected[label]
        patch = sp[sp["placement_id"] == placement_id].sort_values(["row", "col"]).reset_index(drop=True)

        for c, field in enumerate(fields):
            ax = axes[r, c]
            grid = patch[field].to_numpy().reshape(4, 4)
            im = ax.imshow(grid, origin="lower", cmap="turbo")
            ax.set_title(f"{label}\n{field}")
            for rr in range(4):
                for cc in range(4):
                    ax.text(cc, rr, f"{grid[rr, cc]:.2f}", ha="center", va="center", fontsize=7, color="black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Code15: Selected spontaneous feature maps", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_full_trace_examples(sp_store, pc_store, selected_electrodes, out_path: Path):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
    row_order = ["normal", "block", "vuln_fractionated"]

    for r, label in enumerate(row_order):
        placement_id = selected_electrodes[label]["placement_id"]
        electrode_id = selected_electrodes[label]["highest_fi_vuln"]

        t_sp = sp_store[placement_id]["t_ms"]
        y_sp = sp_store[placement_id]["signals"][electrode_id]
        t_pc = pc_store[placement_id]["t_ms"]
        y_pc = pc_store[placement_id]["signals"][electrode_id]

        ax = axes[r, 0]
        ax.plot(t_sp, y_sp, lw=1.0)
        ax.set_title(f"{label} — spontaneous unipolar (elec {electrode_id})")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude (a.u.)")
        ax.grid(alpha=0.25)

        ax = axes[r, 1]
        ax.plot(t_pc, y_pc, lw=1.0, color="tab:orange")
        ax.set_title(f"{label} — paced unipolar (elec {electrode_id})")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude (a.u.)")
        ax.grid(alpha=0.25)

    fig.suptitle("Code15: Full-trace waveform examples", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_event_centered_unipolar_zoom(sp_store, pc_store, sp_df, selected_electrodes, meta_df, cfg: WaveformConfig, out_path: Path):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 14))
    row_order = ["normal", "block", "vuln_fractionated"]
    col_order = ["lowest_fi", "highest_fi_vuln", "highest_block"]

    pretty = {
        "lowest_fi": "lowest FI",
        "highest_fi_vuln": "highest FI+vuln",
        "highest_block": "highest block",
    }

    for r, label in enumerate(row_order):
        placement_id = selected_electrodes[label]["placement_id"]
        patch = sp_df[sp_df["placement_id"] == placement_id]

        for c, key in enumerate(col_order):
            electrode_id = selected_electrodes[label][key]

            row_meta = patch[patch["electrode_id"] == electrode_id].iloc[0]

            t_sp = sp_store[placement_id]["t_ms"]
            y_sp = sp_store[placement_id]["signals"][electrode_id]
            center_sp = sp_store[placement_id]["first_events"][electrode_id]
            x_sp, z_sp = crop_window(t_sp, y_sp, center_sp, cfg.pre_zoom_ms, cfg.post_zoom_ms)

            t_pc = pc_store[placement_id]["t_ms"]
            y_pc = pc_store[placement_id]["signals"][electrode_id]
            center_pc = pc_store[placement_id]["first_events"][electrode_id]
            x_pc, z_pc = crop_window(t_pc, y_pc, center_pc, cfg.pre_zoom_ms, cfg.post_zoom_ms)

            n_defl = int(meta_df[
                (meta_df["branch"] == "spontaneous") &
                (meta_df["placement_id"] == placement_id) &
                (meta_df["electrode_id"] == electrode_id)
            ]["n_deflections"].iloc[0])

            ax = axes[r, c]
            ax.plot(x_sp, z_sp, label="Spont", lw=1.2)
            ax.plot(x_pc, z_pc, "--", label="Paced", lw=1.2, color="tab:orange")
            ax.axvline(0.0, color="gray", lw=0.9)
            ax.set_title(
                f"{label} ({pretty[key]})\n"
                f"elec {electrode_id}  FI={row_meta['FI']:.3f}  "
                f"blk={row_meta['block_field']:.3f}  vuln={row_meta['vuln_field']:.3f}  "
                f"n_defl={n_defl}"
            )
            ax.set_xlabel("time rel. activation (ms)")
            ax.set_ylabel("amplitude (a.u.)")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle("Code15: Event-centered unipolar zoom", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_event_centered_bipolar_zoom(sp_bi, pc_bi, selected, cfg: WaveformConfig, out_path: Path):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 13))
    row_order = ["normal", "block", "vuln_fractionated"]
    pair_order = ["h_0_1", "v_0_4"]

    for r, label in enumerate(row_order):
        placement_id = selected[label]

        for c, pair_name in enumerate(pair_order):
            t_sp = sp_bi[placement_id]["t_ms"]
            y_sp = sp_bi[placement_id]["signals"][pair_name]
            center_sp = sp_bi[placement_id]["first_peak_ms"][pair_name]
            x_sp, z_sp = crop_window(t_sp, y_sp, center_sp, cfg.pre_zoom_ms, cfg.post_zoom_ms)

            t_pc = pc_bi[placement_id]["t_ms"]
            y_pc = pc_bi[placement_id]["signals"][pair_name]
            center_pc = pc_bi[placement_id]["first_peak_ms"][pair_name]
            x_pc, z_pc = crop_window(t_pc, y_pc, center_pc, cfg.pre_zoom_ms, cfg.post_zoom_ms)

            ax = axes[r, c]
            ax.plot(x_sp, z_sp, label="Spont", lw=1.2)
            ax.plot(x_pc, z_pc, "--", label="Paced", lw=1.2, color="tab:orange")
            ax.axvline(0.0, color="gray", lw=0.9)
            ax.set_title(f"{label} — pair {pair_name}")
            ax.set_xlabel("time rel. bipolar peak (ms)")
            ax.set_ylabel("amplitude (a.u.)")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle("Code15: Event-centered bipolar zoom", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 7) Save waveform tables
# =========================================================

def save_unipolar_long_csv(store: dict, branch: str, out_path: Path):
    rows = []
    for placement_id, payload in store.items():
        t_ms = payload["t_ms"]
        for electrode_id, sig in payload["signals"].items():
            rows.append(pd.DataFrame({
                "branch": branch,
                "placement_id": placement_id,
                "electrode_id": electrode_id,
                "time_ms": t_ms,
                "signal": sig,
            }))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_path, index=False)


def save_bipolar_long_csv(store: dict, branch: str, out_path: Path):
    rows = []
    for placement_id, payload in store.items():
        t_ms = payload["t_ms"]
        for pair_name, sig in payload["signals"].items():
            rows.append(pd.DataFrame({
                "branch": branch,
                "placement_id": placement_id,
                "pair_name": pair_name,
                "time_ms": t_ms,
                "signal": sig,
            }))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_path, index=False)


# =========================================================
# 8) Main
# =========================================================

if __name__ == "__main__":
    print("__file__ =", __file__)

    script_dir = Path(__file__).resolve().parent
    print("script_dir =", script_dir)

    output_dir = script_dir / "code15_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("output_dir =", output_dir)
    print("output_dir exists before run =", output_dir.exists())

    cfg = WaveformConfig()

    sp, pc, placement_df, sp_path, pc_path, pl_path = load_code13_exports(script_dir)

    print("Loaded:")
    print("sp_path =", sp_path)
    print("pc_path =", pc_path)
    print("pl_path =", pl_path)

    selected, placement_summary = choose_representative_placements(sp)

    selected_electrodes = {}
    for label, placement_id in selected.items():
        sel = choose_representative_electrodes(sp, placement_id)
        sel["placement_id"] = placement_id
        selected_electrodes[label] = sel

    sp_store, sp_meta = generate_branch_waveforms(sp, "spontaneous", cfg, base_seed=cfg.rng_seed + 1)
    pc_store, pc_meta = generate_branch_waveforms(pc, "paced", cfg, base_seed=cfg.rng_seed + 2)

    sp_bi = build_bipolar_waveforms(sp_store)
    pc_bi = build_bipolar_waveforms(pc_store)

    meta_df = pd.concat([sp_meta, pc_meta], ignore_index=True)

    placement_summary.to_csv(output_dir / "placement_summary.csv", index=False)

    selected_rows = []
    for label, info in selected_electrodes.items():
        selected_rows.append({
            "label": label,
            "placement_id": info["placement_id"],
            "lowest_fi_electrode": info["lowest_fi"],
            "highest_fi_vuln_electrode": info["highest_fi_vuln"],
            "highest_block_electrode": info["highest_block"],
        })
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_examples.csv", index=False)
    meta_df.to_csv(output_dir / "waveform_metadata.csv", index=False)

    save_unipolar_long_csv(sp_store, "spontaneous", output_dir / "spontaneous_unipolar_waveforms.csv")
    save_unipolar_long_csv(pc_store, "paced", output_dir / "paced_unipolar_waveforms.csv")
    save_bipolar_long_csv(sp_bi, "spontaneous", output_dir / "spontaneous_bipolar_waveforms.csv")
    save_bipolar_long_csv(pc_bi, "paced", output_dir / "paced_bipolar_waveforms.csv")

    plot_feature_maps(sp, selected, output_dir / "selected_feature_maps.png")
    plot_full_trace_examples(sp_store, pc_store, selected_electrodes, output_dir / "full_trace_examples.png")
    plot_event_centered_unipolar_zoom(
        sp_store, pc_store, sp, selected_electrodes, meta_df, cfg,
        output_dir / "event_centered_unipolar_zoom.png"
    )
    plot_event_centered_bipolar_zoom(
        sp_bi, pc_bi, selected, cfg,
        output_dir / "event_centered_bipolar_zoom.png"
    )

    print("\nDONE. Files should now exist here:")
    print(output_dir)

    for fname in [
        "placement_summary.csv",
        "selected_examples.csv",
        "waveform_metadata.csv",
        "spontaneous_unipolar_waveforms.csv",
        "paced_unipolar_waveforms.csv",
        "spontaneous_bipolar_waveforms.csv",
        "paced_bipolar_waveforms.csv",
        "selected_feature_maps.png",
        "full_trace_examples.png",
        "event_centered_unipolar_zoom.png",
        "event_centered_bipolar_zoom.png",
    ]:
        fpath = output_dir / fname
        print(fname, "exists =", fpath.exists())