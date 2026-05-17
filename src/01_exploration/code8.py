import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, replace
from pathlib import Path


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class DatasetConfig:
    # Global domain
    domain_width: float = 100.0
    domain_height: float = 80.0

    # Patch geometry
    patch_rows: int = 4
    patch_cols: int = 4
    electrode_spacing: float = 4.0

    # Dataset size
    n_samples: int = 1
    n_placements: int = 10

    # Time step between placements
    minutes_per_step: float = 25.0

    # Trajectory settings
    trajectory_mode: str = "zigzag"
    placement_jitter: float = 2.0
    angle_jitter_deg: float = 20.0

    # Hidden substrate
    n_sources: int = 2
    base_cv: float = 0.7
    min_cv: float = 0.20

    # Raw feature model parameters
    sigma_A: float = 18.0
    sigma_I: float = 15.0

    lat_noise_std: float = 2.0
    amp_noise_std: float = 0.05
    fi_noise_std: float = 0.03
    df_noise_std: float = 0.15

    amp_scale_1: float = 1.0
    amp_scale_2: float = 0.9

    fi_base: float = 0.20
    fi_alpha1: float = 0.18
    fi_alpha2: float = 0.18
    fi_beta: float = 0.30
    fi_gamma: float = 0.25

    df_base: float = 5.5
    df_lambda1: float = 1.0
    df_lambda2: float = 1.0
    df_mu: float = 1.2
    df_nu: float = 0.8

    # Dispersion weights
    disp_c1: float = 0.5
    disp_c2: float = 0.5

    # Smoothing / aggregation params
    sigma_smooth: float = 1.0
    sigma_kernel: float = 6.0
    low_voltage_threshold: float = 0.5
    grad_eps: float = 1e-6

    # Global map grid
    global_grid_step: float = 2.0

    # Confidence masking
    confidence_threshold_ratio: float = 0.15

    # PatchScore weights
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    # DIC coefficients
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2

    random_seed: int = 42


# =========================================================
# 2) Utility functions
# =========================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def radial_bump(x, y, cx, cy, radius, strength):
    sigma = max(radius / 2.0, 1e-6)
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return strength * np.exp(-d2 / (2.0 * sigma ** 2))


def minmax_normalize(series: pd.Series, eps: float = 1e-8) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + eps)


def gaussian_kernel_3x3():
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float)
    return k / k.sum()


def smooth_grid_3x3(grid: np.ndarray) -> np.ndarray:
    kernel = gaussian_kernel_3x3()
    padded = np.pad(grid, pad_width=1, mode="edge")
    out = np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r+3, c:c+3]
            out[r, c] = np.sum(window * kernel)
    return out


# =========================================================
# 3) Hidden substrate generation
# =========================================================

def generate_hidden_substrate(config: DatasetConfig, rng: np.random.Generator) -> dict:
    W = config.domain_width
    H = config.domain_height
    margin = 12.0

    substrate = {}

    for k in range(1, config.n_sources + 1):
        substrate[f"source{k}_x"] = rng.uniform(margin, W - margin)
        substrate[f"source{k}_y"] = rng.uniform(margin, H - margin)

    substrate["block_x"] = rng.uniform(margin, W - margin)
    substrate["block_y"] = rng.uniform(margin, H - margin)
    substrate["block_radius"] = rng.uniform(8.0, 15.0)
    substrate["block_strength"] = rng.uniform(0.3, 0.7)

    substrate["lv_x"] = rng.uniform(margin, W - margin)
    substrate["lv_y"] = rng.uniform(margin, H - margin)
    substrate["lv_radius"] = rng.uniform(6.0, 14.0)
    substrate["lv_strength"] = rng.uniform(0.2, 0.6)

    substrate["vuln_x"] = rng.uniform(margin, W - margin)
    substrate["vuln_y"] = rng.uniform(margin, H - margin)
    substrate["vuln_radius"] = rng.uniform(8.0, 18.0)
    substrate["vuln_strength"] = rng.uniform(0.2, 0.8)

    return substrate


# =========================================================
# 4) Local patch geometry
# =========================================================

def build_local_patch_geometry(config: DatasetConfig) -> pd.DataFrame:
    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing

    x = np.arange(cols) * spacing
    y = np.arange(rows) * spacing

    x = x - np.mean(x)
    y = y - np.mean(y)

    records = []
    electrode_id = 0

    for r, yy in enumerate(y):
        for c, xx in enumerate(x):
            records.append({
                "electrode_id": electrode_id,
                "row": r,
                "col": c,
                "x_local": float(xx),
                "y_local": float(yy),
            })
            electrode_id += 1

    return pd.DataFrame(records)


# =========================================================
# 5) Place patch on global domain
# =========================================================

def place_patch_on_global_domain(local_patch_df: pd.DataFrame,
                                 center_x: float,
                                 center_y: float,
                                 angle_deg: float) -> pd.DataFrame:
    df = local_patch_df.copy()

    R = rotation_matrix(angle_deg)
    local_xy = df[["x_local", "y_local"]].to_numpy()
    rotated_xy = local_xy @ R.T

    df["x_global"] = rotated_xy[:, 0] + center_x
    df["y_global"] = rotated_xy[:, 1] + center_y

    return df


# =========================================================
# 6) Trajectory generation
# =========================================================

def generate_trajectory(config: DatasetConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_placements
    W = config.domain_width
    H = config.domain_height

    local_patch_df = build_local_patch_geometry(config)
    local_xy = local_patch_df[["x_local", "y_local"]].to_numpy()
    max_radius = np.sqrt((local_xy ** 2).sum(axis=1)).max()

    margin_x = max_radius + 2.0
    margin_y = max_radius + 2.0

    usable_w = W - 2 * margin_x
    usable_h = H - 2 * margin_y

    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    xs = np.linspace(margin_x, margin_x + usable_w, n_cols)
    ys = np.linspace(margin_y, margin_y + usable_h, n_rows)

    centers = []
    for ridx, y in enumerate(ys):
        row_xs = xs if ridx % 2 == 0 else xs[::-1]
        for x in row_xs:
            centers.append((x, y))

    centers = centers[:n]

    records = []
    for k, (x0, y0) in enumerate(centers):
        x = x0 + rng.normal(0.0, config.placement_jitter)
        y = y0 + rng.normal(0.0, config.placement_jitter)

        x = clamp(x, margin_x, W - margin_x)
        y = clamp(y, margin_y, H - margin_y)

        angle = rng.normal(0.0, config.angle_jitter_deg)

        records.append({
            "placement_id": k,
            "time_index": k,
            "time_min": float(k * config.minutes_per_step),
            "center_x": float(x),
            "center_y": float(y),
            "angle_deg": float(angle),
        })

    return pd.DataFrame(records)


# =========================================================
# 7) Raw electrode-level feature generation
# =========================================================

def compute_raw_electrode_features(patch_df: pd.DataFrame,
                                   substrate: dict,
                                   config: DatasetConfig,
                                   rng: np.random.Generator) -> pd.DataFrame:
    df = patch_df.copy()

    x = df["x_global"].to_numpy()
    y = df["y_global"].to_numpy()

    d1 = euclidean_distance(x, y, substrate["source1_x"], substrate["source1_y"])
    d2 = euclidean_distance(x, y, substrate["source2_x"], substrate["source2_y"])

    S1 = np.exp(-d1 / config.sigma_I)
    S2 = np.exp(-d2 / config.sigma_I)
    interaction = S1 * S2

    block_field = radial_bump(
        x, y,
        substrate["block_x"], substrate["block_y"],
        substrate["block_radius"], substrate["block_strength"]
    )

    lv_field = radial_bump(
        x, y,
        substrate["lv_x"], substrate["lv_y"],
        substrate["lv_radius"], substrate["lv_strength"]
    )

    vuln_field = radial_bump(
        x, y,
        substrate["vuln_x"], substrate["vuln_y"],
        substrate["vuln_radius"], substrate["vuln_strength"]
    )

    v_eff = config.base_cv * (1.0 - block_field)
    v_eff = np.clip(v_eff, config.min_cv, None)

    T1 = d1 / v_eff
    T2 = d2 / v_eff

    lat_jitter = rng.normal(0.0, config.lat_noise_std, size=len(df))
    LAT = np.minimum(T1, T2) + vuln_field * lat_jitter
    LAT += rng.normal(0.0, 0.5, size=len(df))
    LAT = np.clip(LAT, 0.0, None)

    A = (
        config.amp_scale_1 * np.exp(-d1 / config.sigma_A)
        + config.amp_scale_2 * np.exp(-d2 / config.sigma_A)
        - lv_field
        + rng.normal(0.0, config.amp_noise_std, size=len(df))
    )
    A = np.clip(A, 0.0, None)

    FI = (
        config.fi_base
        + config.fi_alpha1 * S1
        + config.fi_alpha2 * S2
        + config.fi_beta * interaction
        + config.fi_gamma * vuln_field
        + rng.normal(0.0, config.fi_noise_std, size=len(df))
    )
    FI = np.clip(FI, 0.0, 1.0)

    DF = (
        config.df_base
        + config.df_lambda1 * S1
        + config.df_lambda2 * S2
        + config.df_mu * interaction
        + config.df_nu * vuln_field
        + rng.normal(0.0, config.df_noise_std, size=len(df))
    )
    DF = np.clip(DF, 3.0, 12.0)

    df["LAT"] = LAT
    df["A"] = A
    df["FI"] = FI
    df["DF"] = DF
    df["interaction"] = interaction

    return df


# =========================================================
# 8) Derived local features
# =========================================================

def compute_patch_gradients(lat_grid: np.ndarray, spacing: float):
    grad_y, grad_x = np.gradient(lat_grid, spacing, spacing)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = np.arctan2(grad_y, grad_x)
    return grad_x, grad_y, grad_mag, grad_angle


def compute_dispersion_for_patch(fi_grid: np.ndarray,
                                 a_grid: np.ndarray,
                                 config: DatasetConfig) -> np.ndarray:
    rows, cols = fi_grid.shape
    disp = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            fi_vals = []
            a_vals = []

            for rr in range(max(0, r - 1), min(rows, r + 2)):
                for cc in range(max(0, c - 1), min(cols, c + 2)):
                    if rr == r and cc == c:
                        continue
                    fi_vals.append(fi_grid[rr, cc])
                    a_vals.append(a_grid[rr, cc])

            fi_var = np.var(fi_vals) if fi_vals else 0.0
            a_var = np.var(a_vals) if a_vals else 0.0

            disp[r, c] = config.disp_c1 * fi_var + config.disp_c2 * a_var

    return disp


def compute_local_derived_features(patch_df: pd.DataFrame,
                                   config: DatasetConfig) -> pd.DataFrame:
    df = patch_df.copy().sort_values(["row", "col"]).reset_index(drop=True)

    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing

    lat_grid = df["LAT"].to_numpy().reshape(rows, cols)
    fi_grid = df["FI"].to_numpy().reshape(rows, cols)
    a_grid = df["A"].to_numpy().reshape(rows, cols)

    grad_x, grad_y, grad_mag, grad_angle = compute_patch_gradients(lat_grid, spacing)
    disp_grid = compute_dispersion_for_patch(fi_grid, a_grid, config)

    df["LAT_grad_x"] = grad_x.reshape(-1)
    df["LAT_grad_y"] = grad_y.reshape(-1)
    df["LAT_grad_mag"] = grad_mag.reshape(-1)
    df["LAT_grad_angle"] = grad_angle.reshape(-1)
    df["Disp"] = disp_grid.reshape(-1)

    return df


# =========================================================
# 9) Patch-level descriptors
# =========================================================

def compute_std_for_patch(lat_grid: np.ndarray, spacing: float) -> float:
    rows, cols = lat_grid.shape
    vals = []

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                vals.append(abs(lat_grid[r, c] - lat_grid[r, c + 1]) / spacing)
            if r + 1 < rows:
                vals.append(abs(lat_grid[r, c] - lat_grid[r + 1, c]) / spacing)

    return float(np.mean(vals)) if vals else 0.0


def compute_ics_for_patch(lat_grid: np.ndarray,
                          spacing: float,
                          config: DatasetConfig) -> float:
    lat_smooth = smooth_grid_3x3(lat_grid)

    grad_x, grad_y, grad_mag, grad_angle = compute_patch_gradients(lat_smooth, spacing)
    theta = np.mod(grad_angle, np.pi)

    rows, cols = lat_grid.shape
    edge_vals = []
    max_grad = max(float(np.max(grad_mag)), config.grad_eps)

    def angle_diff_axial(a, b):
        raw = abs(a - b)
        return min(raw, np.pi - raw)

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                g1 = grad_mag[r, c]
                g2 = grad_mag[r, c + 1]
                w = 0.0 if (g1 < config.grad_eps or g2 < config.grad_eps) else (g1 * g2) / (max_grad ** 2)
                dtheta = angle_diff_axial(theta[r, c], theta[r, c + 1])
                edge_vals.append(w * dtheta)

            if r + 1 < rows:
                g1 = grad_mag[r, c]
                g2 = grad_mag[r + 1, c]
                w = 0.0 if (g1 < config.grad_eps or g2 < config.grad_eps) else (g1 * g2) / (max_grad ** 2)
                dtheta = angle_diff_axial(theta[r, c], theta[r + 1, c])
                edge_vals.append(w * dtheta)

    return float(np.mean(edge_vals)) if edge_vals else 0.0


def compute_patch_descriptors_for_one_patch(patch_df: pd.DataFrame,
                                            config: DatasetConfig) -> dict:
    df = patch_df.sort_values(["row", "col"]).reset_index(drop=True)
    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing

    lat_grid = df["LAT"].to_numpy().reshape(rows, cols)

    std_p = compute_std_for_patch(lat_grid, spacing)
    ics_p = compute_ics_for_patch(lat_grid, spacing, config)
    disp_p = float(df["Disp"].mean())
    fi_burden = float(df["FI"].mean())
    lv_burden = float((df["A"] < config.low_voltage_threshold).mean())

    return {
        "STD_p": std_p,
        "ICS_p": ics_p,
        "Disp_p": disp_p,
        "FI_burden": fi_burden,
        "LV_burden": lv_burden,
    }


def normalize_and_score_patch_table(patch_df: pd.DataFrame,
                                    config: DatasetConfig) -> pd.DataFrame:
    out = patch_df.copy()

    out["STD_norm"] = minmax_normalize(out["STD_p"])
    out["ICS_norm"] = minmax_normalize(out["ICS_p"])
    out["Disp_norm"] = minmax_normalize(out["Disp_p"])
    out["FI_norm"] = minmax_normalize(out["FI_burden"])
    out["LV_norm"] = minmax_normalize(out["LV_burden"])

    out["DIC_p"] = (
        config.alpha * out["Disp_norm"]
        + config.beta * out["ICS_norm"]
        + config.gamma * (out["Disp_norm"] * out["ICS_norm"])
    )

    out["DIC_norm"] = minmax_normalize(out["DIC_p"])

    out["PatchScore"] = (
        config.w1 * out["DIC_norm"]
        + config.w2 * out["STD_norm"]
        + config.w3 * out["FI_norm"]
        + config.w4 * out["LV_norm"]
    )

    return out


# =========================================================
# 10) Global maps
# =========================================================

def build_global_grid(config: DatasetConfig):
    xs = np.arange(0.0, config.domain_width + config.global_grid_step, config.global_grid_step)
    ys = np.arange(0.0, config.domain_height + config.global_grid_step, config.global_grid_step)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, X, Y


def gaussian_kernel_2d(X, Y, cx, cy, sigma):
    return (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)
    )


def compute_global_maps(patch_df: pd.DataFrame, config: DatasetConfig, eps: float = 1e-8):
    xs, ys, X, Y = build_global_grid(config)

    raw_global = np.zeros_like(X, dtype=float)
    confidence = np.zeros_like(X, dtype=float)

    for _, row in patch_df.iterrows():
        K = gaussian_kernel_2d(X, Y, row["center_x"], row["center_y"], config.sigma_kernel)
        raw_global += row["PatchScore"] * K
        confidence += K

    normalized_score = raw_global / (confidence + eps)

    conf_thr = config.confidence_threshold_ratio * float(np.max(confidence))
    masked_score = np.where(confidence >= conf_thr, normalized_score, np.nan)

    return xs, ys, raw_global, confidence, normalized_score, masked_score, conf_thr


# =========================================================
# 11) Pipeline
# =========================================================

def run_pipeline_for_config(config: DatasetConfig, substrate: dict, placement_df: pd.DataFrame):
    local_patch_df = build_local_patch_geometry(config)
    rng = np.random.default_rng(config.random_seed)

    patch_rows = []

    for _, placement in placement_df.iterrows():
        patch_df = place_patch_on_global_domain(
            local_patch_df=local_patch_df,
            center_x=placement["center_x"],
            center_y=placement["center_y"],
            angle_deg=placement["angle_deg"],
        )

        patch_df = compute_raw_electrode_features(
            patch_df=patch_df,
            substrate=substrate,
            config=config,
            rng=rng
        )

        patch_df = compute_local_derived_features(
            patch_df=patch_df,
            config=config
        )

        desc = compute_patch_descriptors_for_one_patch(patch_df, config)
        desc.update({
            "placement_id": int(placement["placement_id"]),
            "time_index": int(placement["time_index"]),
            "time_min": float(placement["time_min"]),
            "center_x": float(placement["center_x"]),
            "center_y": float(placement["center_y"]),
            "angle_deg": float(placement["angle_deg"]),
        })
        patch_rows.append(desc)

    patch_summary_df = pd.DataFrame(patch_rows)
    patch_summary_df = normalize_and_score_patch_table(patch_summary_df, config)

    xs, ys, raw_global, confidence, normalized_score, masked_score, conf_thr = compute_global_maps(
        patch_df=patch_summary_df,
        config=config
    )

    summary = {
        "low_voltage_threshold": config.low_voltage_threshold,
        "sigma_kernel": config.sigma_kernel,
        "patchscore_mean": float(patch_summary_df["PatchScore"].mean()),
        "patchscore_max": float(patch_summary_df["PatchScore"].max()),
        "maskedscore_max": float(np.nanmax(masked_score)),
        "confidence_max": float(np.max(confidence)),
        "confidence_threshold": float(conf_thr),
        "n_masked_valid_pixels": int(np.sum(~np.isnan(masked_score))),
    }

    return patch_summary_df, xs, ys, masked_score, summary


# =========================================================
# 12) Plotting
# =========================================================

def plot_experiment_grid(results, title, output_path: Path):
    n = len(results)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 4 * n))

    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, results):
        masked_score = item["masked_score"]
        config = item["config"]
        placement_df = item["placement_df"]

        extent = [0, config.domain_width, 0, config.domain_height]
        im = ax.imshow(masked_score, origin="lower", extent=extent, aspect="auto")
        ax.scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x")
        ax.set_title(
            f"LV thr={config.low_voltage_threshold:.2f}, sigma_kernel={config.sigma_kernel:.1f}"
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 13) Main
# =========================================================

if __name__ == "__main__":
    base_config = DatasetConfig(
        n_samples=1,
        n_placements=10,
        random_seed=42,
        low_voltage_threshold=0.50,
        sigma_kernel=6.0
    )

    rng = np.random.default_rng(base_config.random_seed)

    # Keep the same hidden world and placement path for fair comparison
    substrate = generate_hidden_substrate(base_config, rng)
    placement_df = generate_trajectory(base_config, rng)

    output_dir = Path("manual_check_ofat_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Experiment A: vary sigma only
    # -----------------------------------------------------
    sigma_values = [4.0, 6.0, 8.0]
    sigma_results = []
    sigma_summary_rows = []

    for sigma in sigma_values:
        cfg = replace(base_config, sigma_kernel=sigma, low_voltage_threshold=0.50)
        patch_df, xs, ys, masked_score, summary = run_pipeline_for_config(
            config=cfg,
            substrate=substrate,
            placement_df=placement_df
        )

        sigma_results.append({
            "config": cfg,
            "placement_df": placement_df,
            "masked_score": masked_score,
        })
        sigma_summary_rows.append(summary)

        patch_df.to_csv(output_dir / f"sigma_patch_summary_{sigma:.1f}.csv", index=False)
        pd.DataFrame(masked_score, index=ys, columns=xs).to_csv(output_dir / f"sigma_masked_map_{sigma:.1f}.csv")

    sigma_summary_df = pd.DataFrame(sigma_summary_rows)
    sigma_summary_df.to_csv(output_dir / "sigma_only_summary.csv", index=False)

    plot_experiment_grid(
        sigma_results,
        title="Sensitivity A: sigma_kernel only (LV threshold fixed at 0.50)",
        output_path=output_dir / "sigma_only_maps.png"
    )

    # -----------------------------------------------------
    # Experiment B: vary LV threshold only
    # -----------------------------------------------------
    lv_values = [0.35, 0.50, 0.65]
    lv_results = []
    lv_summary_rows = []

    for lv in lv_values:
        cfg = replace(base_config, sigma_kernel=6.0, low_voltage_threshold=lv)
        patch_df, xs, ys, masked_score, summary = run_pipeline_for_config(
            config=cfg,
            substrate=substrate,
            placement_df=placement_df
        )

        lv_results.append({
            "config": cfg,
            "placement_df": placement_df,
            "masked_score": masked_score,
        })
        lv_summary_rows.append(summary)

        patch_df.to_csv(output_dir / f"lv_patch_summary_{lv:.2f}.csv", index=False)
        pd.DataFrame(masked_score, index=ys, columns=xs).to_csv(output_dir / f"lv_masked_map_{lv:.2f}.csv")

    lv_summary_df = pd.DataFrame(lv_summary_rows)
    lv_summary_df.to_csv(output_dir / "lv_only_summary.csv", index=False)

    plot_experiment_grid(
        lv_results,
        title="Sensitivity B: low_voltage_threshold only (sigma_kernel fixed at 6.0)",
        output_path=output_dir / "lv_only_maps.png"
    )

    print("OFAT sensitivity-analysis files saved to: manual_check_ofat_sensitivity/")

    print("\nSigma-only summary:")
    print(sigma_summary_df)

    print("\nLV-threshold-only summary:")
    print(lv_summary_df)