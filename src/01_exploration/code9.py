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
    low_voltage_threshold: float = 0.50
    grad_eps: float = 1e-6

    # Global map grid
    global_grid_step: float = 2.0

    # Confidence masking
    confidence_threshold_ratio: float = 0.15

    # Hotspot extraction
    hotspot_quantile: float = 0.90   # top 10% of valid masked values

    # PatchScore weights
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    # DIC coefficients
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2

    # Seeds
    substrate_seed: int = 42
    feature_seed: int = 999


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

    return xs, ys, masked_score, confidence, conf_thr


# =========================================================
# 11) Hotspot analysis
# =========================================================

def analyze_hotspot(masked_score: np.ndarray, xs: np.ndarray, ys: np.ndarray, quantile: float):
    valid_vals = masked_score[~np.isnan(masked_score)]
    if valid_vals.size == 0:
        return {
            "hotspot_threshold": np.nan,
            "hotspot_area_pixels": 0,
            "hotspot_centroid_x": np.nan,
            "hotspot_centroid_y": np.nan,
            "peak_x": np.nan,
            "peak_y": np.nan,
            "peak_score": np.nan,
            "hotspot_mask": np.zeros_like(masked_score, dtype=bool),
        }

    thr = np.quantile(valid_vals, quantile)
    hotspot_mask = (~np.isnan(masked_score)) & (masked_score >= thr)

    ys_grid, xs_grid = np.meshgrid(ys, xs, indexing="ij")

    weights = np.where(hotspot_mask, masked_score, 0.0)
    total_w = np.sum(weights)

    if total_w > 0:
        centroid_x = float(np.sum(xs_grid * weights) / total_w)
        centroid_y = float(np.sum(ys_grid * weights) / total_w)
    else:
        centroid_x = np.nan
        centroid_y = np.nan

    max_idx = np.nanargmax(masked_score)
    peak_r, peak_c = np.unravel_index(max_idx, masked_score.shape)
    peak_x = float(xs[peak_c])
    peak_y = float(ys[peak_r])
    peak_score = float(masked_score[peak_r, peak_c])

    return {
        "hotspot_threshold": float(thr),
        "hotspot_area_pixels": int(np.sum(hotspot_mask)),
        "hotspot_centroid_x": centroid_x,
        "hotspot_centroid_y": centroid_y,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_score": peak_score,
        "hotspot_mask": hotspot_mask,
    }


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(inter / union)


# =========================================================
# 12) Pipeline for one trajectory
# =========================================================

def run_pipeline_for_trajectory(config: DatasetConfig,
                                substrate: dict,
                                trajectory_seed: int):
    traj_rng = np.random.default_rng(trajectory_seed)
    feature_rng = np.random.default_rng(config.feature_seed)

    placement_df = generate_trajectory(config, traj_rng)
    local_patch_df = build_local_patch_geometry(config)

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
            rng=feature_rng
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

    xs, ys, masked_score, confidence, conf_thr = compute_global_maps(
        patch_df=patch_summary_df,
        config=config
    )

    hotspot_info = analyze_hotspot(
        masked_score=masked_score,
        xs=xs,
        ys=ys,
        quantile=config.hotspot_quantile
    )

    summary = {
        "trajectory_seed": trajectory_seed,
        "patchscore_mean": float(patch_summary_df["PatchScore"].mean()),
        "patchscore_max": float(patch_summary_df["PatchScore"].max()),
        "confidence_threshold": float(conf_thr),
        "n_masked_valid_pixels": int(np.sum(~np.isnan(masked_score))),
        "hotspot_threshold": hotspot_info["hotspot_threshold"],
        "hotspot_area_pixels": hotspot_info["hotspot_area_pixels"],
        "hotspot_centroid_x": hotspot_info["hotspot_centroid_x"],
        "hotspot_centroid_y": hotspot_info["hotspot_centroid_y"],
        "peak_x": hotspot_info["peak_x"],
        "peak_y": hotspot_info["peak_y"],
        "peak_score": hotspot_info["peak_score"],
    }

    return placement_df, patch_summary_df, xs, ys, masked_score, hotspot_info, summary


# =========================================================
# 13) Plotting
# =========================================================

def plot_trajectory_robustness(results, config: DatasetConfig, output_path: Path):
    n = len(results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, results):
        masked_score = item["masked_score"]
        placement_df = item["placement_df"]
        hotspot_info = item["hotspot_info"]
        seed = item["trajectory_seed"]

        extent = [0, config.domain_width, 0, config.domain_height]
        im = ax.imshow(masked_score, origin="lower", extent=extent, aspect="auto")
        ax.scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x", label="placements")

        ax.scatter(
            hotspot_info["hotspot_centroid_x"],
            hotspot_info["hotspot_centroid_y"],
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="hotspot centroid"
        )

        ax.scatter(
            hotspot_info["peak_x"],
            hotspot_info["peak_y"],
            s=70,
            marker="*",
            label="peak"
        )

        ax.set_title(f"Trajectory seed = {seed}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle("Trajectory Robustness: Masked Suspicion Maps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 14) Main
# =========================================================

if __name__ == "__main__":
    config = DatasetConfig()

    output_dir = Path("manual_check_trajectory_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix hidden world
    substrate_rng = np.random.default_rng(config.substrate_seed)
    substrate = generate_hidden_substrate(config, substrate_rng)

    trajectory_seeds = [101, 202, 303, 404]

    results = []
    summary_rows = []

    for seed in trajectory_seeds:
        placement_df, patch_df, xs, ys, masked_score, hotspot_info, summary = run_pipeline_for_trajectory(
            config=config,
            substrate=substrate,
            trajectory_seed=seed
        )

        results.append({
            "trajectory_seed": seed,
            "placement_df": placement_df,
            "patch_df": patch_df,
            "xs": xs,
            "ys": ys,
            "masked_score": masked_score,
            "hotspot_info": hotspot_info,
        })
        summary_rows.append(summary)

        patch_df.to_csv(output_dir / f"patch_summary_seed_{seed}.csv", index=False)
        pd.DataFrame(masked_score, index=ys, columns=xs).to_csv(output_dir / f"masked_map_seed_{seed}.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "trajectory_robustness_summary.csv", index=False)

    # Pairwise comparisons
    pairwise_rows = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]

            cdx = a["hotspot_info"]["hotspot_centroid_x"] - b["hotspot_info"]["hotspot_centroid_x"]
            cdy = a["hotspot_info"]["hotspot_centroid_y"] - b["hotspot_info"]["hotspot_centroid_y"]
            centroid_distance = float(np.sqrt(cdx ** 2 + cdy ** 2))

            pdx = a["hotspot_info"]["peak_x"] - b["hotspot_info"]["peak_x"]
            pdy = a["hotspot_info"]["peak_y"] - b["hotspot_info"]["peak_y"]
            peak_distance = float(np.sqrt(pdx ** 2 + pdy ** 2))

            iou = mask_iou(a["hotspot_info"]["hotspot_mask"], b["hotspot_info"]["hotspot_mask"])

            pairwise_rows.append({
                "seed_a": a["trajectory_seed"],
                "seed_b": b["trajectory_seed"],
                "centroid_distance_mm": centroid_distance,
                "peak_distance_mm": peak_distance,
                "hotspot_iou": iou,
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(output_dir / "trajectory_pairwise_metrics.csv", index=False)

    plot_trajectory_robustness(
        results=results,
        config=config,
        output_path=output_dir / "trajectory_robustness_maps.png"
    )

    print("Trajectory-robustness files saved to: manual_check_trajectory_robustness/")

    print("\nTrajectory robustness summary:")
    print(summary_df)

    print("\nTrajectory pairwise metrics:")
    print(pairwise_df)