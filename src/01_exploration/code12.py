import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass
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

    # Sequential mapping
    n_placements: int = 10
    minutes_per_step: float = 25.0
    placement_jitter: float = 2.0
    angle_jitter_deg: float = 20.0

    # Hidden substrate
    n_sources: int = 2
    base_cv: float = 0.7
    min_cv: float = 0.20

    # Raw feature params
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

    # Dispersion
    disp_c1: float = 0.5
    disp_c2: float = 0.5

    # Global aggregation
    sigma_kernel: float = 6.0
    global_grid_step: float = 2.0
    confidence_threshold_ratio: float = 0.15

    # Low voltage
    low_voltage_threshold: float = 0.50

    # Gradient reliability
    grad_eps: float = 1e-6

    # Hotspot
    hotspot_quantile: float = 0.90

    # Spontaneous PatchScore weights
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    # DIC coefficients
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2

    # Paced score weights
    wp1: float = 0.40
    wp2: float = 0.30
    wp3: float = 0.20
    wp4: float = 0.10

    # Paced branch
    pacing_x: float = 6.0
    pacing_y: float = 40.0
    paced_lat_noise_std: float = 0.5
    paced_block_penalty: float = 0.85
    paced_lv_penalty: float = 0.10
    paced_vuln_penalty: float = 0.20
    ldh_threshold_ms: float = 12.0

    # Multilayer graph parameters
    graph_spatial_sigma: float = 25.0
    graph_score_tau: float = 0.15
    graph_w_spatial: float = 0.45
    graph_w_temporal: float = 0.20
    graph_w_score: float = 0.35
    graph_self_loop: float = 0.50
    multilayer_discord_penalty: float = 0.50

    # Seeds
    substrate_seed: int = 42
    trajectory_seed: int = 101
    feature_seed_spont: int = 999
    feature_seed_paced: int = 111


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


def minmax_normalize_series(series: pd.Series, eps: float = 1e-8) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + eps)


def minmax_normalize_array(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = arr.copy().astype(float)
    valid = ~np.isnan(out)
    if np.sum(valid) == 0:
        return out
    mn = np.nanmin(out)
    mx = np.nanmax(out)
    out[valid] = (out[valid] - mn) / (mx - mn + eps)
    return out


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


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(inter / union)


def weighted_overlap_score(masked_score: np.ndarray, truth_mask: np.ndarray) -> float:
    vals = np.where(truth_mask, masked_score, np.nan)
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


def point_in_circle(px, py, cx, cy, radius):
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2


def row_normalize_matrix(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row_sums = A.sum(axis=1, keepdims=True)
    return A / (row_sums + eps)


# =========================================================
# 3) Hidden substrate
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
# 4) Geometry / trajectory
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
        x = clamp(x0 + rng.normal(0.0, config.placement_jitter), margin_x, W - margin_x)
        y = clamp(y0 + rng.normal(0.0, config.placement_jitter), margin_y, H - margin_y)
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
# 5) Field helpers
# =========================================================

def compute_substrate_fields(x, y, substrate, config: DatasetConfig):
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

    return d1, d2, S1, S2, interaction, block_field, lv_field, vuln_field


# =========================================================
# 6) Raw feature generation: spontaneous
# =========================================================

def compute_spontaneous_raw_features(patch_df, substrate, config, rng):
    df = patch_df.copy()
    x = df["x_global"].to_numpy()
    y = df["y_global"].to_numpy()

    d1, d2, S1, S2, interaction, block_field, lv_field, vuln_field = compute_substrate_fields(
        x, y, substrate, config
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
    df["block_field"] = block_field
    df["lv_field"] = lv_field
    df["vuln_field"] = vuln_field
    return df


# =========================================================
# 7) Raw feature generation: paced
# =========================================================

def compute_paced_raw_features(patch_df, substrate, config, rng):
    df = patch_df.copy()
    x = df["x_global"].to_numpy()
    y = df["y_global"].to_numpy()

    d1, d2, S1, S2, interaction, block_field, lv_field, vuln_field = compute_substrate_fields(
        x, y, substrate, config
    )

    d_pace = euclidean_distance(x, y, config.pacing_x, config.pacing_y)

    v_eff_paced = config.base_cv * (
        1.0
        - config.paced_block_penalty * block_field
        - config.paced_lv_penalty * lv_field
        - config.paced_vuln_penalty * vuln_field
    )
    v_eff_paced = np.clip(v_eff_paced, config.min_cv, None)

    LAT_paced = d_pace / v_eff_paced
    LAT_paced += rng.normal(0.0, config.paced_lat_noise_std, size=len(df))
    LAT_paced = np.clip(LAT_paced, 0.0, None)

    A = (
        config.amp_scale_1 * np.exp(-d1 / config.sigma_A)
        + config.amp_scale_2 * np.exp(-d2 / config.sigma_A)
        - lv_field
        + rng.normal(0.0, config.amp_noise_std, size=len(df))
    )
    A = np.clip(A, 0.0, None)

    FI_paced = (
        0.10
        + 0.08 * block_field
        + 0.05 * vuln_field
        + 0.04 * lv_field
        + rng.normal(0.0, 0.02, size=len(df))
    )
    FI_paced = np.clip(FI_paced, 0.0, 1.0)

    df["LAT"] = LAT_paced
    df["A"] = A
    df["FI"] = FI_paced
    df["interaction"] = interaction
    df["block_field"] = block_field
    df["lv_field"] = lv_field
    df["vuln_field"] = vuln_field
    df["d_pace"] = d_pace
    return df


# =========================================================
# 8) Derived local features
# =========================================================

def compute_patch_gradients(lat_grid: np.ndarray, spacing: float):
    grad_y, grad_x = np.gradient(lat_grid, spacing, spacing)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = np.arctan2(grad_y, grad_x)
    return grad_x, grad_y, grad_mag, grad_angle


def compute_dispersion_for_patch(fi_grid, a_grid, config):
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


def compute_local_derived_features(patch_df, config):
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
# 9) Patch-level descriptors and scores
# =========================================================

def compute_std_for_patch(lat_grid, spacing):
    rows, cols = lat_grid.shape
    vals = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                vals.append(abs(lat_grid[r, c] - lat_grid[r, c + 1]) / spacing)
            if r + 1 < rows:
                vals.append(abs(lat_grid[r, c] - lat_grid[r + 1, c]) / spacing)
    return float(np.mean(vals)) if vals else 0.0


def compute_ldh_for_patch(lat_grid, threshold_ms):
    rows, cols = lat_grid.shape
    flags = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                flags.append(float(abs(lat_grid[r, c] - lat_grid[r, c + 1]) >= threshold_ms))
            if r + 1 < rows:
                flags.append(float(abs(lat_grid[r, c] - lat_grid[r + 1, c]) >= threshold_ms))
    return float(np.mean(flags)) if flags else 0.0


def compute_ics_for_patch(lat_grid, spacing, config):
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


def compute_spontaneous_patch_descriptors(patch_df, config):
    df = patch_df.sort_values(["row", "col"]).reset_index(drop=True)
    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing
    lat_grid = df["LAT"].to_numpy().reshape(rows, cols)

    return {
        "STD_p": compute_std_for_patch(lat_grid, spacing),
        "ICS_p": compute_ics_for_patch(lat_grid, spacing, config),
        "Disp_p": float(df["Disp"].mean()),
        "FI_burden": float(df["FI"].mean()),
        "LV_burden": float((df["A"] < config.low_voltage_threshold).mean()),
    }


def compute_paced_patch_descriptors(patch_df, config):
    df = patch_df.sort_values(["row", "col"]).reset_index(drop=True)
    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing
    lat_grid = df["LAT"].to_numpy().reshape(rows, cols)

    return {
        "STD_paced": compute_std_for_patch(lat_grid, spacing),
        "ICS_paced": compute_ics_for_patch(lat_grid, spacing, config),
        "LDH_paced": compute_ldh_for_patch(lat_grid, config.ldh_threshold_ms),
        "LV_burden": float((df["A"] < config.low_voltage_threshold).mean()),
    }


def normalize_and_score_spontaneous_patch_table(patch_df, config):
    out = patch_df.copy()

    out["STD_norm"] = minmax_normalize_series(out["STD_p"])
    out["ICS_norm"] = minmax_normalize_series(out["ICS_p"])
    out["Disp_norm"] = minmax_normalize_series(out["Disp_p"])
    out["FI_norm"] = minmax_normalize_series(out["FI_burden"])
    out["LV_norm"] = minmax_normalize_series(out["LV_burden"])

    out["DIC_p"] = (
        config.alpha * out["Disp_norm"]
        + config.beta * out["ICS_norm"]
        + config.gamma * (out["Disp_norm"] * out["ICS_norm"])
    )
    out["DIC_norm"] = minmax_normalize_series(out["DIC_p"])

    out["SpontScore"] = (
        config.w1 * out["DIC_norm"]
        + config.w2 * out["STD_norm"]
        + config.w3 * out["FI_norm"]
        + config.w4 * out["LV_norm"]
    )
    return out


def normalize_and_score_paced_patch_table(patch_df, config):
    out = patch_df.copy()

    out["STD_norm"] = minmax_normalize_series(out["STD_paced"])
    out["ICS_norm"] = minmax_normalize_series(out["ICS_paced"])
    out["LDH_norm"] = minmax_normalize_series(out["LDH_paced"])
    out["LV_norm"] = minmax_normalize_series(out["LV_burden"])

    out["PacedScore"] = (
        config.wp1 * out["STD_norm"]
        + config.wp2 * out["LDH_norm"]
        + config.wp3 * out["LV_norm"]
        + config.wp4 * out["ICS_norm"]
    )
    return out


# =========================================================
# 10) Global map builders
# =========================================================

def build_global_grid(config):
    xs = np.arange(0.0, config.domain_width + config.global_grid_step, config.global_grid_step)
    ys = np.arange(0.0, config.domain_height + config.global_grid_step, config.global_grid_step)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, X, Y


def gaussian_kernel_2d(X, Y, cx, cy, sigma):
    return (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2)
    )


def compute_global_map_from_patch_scores(patch_df, score_col, config, eps=1e-8):
    xs, ys, X, Y = build_global_grid(config)

    raw_global = np.zeros_like(X, dtype=float)
    confidence = np.zeros_like(X, dtype=float)

    for _, row in patch_df.iterrows():
        K = gaussian_kernel_2d(X, Y, row["center_x"], row["center_y"], config.sigma_kernel)
        raw_global += row[score_col] * K
        confidence += K

    normalized_score = raw_global / (confidence + eps)
    conf_thr = config.confidence_threshold_ratio * float(np.max(confidence))
    masked_score = np.where(confidence >= conf_thr, normalized_score, np.nan)

    return xs, ys, raw_global, confidence, normalized_score, masked_score, conf_thr


# =========================================================
# 11) Graph construction
# =========================================================

def build_spatial_similarity_matrix(centers_xy: np.ndarray, sigma: float) -> np.ndarray:
    n = centers_xy.shape[0]
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(centers_xy[i] - centers_xy[j])
            A[i, j] = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return A


def build_temporal_chain_matrix(time_index: np.ndarray) -> np.ndarray:
    n = len(time_index)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(int(time_index[i]) - int(time_index[j])) == 1:
                A[i, j] = 1.0
    return A


def build_score_similarity_matrix(scores: np.ndarray, tau: float) -> np.ndarray:
    n = len(scores)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A[i, j] = np.exp(-abs(scores[i] - scores[j]) / max(tau, 1e-8))
    return A


def build_layer_adjacency(centers_xy, time_index, scores, config):
    A_spatial = build_spatial_similarity_matrix(centers_xy, config.graph_spatial_sigma)
    A_temporal = build_temporal_chain_matrix(time_index)
    A_score = build_score_similarity_matrix(scores, config.graph_score_tau)

    A = (
        config.graph_w_spatial * A_spatial
        + config.graph_w_temporal * A_temporal
        + config.graph_w_score * A_score
    )
    A += config.graph_self_loop * np.eye(len(scores))
    A = row_normalize_matrix(A)
    return A, A_spatial, A_temporal, A_score


# =========================================================
# 12) Multilayer fusion
# =========================================================

def build_multilayer_node_table(spont_patch_df, paced_patch_df, config):
    df = spont_patch_df[[
        "placement_id", "time_index", "time_min", "center_x", "center_y", "angle_deg", "SpontScore"
    ]].copy()

    df = df.merge(
        paced_patch_df[["placement_id", "PacedScore"]],
        on="placement_id",
        how="inner"
    )

    centers_xy = df[["center_x", "center_y"]].to_numpy()
    time_index = df["time_index"].to_numpy()

    A_sp, A_spatial, A_temporal, A_spscore = build_layer_adjacency(
        centers_xy, time_index, df["SpontScore"].to_numpy(), config
    )
    A_pc, _, _, A_pcscore = build_layer_adjacency(
        centers_xy, time_index, df["PacedScore"].to_numpy(), config
    )

    sp_smooth = A_sp @ df["SpontScore"].to_numpy()
    pc_smooth = A_pc @ df["PacedScore"].to_numpy()

    concord_raw = np.sqrt(np.clip(sp_smooth, 0, None) * np.clip(pc_smooth, 0, None))
    discord_raw = np.abs(sp_smooth - pc_smooth)

    core_raw = concord_raw - config.multilayer_discord_penalty * discord_raw

    df["SpontSmooth"] = sp_smooth
    df["PacedSmooth"] = pc_smooth
    df["ConcordRaw"] = concord_raw
    df["DiscordRaw"] = discord_raw
    df["CoreRaw"] = core_raw

    df["ConcordNorm"] = minmax_normalize_series(pd.Series(concord_raw)).to_numpy()
    df["DiscordNorm"] = minmax_normalize_series(pd.Series(discord_raw)).to_numpy()
    df["CoreScore"] = minmax_normalize_series(pd.Series(core_raw)).to_numpy()
    df["DynamicScore"] = df["DiscordNorm"]

    graph_parts = {
        "A_sp": A_sp,
        "A_pc": A_pc,
        "A_spatial": A_spatial,
        "A_temporal": A_temporal,
        "A_spscore": A_spscore,
        "A_pcscore": A_pcscore,
    }

    return df, graph_parts


# =========================================================
# 13) Hotspot / truth analysis
# =========================================================

def analyze_hotspot(masked_score, xs, ys, quantile):
    valid_vals = masked_score[~np.isnan(masked_score)]
    if valid_vals.size == 0:
        return {
            "hotspot_threshold": np.nan,
            "hotspot_mask": np.zeros_like(masked_score, dtype=bool),
            "hotspot_area_pixels": 0,
            "hotspot_centroid_x": np.nan,
            "hotspot_centroid_y": np.nan,
            "peak_x": np.nan,
            "peak_y": np.nan,
            "peak_score": np.nan,
        }

    thr = np.quantile(valid_vals, quantile)
    hotspot_mask = (~np.isnan(masked_score)) & (masked_score >= thr)

    ys_grid, xs_grid = np.meshgrid(ys, xs, indexing="ij")
    weights = np.where(hotspot_mask, masked_score, 0.0)
    total_w = np.sum(weights)

    centroid_x = float(np.sum(xs_grid * weights) / total_w)
    centroid_y = float(np.sum(ys_grid * weights) / total_w)

    max_idx = np.nanargmax(masked_score)
    peak_r, peak_c = np.unravel_index(max_idx, masked_score.shape)
    peak_x = float(xs[peak_c])
    peak_y = float(ys[peak_r])
    peak_score = float(masked_score[peak_r, peak_c])

    return {
        "hotspot_threshold": float(thr),
        "hotspot_mask": hotspot_mask,
        "hotspot_area_pixels": int(np.sum(hotspot_mask)),
        "hotspot_centroid_x": centroid_x,
        "hotspot_centroid_y": centroid_y,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_score": peak_score,
    }


def build_truth_masks(xs, ys, substrate):
    X, Y = np.meshgrid(xs, ys)

    block_mask = ((X - substrate["block_x"]) ** 2 + (Y - substrate["block_y"]) ** 2) <= substrate["block_radius"] ** 2
    lv_mask = ((X - substrate["lv_x"]) ** 2 + (Y - substrate["lv_y"]) ** 2) <= substrate["lv_radius"] ** 2
    vuln_mask = ((X - substrate["vuln_x"]) ** 2 + (Y - substrate["vuln_y"]) ** 2) <= substrate["vuln_radius"] ** 2
    union_mask = block_mask | lv_mask | vuln_mask

    return {
        "block_mask": block_mask,
        "lv_mask": lv_mask,
        "vuln_mask": vuln_mask,
        "union_mask": union_mask,
    }


def validate_map_against_truth(branch_name, masked_score, hotspot, xs, ys, substrate):
    truth_masks = build_truth_masks(xs, ys, substrate)

    metrics = {
        "branch": branch_name,
        "peak_score": hotspot["peak_score"],
        "hotspot_area_pixels": hotspot["hotspot_area_pixels"],

        "centroid_to_block_mm": euclidean_distance(
            hotspot["hotspot_centroid_x"], hotspot["hotspot_centroid_y"],
            substrate["block_x"], substrate["block_y"]
        ),
        "centroid_to_lv_mm": euclidean_distance(
            hotspot["hotspot_centroid_x"], hotspot["hotspot_centroid_y"],
            substrate["lv_x"], substrate["lv_y"]
        ),
        "centroid_to_vuln_mm": euclidean_distance(
            hotspot["hotspot_centroid_x"], hotspot["hotspot_centroid_y"],
            substrate["vuln_x"], substrate["vuln_y"]
        ),

        "peak_to_block_mm": euclidean_distance(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["block_x"], substrate["block_y"]
        ),
        "peak_to_lv_mm": euclidean_distance(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["lv_x"], substrate["lv_y"]
        ),
        "peak_to_vuln_mm": euclidean_distance(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["vuln_x"], substrate["vuln_y"]
        ),

        "peak_inside_block": point_in_circle(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["block_x"], substrate["block_y"], substrate["block_radius"]
        ),
        "peak_inside_lv": point_in_circle(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["lv_x"], substrate["lv_y"], substrate["lv_radius"]
        ),
        "peak_inside_vuln": point_in_circle(
            hotspot["peak_x"], hotspot["peak_y"],
            substrate["vuln_x"], substrate["vuln_y"], substrate["vuln_radius"]
        ),

        "iou_hotspot_block": mask_iou(hotspot["hotspot_mask"], truth_masks["block_mask"]),
        "iou_hotspot_lv": mask_iou(hotspot["hotspot_mask"], truth_masks["lv_mask"]),
        "iou_hotspot_vuln": mask_iou(hotspot["hotspot_mask"], truth_masks["vuln_mask"]),
        "iou_hotspot_union": mask_iou(hotspot["hotspot_mask"], truth_masks["union_mask"]),

        "mean_score_in_block": weighted_overlap_score(masked_score, truth_masks["block_mask"]),
        "mean_score_in_lv": weighted_overlap_score(masked_score, truth_masks["lv_mask"]),
        "mean_score_in_vuln": weighted_overlap_score(masked_score, truth_masks["vuln_mask"]),
        "mean_score_in_union": weighted_overlap_score(masked_score, truth_masks["union_mask"]),
    }

    return metrics, truth_masks


# =========================================================
# 14) Full pipeline
# =========================================================

def run_full_multilayer_pipeline(config):
    substrate_rng = np.random.default_rng(config.substrate_seed)
    traj_rng = np.random.default_rng(config.trajectory_seed)
    spont_rng = np.random.default_rng(config.feature_seed_spont)
    paced_rng = np.random.default_rng(config.feature_seed_paced)

    substrate = generate_hidden_substrate(config, substrate_rng)
    placement_df = generate_trajectory(config, traj_rng)
    local_patch_df = build_local_patch_geometry(config)

    spont_patch_rows = []
    paced_patch_rows = []

    for _, placement in placement_df.iterrows():
        base_patch = place_patch_on_global_domain(
            local_patch_df=local_patch_df,
            center_x=placement["center_x"],
            center_y=placement["center_y"],
            angle_deg=placement["angle_deg"],
        )

        sp_patch = compute_spontaneous_raw_features(base_patch, substrate, config, spont_rng)
        sp_patch = compute_local_derived_features(sp_patch, config)
        sp_desc = compute_spontaneous_patch_descriptors(sp_patch, config)
        sp_desc.update({
            "placement_id": int(placement["placement_id"]),
            "time_index": int(placement["time_index"]),
            "time_min": float(placement["time_min"]),
            "center_x": float(placement["center_x"]),
            "center_y": float(placement["center_y"]),
            "angle_deg": float(placement["angle_deg"]),
        })
        spont_patch_rows.append(sp_desc)

        pc_patch = compute_paced_raw_features(base_patch, substrate, config, paced_rng)
        pc_patch = compute_local_derived_features(pc_patch, config)
        pc_desc = compute_paced_patch_descriptors(pc_patch, config)
        pc_desc.update({
            "placement_id": int(placement["placement_id"]),
            "time_index": int(placement["time_index"]),
            "time_min": float(placement["time_min"]),
            "center_x": float(placement["center_x"]),
            "center_y": float(placement["center_y"]),
            "angle_deg": float(placement["angle_deg"]),
        })
        paced_patch_rows.append(pc_desc)

    spont_patch_df = pd.DataFrame(spont_patch_rows)
    paced_patch_df = pd.DataFrame(paced_patch_rows)

    spont_patch_df = normalize_and_score_spontaneous_patch_table(spont_patch_df, config)
    paced_patch_df = normalize_and_score_paced_patch_table(paced_patch_df, config)

    multilayer_node_df, graph_parts = build_multilayer_node_table(spont_patch_df, paced_patch_df, config)

    xs, ys, _, _, _, sp_masked, _ = compute_global_map_from_patch_scores(
        spont_patch_df, "SpontScore", config
    )
    _, _, _, _, _, pc_masked, _ = compute_global_map_from_patch_scores(
        paced_patch_df, "PacedScore", config
    )
    _, _, _, _, _, core_masked, _ = compute_global_map_from_patch_scores(
        multilayer_node_df.rename(columns={"CoreScore": "NodeScore"}), "NodeScore", config
    )
    _, _, _, _, _, dynamic_masked, _ = compute_global_map_from_patch_scores(
        multilayer_node_df.rename(columns={"DynamicScore": "NodeScore"}), "NodeScore", config
    )

    sp_hotspot = analyze_hotspot(sp_masked, xs, ys, config.hotspot_quantile)
    pc_hotspot = analyze_hotspot(pc_masked, xs, ys, config.hotspot_quantile)
    core_hotspot = analyze_hotspot(core_masked, xs, ys, config.hotspot_quantile)
    dyn_hotspot = analyze_hotspot(dynamic_masked, xs, ys, config.hotspot_quantile)

    sp_metrics, truth_masks = validate_map_against_truth("spontaneous", sp_masked, sp_hotspot, xs, ys, substrate)
    pc_metrics, _ = validate_map_against_truth("paced", pc_masked, pc_hotspot, xs, ys, substrate)
    core_metrics, _ = validate_map_against_truth("multilayer_core", core_masked, core_hotspot, xs, ys, substrate)
    dyn_metrics, _ = validate_map_against_truth("multilayer_dynamic", dynamic_masked, dyn_hotspot, xs, ys, substrate)

    graph_summary = {
        "mean_spont_score": float(multilayer_node_df["SpontScore"].mean()),
        "mean_paced_score": float(multilayer_node_df["PacedScore"].mean()),
        "mean_concord_raw": float(multilayer_node_df["ConcordRaw"].mean()),
        "mean_discord_raw": float(multilayer_node_df["DiscordRaw"].mean()),
        "mean_core_score": float(multilayer_node_df["CoreScore"].mean()),
        "mean_dynamic_score": float(multilayer_node_df["DynamicScore"].mean()),
        "core_vs_dyn_centroid_distance_mm": float(euclidean_distance(
            core_hotspot["hotspot_centroid_x"], core_hotspot["hotspot_centroid_y"],
            dyn_hotspot["hotspot_centroid_x"], dyn_hotspot["hotspot_centroid_y"]
        )),
        "core_vs_sp_iou": mask_iou(core_hotspot["hotspot_mask"], sp_hotspot["hotspot_mask"]),
        "core_vs_pc_iou": mask_iou(core_hotspot["hotspot_mask"], pc_hotspot["hotspot_mask"]),
        "dyn_vs_sp_iou": mask_iou(dyn_hotspot["hotspot_mask"], sp_hotspot["hotspot_mask"]),
        "dyn_vs_pc_iou": mask_iou(dyn_hotspot["hotspot_mask"], pc_hotspot["hotspot_mask"]),
    }

    return {
        "config": config,
        "substrate": substrate,
        "placement_df": placement_df,
        "spont_patch_df": spont_patch_df,
        "paced_patch_df": paced_patch_df,
        "multilayer_node_df": multilayer_node_df,
        "graph_parts": graph_parts,
        "xs": xs,
        "ys": ys,
        "sp_masked": sp_masked,
        "pc_masked": pc_masked,
        "core_masked": core_masked,
        "dynamic_masked": dynamic_masked,
        "sp_hotspot": sp_hotspot,
        "pc_hotspot": pc_hotspot,
        "core_hotspot": core_hotspot,
        "dyn_hotspot": dyn_hotspot,
        "truth_masks": truth_masks,
        "sp_metrics": sp_metrics,
        "pc_metrics": pc_metrics,
        "core_metrics": core_metrics,
        "dyn_metrics": dyn_metrics,
        "graph_summary": graph_summary,
    }


# =========================================================
# 15) Plotting
# =========================================================

def add_substrate_overlay(ax, substrate):
    block_circle = Circle(
        (substrate["block_x"], substrate["block_y"]),
        substrate["block_radius"],
        fill=False, linestyle="--", linewidth=2
    )
    lv_circle = Circle(
        (substrate["lv_x"], substrate["lv_y"]),
        substrate["lv_radius"],
        fill=False, linestyle=":", linewidth=2
    )
    vuln_circle = Circle(
        (substrate["vuln_x"], substrate["vuln_y"]),
        substrate["vuln_radius"],
        fill=False, linestyle="-.", linewidth=2
    )
    ax.add_patch(block_circle)
    ax.add_patch(lv_circle)
    ax.add_patch(vuln_circle)

    ax.scatter(substrate["source1_x"], substrate["source1_y"], marker="*", s=120, label="source1")
    ax.scatter(substrate["source2_x"], substrate["source2_y"], marker="*", s=120, label="source2")


def save_multilayer_figures(results, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = results["config"]
    substrate = results["substrate"]
    placement_df = results["placement_df"]

    extent = [0, config.domain_width, 0, config.domain_height]

    # Figure 1: four main maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    maps = [
        ("Spontaneous map", results["sp_masked"]),
        ("Paced map", results["pc_masked"]),
        ("Multilayer core map", results["core_masked"]),
        ("Multilayer dynamic map", results["dynamic_masked"]),
    ]

    for ax, (title, arr) in zip(axes.flatten(), maps):
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto")
        ax.scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x", label="placements")
        ax.scatter(config.pacing_x, config.pacing_y, s=70, marker="^", label="pacing site")
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out / "multilayer_maps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: overlays
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes2[0].imshow(results["core_masked"], origin="lower", extent=extent, aspect="auto")
    axes2[0].scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x", label="placements")
    axes2[0].scatter(results["core_hotspot"]["hotspot_centroid_x"], results["core_hotspot"]["hotspot_centroid_y"],
                     s=90, marker="o", facecolors="none", edgecolors="red", linewidths=2, label="core centroid")
    axes2[0].scatter(results["core_hotspot"]["peak_x"], results["core_hotspot"]["peak_y"],
                     s=90, marker="*", label="core peak")
    add_substrate_overlay(axes2[0], substrate)
    axes2[0].set_title("Multilayer core vs substrate")
    axes2[0].set_xlabel("x (mm)")
    axes2[0].set_ylabel("y (mm)")
    axes2[0].legend(loc="upper right", fontsize=7)
    fig2.colorbar(im0, ax=axes2[0], fraction=0.046, pad=0.04)

    im1 = axes2[1].imshow(results["dynamic_masked"], origin="lower", extent=extent, aspect="auto")
    axes2[1].scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x", label="placements")
    axes2[1].scatter(results["dyn_hotspot"]["hotspot_centroid_x"], results["dyn_hotspot"]["hotspot_centroid_y"],
                     s=90, marker="o", facecolors="none", edgecolors="red", linewidths=2, label="dynamic centroid")
    axes2[1].scatter(results["dyn_hotspot"]["peak_x"], results["dyn_hotspot"]["peak_y"],
                     s=90, marker="*", label="dynamic peak")
    add_substrate_overlay(axes2[1], substrate)
    axes2[1].set_title("Multilayer dynamic vs substrate")
    axes2[1].set_xlabel("x (mm)")
    axes2[1].set_ylabel("y (mm)")
    axes2[1].legend(loc="upper right", fontsize=7)
    fig2.colorbar(im1, ax=axes2[1], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig(out / "multilayer_overlays.png", dpi=220, bbox_inches="tight")
    plt.close(fig2)


# =========================================================
# 16) Main
# =========================================================

if __name__ == "__main__":
    config = DatasetConfig()
    output_dir = Path("manual_check_multilayer_graph")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_full_multilayer_pipeline(config)

    pd.DataFrame([results["sp_metrics"]]).to_csv(output_dir / "spontaneous_validation_metrics.csv", index=False)
    pd.DataFrame([results["pc_metrics"]]).to_csv(output_dir / "paced_validation_metrics.csv", index=False)
    pd.DataFrame([results["core_metrics"]]).to_csv(output_dir / "multilayer_core_metrics.csv", index=False)
    pd.DataFrame([results["dyn_metrics"]]).to_csv(output_dir / "multilayer_dynamic_metrics.csv", index=False)
    pd.DataFrame([results["graph_summary"]]).to_csv(output_dir / "multilayer_graph_summary.csv", index=False)

    results["multilayer_node_df"].to_csv(output_dir / "multilayer_node_table.csv", index=False)
    results["spont_patch_df"].to_csv(output_dir / "spontaneous_patch_summary.csv", index=False)
    results["paced_patch_df"].to_csv(output_dir / "paced_patch_summary.csv", index=False)

    pd.DataFrame(results["sp_masked"], index=results["ys"], columns=results["xs"]).to_csv(
        output_dir / "spontaneous_masked_map.csv"
    )
    pd.DataFrame(results["pc_masked"], index=results["ys"], columns=results["xs"]).to_csv(
        output_dir / "paced_masked_map.csv"
    )
    pd.DataFrame(results["core_masked"], index=results["ys"], columns=results["xs"]).to_csv(
        output_dir / "multilayer_core_map.csv"
    )
    pd.DataFrame(results["dynamic_masked"], index=results["ys"], columns=results["xs"]).to_csv(
        output_dir / "multilayer_dynamic_map.csv"
    )

    save_multilayer_figures(results, output_dir)

    print("Multilayer-graph files saved to: manual_check_multilayer_graph/")

    print("\nMultilayer graph summary:")
    print(pd.DataFrame([results["graph_summary"]]).T)

    print("\nMultilayer core metrics:")
    print(pd.DataFrame([results["core_metrics"]]).T)

    print("\nMultilayer dynamic metrics:")
    print(pd.DataFrame([results["dyn_metrics"]]).T)