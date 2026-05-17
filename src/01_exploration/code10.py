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
    domain_width: float = 100.0
    domain_height: float = 80.0

    patch_rows: int = 4
    patch_cols: int = 4
    electrode_spacing: float = 4.0

    n_placements: int = 10
    minutes_per_step: float = 25.0

    placement_jitter: float = 2.0
    angle_jitter_deg: float = 20.0

    n_sources: int = 2
    base_cv: float = 0.7
    min_cv: float = 0.20

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

    disp_c1: float = 0.5
    disp_c2: float = 0.5

    sigma_kernel: float = 6.0
    low_voltage_threshold: float = 0.50
    grad_eps: float = 1e-6
    global_grid_step: float = 2.0
    confidence_threshold_ratio: float = 0.15
    hotspot_quantile: float = 0.90

    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2

    substrate_seed: int = 42
    feature_seed: int = 999


# =========================================================
# 2) Utility
# =========================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def rotation_matrix(angle_deg):
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def radial_bump(x, y, cx, cy, radius, strength):
    sigma = max(radius / 2.0, 1e-6)
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return strength * np.exp(-d2 / (2.0 * sigma ** 2))


def minmax_normalize(series: pd.Series, eps: float = 1e-8):
    return (series - series.min()) / (series.max() - series.min() + eps)


def gaussian_kernel_3x3():
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float)
    return k / k.sum()


def smooth_grid_3x3(grid):
    kernel = gaussian_kernel_3x3()
    padded = np.pad(grid, pad_width=1, mode="edge")
    out = np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r+3, c:c+3]
            out[r, c] = np.sum(window * kernel)
    return out


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(inter / union)


def weighted_overlap_score(masked_score, truth_mask):
    vals = np.where(truth_mask, masked_score, np.nan)
    if np.all(np.isnan(vals)):
        return np.nan
    return float(np.nanmean(vals))


# =========================================================
# 3) Hidden substrate
# =========================================================

def generate_hidden_substrate(config: DatasetConfig, rng):
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

def build_local_patch_geometry(config):
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


def place_patch_on_global_domain(local_patch_df, center_x, center_y, angle_deg):
    df = local_patch_df.copy()
    R = rotation_matrix(angle_deg)
    local_xy = df[["x_local", "y_local"]].to_numpy()
    rotated_xy = local_xy @ R.T
    df["x_global"] = rotated_xy[:, 0] + center_x
    df["y_global"] = rotated_xy[:, 1] + center_y
    return df


def generate_trajectory(config, rng):
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
# 5) Feature generation
# =========================================================

def compute_raw_electrode_features(patch_df, substrate, config, rng):
    df = patch_df.copy()
    x = df["x_global"].to_numpy()
    y = df["y_global"].to_numpy()

    d1 = euclidean_distance(x, y, substrate["source1_x"], substrate["source1_y"])
    d2 = euclidean_distance(x, y, substrate["source2_x"], substrate["source2_y"])

    S1 = np.exp(-d1 / config.sigma_I)
    S2 = np.exp(-d2 / config.sigma_I)
    interaction = S1 * S2

    block_field = radial_bump(x, y, substrate["block_x"], substrate["block_y"],
                              substrate["block_radius"], substrate["block_strength"])
    lv_field = radial_bump(x, y, substrate["lv_x"], substrate["lv_y"],
                           substrate["lv_radius"], substrate["lv_strength"])
    vuln_field = radial_bump(x, y, substrate["vuln_x"], substrate["vuln_y"],
                             substrate["vuln_radius"], substrate["vuln_strength"])

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


def compute_patch_gradients(lat_grid, spacing):
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
# 6) Patch-level scoring
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


def compute_patch_descriptors_for_one_patch(patch_df, config):
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


def normalize_and_score_patch_table(patch_df, config):
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
# 7) Global map
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


def compute_global_maps(patch_df, config, eps=1e-8):
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
# 8) Hotspot extraction
# =========================================================

def analyze_hotspot(masked_score, xs, ys, quantile):
    valid_vals = masked_score[~np.isnan(masked_score)]
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


# =========================================================
# 9) Truth masks
# =========================================================

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


def point_in_circle(px, py, cx, cy, radius):
    return (px - cx) ** 2 + (py - cy) ** 2 <= radius ** 2


# =========================================================
# 10) One complete run
# =========================================================

def run_baseline_and_validate(config, substrate, trajectory_seed):
    traj_rng = np.random.default_rng(trajectory_seed)
    feature_rng = np.random.default_rng(config.feature_seed)

    placement_df = generate_trajectory(config, traj_rng)
    local_patch_df = build_local_patch_geometry(config)

    patch_rows = []
    for _, placement in placement_df.iterrows():
        patch_df = place_patch_on_global_domain(
            local_patch_df,
            placement["center_x"],
            placement["center_y"],
            placement["angle_deg"],
        )
        patch_df = compute_raw_electrode_features(patch_df, substrate, config, feature_rng)
        patch_df = compute_local_derived_features(patch_df, config)

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

    xs, ys, masked_score, confidence, conf_thr = compute_global_maps(patch_summary_df, config)
    hotspot = analyze_hotspot(masked_score, xs, ys, config.hotspot_quantile)
    truth_masks = build_truth_masks(xs, ys, substrate)

    metrics = {
        "trajectory_seed": trajectory_seed,
        "peak_score": hotspot["peak_score"],
        "hotspot_area_pixels": hotspot["hotspot_area_pixels"],
        "conf_threshold": conf_thr,

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

    return placement_df, patch_summary_df, xs, ys, masked_score, hotspot, truth_masks, metrics


# =========================================================
# 11) Visualization
# =========================================================

def plot_validation_figure(config, substrate, placement_df, xs, ys, masked_score, hotspot, output_path):
    extent = [0, config.domain_width, 0, config.domain_height]
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(masked_score, origin="lower", extent=extent, aspect="auto")
    ax.scatter(placement_df["center_x"], placement_df["center_y"], s=25, marker="x", label="placements")

    ax.scatter(hotspot["hotspot_centroid_x"], hotspot["hotspot_centroid_y"],
               s=100, marker="o", facecolors="none", edgecolors="red",
               linewidths=2, label="hotspot centroid")
    ax.scatter(hotspot["peak_x"], hotspot["peak_y"],
               s=100, marker="*", label="peak")

    block_circle = Circle((substrate["block_x"], substrate["block_y"]),
                          substrate["block_radius"], fill=False, linestyle="--", linewidth=2)
    lv_circle = Circle((substrate["lv_x"], substrate["lv_y"]),
                       substrate["lv_radius"], fill=False, linestyle=":", linewidth=2)
    vuln_circle = Circle((substrate["vuln_x"], substrate["vuln_y"]),
                         substrate["vuln_radius"], fill=False, linestyle="-.", linewidth=2)

    ax.add_patch(block_circle)
    ax.add_patch(lv_circle)
    ax.add_patch(vuln_circle)

    ax.scatter(substrate["source1_x"], substrate["source1_y"], marker="*", s=120, label="source1")
    ax.scatter(substrate["source2_x"], substrate["source2_y"], marker="*", s=120, label="source2")

    ax.set_title("Hotspot vs Hidden Substrate")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 12) Main
# =========================================================

if __name__ == "__main__":
    config = DatasetConfig()
    output_dir = Path("manual_check_hidden_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    substrate_rng = np.random.default_rng(config.substrate_seed)
    substrate = generate_hidden_substrate(config, substrate_rng)

    # baseline trajectory seed
    trajectory_seed = 101

    placement_df, patch_df, xs, ys, masked_score, hotspot, truth_masks, metrics = run_baseline_and_validate(
        config=config,
        substrate=substrate,
        trajectory_seed=trajectory_seed
    )

    pd.DataFrame([metrics]).to_csv(output_dir / "hidden_validation_metrics.csv", index=False)
    patch_df.to_csv(output_dir / "patch_summary_table.csv", index=False)
    pd.DataFrame(masked_score, index=ys, columns=xs).to_csv(output_dir / "masked_suspicion_map.csv")

    plot_validation_figure(
        config=config,
        substrate=substrate,
        placement_df=placement_df,
        xs=xs,
        ys=ys,
        masked_score=masked_score,
        hotspot=hotspot,
        output_path=output_dir / "hidden_validation_overlay.png"
    )

    print("Hidden-validation files saved to: manual_check_hidden_validation/")
    print("\nHidden validation metrics:")
    print(pd.DataFrame([metrics]).T)