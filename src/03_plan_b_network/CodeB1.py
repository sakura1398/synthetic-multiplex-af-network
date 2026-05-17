# codeB1.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dataclasses import dataclass
from pathlib import Path


print("RUNNING CODEB1: synthetic multiplex network construction")


# =========================================================
# 1) Configuration
# =========================================================
# This dataclass stores all main parameters in one place so that:
# 1) the experiment is reproducible,
# 2) parameter tuning is easier,
# 3) reporting is clearer.

@dataclass
class B1Config:
    # -------------------------
    # Global domain geometry
    # -------------------------
    domain_width: float = 100.0
    domain_height: float = 80.0

    # -------------------------
    # Local patch geometry
    # -------------------------
    patch_rows: int = 4
    patch_cols: int = 4
    electrode_spacing: float = 4.0

    # -------------------------
    # Sequential mapping settings
    # -------------------------
    # We increase the number of placements to make the graph analysis
    # more meaningful for network science metrics.
    n_placements: int = 36
    minutes_per_step: float = 25.0
    placement_jitter: float = 1.5
    angle_jitter_deg: float = 15.0

    # -------------------------
    # Hidden substrate settings
    # -------------------------
    n_sources: int = 2
    base_cv: float = 0.7
    min_cv: float = 0.20

    sigma_A: float = 18.0
    sigma_I: float = 15.0

    lat_noise_std: float = 1.5
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

    low_voltage_threshold: float = 0.50
    grad_eps: float = 1e-6

    # -------------------------
    # Spontaneous patch scoring
    # -------------------------
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    # DIC coefficients
    alpha_dic: float = 0.4
    beta_dic: float = 0.4
    gamma_dic: float = 0.2

    # -------------------------
    # Paced patch scoring
    # -------------------------
    wp1: float = 0.40
    wp2: float = 0.30
    wp3: float = 0.20
    wp4: float = 0.10

    # Paced branch settings
    pacing_x: float = 6.0
    pacing_y: float = 40.0
    paced_lat_noise_std: float = 0.5
    paced_block_penalty: float = 0.85
    paced_lv_penalty: float = 0.10
    paced_vuln_penalty: float = 0.20
    ldh_threshold_ms: float = 12.0

    # -------------------------
    # Graph construction settings
    # -------------------------
    spatial_sigma_mm: float = 18.0
    feature_sigma: float = 2.0

    # These weights combine spatial, temporal, and feature similarity
    alpha_graph: float = 0.50   # spatial contribution
    beta_graph: float = 0.20    # temporal contribution
    gamma_graph: float = 0.30   # feature contribution

    # k-nearest style sparsification
    k_spatial: int = 4
    k_feature: int = 2

    # -------------------------
    # Random seeds
    # -------------------------
    substrate_seed: int = 42
    trajectory_seed: int = 101
    feature_seed_spont: int = 999
    feature_seed_paced: int = 111


# =========================================================
# 2) Utility functions
# =========================================================

def clamp(value, low, high):
    """Clamp a scalar value between low and high."""
    return max(low, min(high, value))


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """Return a 2D rotation matrix for a given angle in degrees."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two 2D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def radial_bump(x, y, cx, cy, radius, strength):
    """
    Create a smooth radial field centered at (cx, cy).
    This is used for hidden substrate components such as block,
    low-voltage region, and vulnerability region.
    """
    sigma = max(radius / 2.0, 1e-6)
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return strength * np.exp(-d2 / (2.0 * sigma ** 2))


def minmax_normalize_series(series: pd.Series, eps: float = 1e-8) -> pd.Series:
    """Apply min-max normalization to a pandas Series."""
    return (series - series.min()) / (series.max() - series.min() + eps)


def gaussian_kernel_3x3():
    """Return a simple normalized 3x3 Gaussian-like smoothing kernel."""
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float)
    return k / k.sum()


def smooth_grid_3x3(grid: np.ndarray) -> np.ndarray:
    """
    Smooth a 2D grid with a small Gaussian-like 3x3 kernel.
    This is used before computing gradient directions for ICS.
    """
    kernel = gaussian_kernel_3x3()
    padded = np.pad(grid, pad_width=1, mode="edge")
    out = np.zeros_like(grid, dtype=float)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r:r+3, c:c+3]
            out[r, c] = np.sum(window * kernel)
    return out


def row_standardize(df: pd.DataFrame, cols):
    """
    Standardize selected columns (z-score style) so feature similarity
    is not dominated by scale differences.
    """
    out = df[cols].copy().astype(float)
    for col in cols:
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        if sd < 1e-8:
            sd = 1.0
        out[col] = (out[col] - mu) / sd
    return out


def pairwise_squared_dist(X: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between rows of X."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff ** 2, axis=2)


def keep_top_k_per_row(W: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only the top-k non-diagonal entries in each row.
    This is used to sparsify similarity matrices.
    """
    A = np.zeros_like(W, dtype=float)
    n = W.shape[0]
    for i in range(n):
        row = W[i].copy()
        row[i] = -np.inf
        idx = np.argsort(row)[::-1][:k]
        idx = [j for j in idx if np.isfinite(row[j]) and row[j] > 0]
        A[i, idx] = W[i, idx]
    return A


def symmetrize_matrix(W: np.ndarray, mode: str = "max") -> np.ndarray:
    """
    Symmetrize a weighted matrix.
    mode='max' keeps the larger of W_ij and W_ji.
    mode='mean' averages them.
    """
    if mode == "max":
        return np.maximum(W, W.T)
    if mode == "mean":
        return 0.5 * (W + W.T)
    raise ValueError("mode must be 'max' or 'mean'")


def adjacency_to_edge_table(W: np.ndarray, node_ids):
    """
    Convert an adjacency matrix into an edge list table.
    Only upper-triangular nonzero edges are stored.
    """
    rows = []
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                rows.append({
                    "node_i": int(node_ids[i]),
                    "node_j": int(node_ids[j]),
                    "weight": float(W[i, j]),
                })
    return pd.DataFrame(rows)


# =========================================================
# 3) Hidden substrate generation
# =========================================================

def generate_hidden_substrate(config: B1Config, rng: np.random.Generator) -> dict:
    """
    Generate a synthetic hidden substrate with:
    - two source regions
    - one block region
    - one low-voltage region
    - one vulnerability region
    """
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

def build_local_patch_geometry(config: B1Config) -> pd.DataFrame:
    """
    Build a local 4x4 electrode grid centered around zero.
    The local geometry is later rotated and translated to the global domain.
    """
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
    """
    Rotate and translate the local patch into the global domain.
    """
    df = local_patch_df.copy()
    R = rotation_matrix(angle_deg)
    local_xy = df[["x_local", "y_local"]].to_numpy()
    rotated_xy = local_xy @ R.T
    df["x_global"] = rotated_xy[:, 0] + center_x
    df["y_global"] = rotated_xy[:, 1] + center_y
    return df


def generate_trajectory(config: B1Config, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate a sequential serpentine trajectory over the global domain.
    This simulates how a catheter patch moves from placement to placement.
    """
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
# 5) Substrate fields
# =========================================================

def compute_substrate_fields(x, y, substrate, config: B1Config):
    """
    Compute the hidden substrate fields at positions (x, y):
    - source influence fields
    - interaction field
    - block field
    - low-voltage field
    - vulnerability field
    """
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
# 6) Raw feature generation
# =========================================================

def compute_spontaneous_raw_features(patch_df, substrate, config, rng):
    """
    Generate spontaneous branch raw features at electrode level.
    These features are synthetic but grounded in the hidden substrate fields.
    """
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


def compute_paced_raw_features(patch_df, substrate, config, rng):
    """
    Generate paced branch raw features at electrode level.
    Pacing starts from a fixed pacing site and propagation is slowed
    by block, low-voltage burden, and vulnerability.
    """
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
    return df


# =========================================================
# 7) Derived local features
# =========================================================

def compute_patch_gradients(lat_grid: np.ndarray, spacing: float):
    """
    Compute spatial gradients of LAT over a 2D patch.
    """
    grad_y, grad_x = np.gradient(lat_grid, spacing, spacing)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_angle = np.arctan2(grad_y, grad_x)
    return grad_x, grad_y, grad_mag, grad_angle


def compute_dispersion_for_patch(fi_grid, a_grid, config):
    """
    Compute a local dispersion measure for each electrode
    based on neighbor variability in FI and amplitude.
    """
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
    """
    Add derived electrode-level local features:
    - LAT gradients
    - LAT gradient magnitude and angle
    - dispersion
    """
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
# 8) Patch-level descriptors and scores
# =========================================================

def compute_std_for_patch(lat_grid, spacing):
    """
    Compute a simple LAT heterogeneity descriptor based on differences
    between neighboring electrodes.
    """
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
    """
    Compute a large-delay heterogeneity (LDH) burden:
    fraction of neighboring LAT differences above a threshold.
    """
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
    """
    Compute an incoherence-like score based on disagreement
    between local gradient directions.
    """
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
    """
    Compute spontaneous patch-level descriptors from the 4x4 electrode grid.
    """
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
    """
    Compute paced patch-level descriptors from the 4x4 electrode grid.
    """
    df = patch_df.sort_values(["row", "col"]).reset_index(drop=True)
    rows = config.patch_rows
    cols = config.patch_cols
    spacing = config.electrode_spacing
    lat_grid = df["LAT"].to_numpy().reshape(rows, cols)

    return {
        "STD_paced": compute_std_for_patch(lat_grid, spacing),
        "ICS_paced": compute_ics_for_patch(lat_grid, spacing, config),
        "LDH_paced": compute_ldh_for_patch(lat_grid, config.ldh_threshold_ms),
        "LV_burden_paced": float((df["A"] < config.low_voltage_threshold).mean()),
    }


def normalize_and_score_spontaneous_patch_table(patch_df, config):
    """
    Normalize spontaneous patch descriptors and compute the final
    spontaneous patch score.
    """
    out = patch_df.copy()

    out["STD_norm"] = minmax_normalize_series(out["STD_p"])
    out["ICS_norm"] = minmax_normalize_series(out["ICS_p"])
    out["Disp_norm"] = minmax_normalize_series(out["Disp_p"])
    out["FI_norm"] = minmax_normalize_series(out["FI_burden"])
    out["LV_norm"] = minmax_normalize_series(out["LV_burden"])

    out["DIC_p"] = (
        config.alpha_dic * out["Disp_norm"]
        + config.beta_dic * out["ICS_norm"]
        + config.gamma_dic * (out["Disp_norm"] * out["ICS_norm"])
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
    """
    Normalize paced patch descriptors and compute the final paced patch score.
    """
    out = patch_df.copy()

    out["STD_norm_pc"] = minmax_normalize_series(out["STD_paced"])
    out["ICS_norm_pc"] = minmax_normalize_series(out["ICS_paced"])
    out["LDH_norm_pc"] = minmax_normalize_series(out["LDH_paced"])
    out["LV_norm_pc"] = minmax_normalize_series(out["LV_burden_paced"])

    out["PacedScore"] = (
        config.wp1 * out["STD_norm_pc"]
        + config.wp2 * out["LDH_norm_pc"]
        + config.wp3 * out["LV_norm_pc"]
        + config.wp4 * out["ICS_norm_pc"]
    )
    return out


# =========================================================
# 9) Build synthetic patch dataset for Plan B
# =========================================================

def build_patch_dataset(config: B1Config):
    """
    Generate the synthetic patch-level dataset used for multiplex network construction.
    This function returns:
    - node_table: merged spontaneous + paced patch attributes
    - placement_df: placement geometry and timing
    - substrate_table: hidden substrate parameters for documentation
    """
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

        # Build spontaneous patch
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
            "mean_block_field": float(sp_patch["block_field"].mean()),
            "mean_lv_field": float(sp_patch["lv_field"].mean()),
            "mean_vuln_field": float(sp_patch["vuln_field"].mean()),
            "mean_interaction": float(sp_patch["interaction"].mean()),
            "mean_LAT_sp": float(sp_patch["LAT"].mean()),
            "mean_A_sp": float(sp_patch["A"].mean()),
            "mean_FI_sp": float(sp_patch["FI"].mean()),
        })
        spont_patch_rows.append(sp_desc)

        # Build paced patch
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
            "mean_LAT_pc": float(pc_patch["LAT"].mean()),
            "mean_A_pc": float(pc_patch["A"].mean()),
            "mean_FI_pc": float(pc_patch["FI"].mean()),
        })
        paced_patch_rows.append(pc_desc)

    spont_patch_df = pd.DataFrame(spont_patch_rows)
    paced_patch_df = pd.DataFrame(paced_patch_rows)

    spont_patch_df = normalize_and_score_spontaneous_patch_table(spont_patch_df, config)
    paced_patch_df = normalize_and_score_paced_patch_table(paced_patch_df, config)

    # Merge the two branches into one node table
    node_table = spont_patch_df.merge(
        paced_patch_df[[
            "placement_id", "STD_paced", "ICS_paced", "LDH_paced", "LV_burden_paced",
            "mean_LAT_pc", "mean_A_pc", "mean_FI_pc", "PacedScore"
        ]],
        on="placement_id",
        how="inner"
    )

    # Also keep normalized scores for later role analysis
    node_table["u_sp"] = minmax_normalize_series(node_table["SpontScore"])
    node_table["u_pc"] = minmax_normalize_series(node_table["PacedScore"])

    substrate_table = pd.DataFrame([substrate])

    return node_table, placement_df, substrate_table


# =========================================================
# 10) Similarity matrices
# =========================================================

def compute_spatial_similarity(node_table: pd.DataFrame, sigma_mm: float):
    """
    Build a dense spatial similarity matrix based on patch center distance.
    """
    XY = node_table[["center_x", "center_y"]].to_numpy()
    d2 = pairwise_squared_dist(XY)
    W = np.exp(-d2 / (2.0 * sigma_mm ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def compute_temporal_adjacency(node_table: pd.DataFrame):
    """
    Build a simple temporal adjacency matrix where only consecutive
    placements are connected.
    """
    t = node_table["time_index"].to_numpy()
    n = len(t)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(int(t[i]) - int(t[j])) == 1:
                W[i, j] = 1.0
    return W


def compute_feature_similarity(node_table: pd.DataFrame, cols, sigma_f: float):
    """
    Build a dense feature similarity matrix from standardized node attributes.
    """
    Z = row_standardize(node_table, cols).to_numpy()
    d2 = pairwise_squared_dist(Z)
    W = np.exp(-d2 / (2.0 * sigma_f ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def build_sparse_layer_adjacency(spatial_W, temporal_W, feature_W, config: B1Config):
    """
    Build a sparse weighted layer adjacency matrix by:
    1) sparsifying spatial similarity
    2) sparsifying feature similarity
    3) keeping simple temporal adjacency
    4) combining them with alpha, beta, gamma weights
    """
    spatial_sparse = keep_top_k_per_row(spatial_W, config.k_spatial)
    feature_sparse = keep_top_k_per_row(feature_W, config.k_feature)

    spatial_sparse = symmetrize_matrix(spatial_sparse, mode="max")
    feature_sparse = symmetrize_matrix(feature_sparse, mode="max")
    temporal_sparse = symmetrize_matrix(temporal_W, mode="max")

    A = (
        config.alpha_graph * spatial_sparse
        + config.beta_graph * temporal_sparse
        + config.gamma_graph * feature_sparse
    )

    A = symmetrize_matrix(A, mode="max")
    np.fill_diagonal(A, 0.0)
    return A, spatial_sparse, temporal_sparse, feature_sparse


def build_supra_adjacency(A_sp, A_pc, omega):
    """
    Build the multiplex supra-adjacency matrix:
        [ A_sp   omega I ]
        [ omega I  A_pc ]
    """
    n = A_sp.shape[0]
    supra = np.zeros((2 * n, 2 * n), dtype=float)
    supra[:n, :n] = A_sp
    supra[n:, n:] = A_pc
    supra[:n, n:] = omega * np.eye(n)
    supra[n:, :n] = omega * np.eye(n)
    return supra


# =========================================================
# 11) Diagnostics / summaries
# =========================================================

def summarize_graph(W: np.ndarray, layer_name: str):
    """
    Compute basic graph diagnostics for a weighted adjacency matrix.
    """
    G = nx.from_numpy_array(W)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    if m > 0:
        strengths = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        mean_strength = float(np.mean(strengths))
        mean_degree = float(np.mean(degrees))
        avg_clustering = float(nx.average_clustering(G, weight="weight"))
    else:
        mean_strength = 0.0
        mean_degree = 0.0
        avg_clustering = 0.0

    n_components = nx.number_connected_components(G)

    lines = []
    lines.append(f"Layer: {layer_name}")
    lines.append(f"  nodes = {n}")
    lines.append(f"  edges = {m}")
    lines.append(f"  density = {density:.6f}")
    lines.append(f"  connected_components = {n_components}")
    lines.append(f"  mean_degree = {mean_degree:.4f}")
    lines.append(f"  mean_strength = {mean_strength:.4f}")
    lines.append(f"  avg_clustering = {avg_clustering:.6f}")
    return "\n".join(lines), G


# =========================================================
# 12) Plotting
# =========================================================

def plot_patch_layout(node_table: pd.DataFrame, out_path: Path):
    """
    Plot the patch layout over the domain, colored by spontaneous and paced scores.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(
        node_table["center_x"], node_table["center_y"],
        c=node_table["SpontScore"], s=110
    )
    for _, row in node_table.iterrows():
        ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=7)
    ax.set_title("Patch layout coloured by SpontScore")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0, node_table["center_x"].max() + 8)
    ax.set_ylim(0, node_table["center_y"].max() + 8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(
        node_table["center_x"], node_table["center_y"],
        c=node_table["PacedScore"], s=110
    )
    for _, row in node_table.iterrows():
        ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=7)
    ax.set_title("Patch layout coloured by PacedScore")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0, node_table["center_x"].max() + 8)
    ax.set_ylim(0, node_table["center_y"].max() + 8)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_network_overlay(node_table: pd.DataFrame, W: np.ndarray, title: str, score_col: str, out_path: Path):
    """
    Overlay graph edges on the 2D patch layout.
    Edge transparency is proportional to edge weight.
    """
    xy = node_table[["center_x", "center_y"]].to_numpy()
    scores = node_table[score_col].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    max_w = np.max(W) if np.max(W) > 0 else 1.0
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                alpha = 0.15 + 0.70 * (W[i, j] / max_w)
                ax.plot(
                    [xy[i, 0], xy[j, 0]],
                    [xy[i, 1], xy[j, 1]],
                    linewidth=1.0,
                    alpha=alpha
                )

    sc = ax.scatter(xy[:, 0], xy[:, 1], c=scores, s=120, zorder=3)
    for _, row in node_table.iterrows():
        ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_supra_heatmap(supra: np.ndarray, out_path: Path):
    """
    Plot the supra-adjacency matrix as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(supra, aspect="auto")
    ax.set_title("Supra-adjacency heatmap")
    ax.set_xlabel("node index")
    ax.set_ylabel("node index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_scatter_overview(node_table: pd.DataFrame, out_path: Path):
    """
    Create simple scatter plots to visually inspect the relationship
    between hidden substrate summaries and branch scores.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(node_table["mean_block_field"], node_table["SpontScore"])
    axes[0].set_xlabel("mean_block_field")
    axes[0].set_ylabel("SpontScore")
    axes[0].set_title("SpontScore vs mean block")

    axes[1].scatter(node_table["mean_vuln_field"], node_table["PacedScore"])
    axes[1].set_xlabel("mean_vuln_field")
    axes[1].set_ylabel("PacedScore")
    axes[1].set_title("PacedScore vs mean vuln")

    axes[2].scatter(node_table["u_sp"], node_table["u_pc"])
    axes[2].set_xlabel("u_sp")
    axes[2].set_ylabel("u_pc")
    axes[2].set_title("Layer score overview")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 13) Dataset card draft
# =========================================================

def build_dataset_card_text(config: B1Config, substrate_table: pd.DataFrame):
    """
    Build a short dataset card draft for the synthetic dataset.
    This is useful for the assignment write-up.
    """
    s = substrate_table.iloc[0].to_dict()

    text = f"""
Dataset name:
    Synthetic multiplex atrial mapping patch network

Motivation:
    This dataset was generated to study whether paced dynamics can act as a probe of hidden structure
    in a synthetic multiplex atrial mapping network.

How the dataset was generated:
    - Domain size: {config.domain_width} x {config.domain_height}
    - Patch geometry: {config.patch_rows}x{config.patch_cols} electrodes, spacing {config.electrode_spacing} mm
    - Number of placements: {config.n_placements}
    - Time between placements: {config.minutes_per_step} minutes
    - Two branches were generated: spontaneous and paced
    - Hidden substrate includes:
        * two source regions
        * one block region
        * one low-voltage region
        * one vulnerability region

Hidden substrate parameters for this run:
    source1 = ({s['source1_x']:.3f}, {s['source1_y']:.3f})
    source2 = ({s['source2_x']:.3f}, {s['source2_y']:.3f})
    block   = center ({s['block_x']:.3f}, {s['block_y']:.3f}), radius {s['block_radius']:.3f}, strength {s['block_strength']:.3f}
    lv      = center ({s['lv_x']:.3f}, {s['lv_y']:.3f}), radius {s['lv_radius']:.3f}, strength {s['lv_strength']:.3f}
    vuln    = center ({s['vuln_x']:.3f}, {s['vuln_y']:.3f}), radius {s['vuln_radius']:.3f}, strength {s['vuln_strength']:.3f}

Variables:
    Node-level variables include:
    - geometric variables: placement_id, time_index, time_min, center_x, center_y, angle_deg
    - spontaneous patch descriptors: STD_p, ICS_p, Disp_p, FI_burden, LV_burden, SpontScore
    - paced patch descriptors: STD_paced, ICS_paced, LDH_paced, LV_burden_paced, PacedScore
    - hidden truth summaries: mean_block_field, mean_lv_field, mean_vuln_field, mean_interaction

Intended use:
    - multiplex network construction
    - comparison of spontaneous and paced layers
    - identification of stable, condition-dependent, and bridge-like regions

Limitations:
    - fully synthetic dataset
    - 2D simplified domain
    - simplified electrophysiological rules
    - catheter contact quality and local impedance are not explicitly modeled in this version

Ethical considerations:
    - no human subject data
    - fully synthetic
"""
    return text.strip()


# =========================================================
# 14) Main
# =========================================================

if __name__ == "__main__":
    # Initialize configuration
    config = B1Config()

    # Create output folder next to the script
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "codeB1_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Generate the synthetic patch-level dataset
    # -------------------------
    node_table, placement_df, substrate_table = build_patch_dataset(config)

    # Save raw tables for transparency and later use
    node_table.to_csv(output_dir / "b1_node_table.csv", index=False)
    placement_df.to_csv(output_dir / "b1_placement_table.csv", index=False)
    substrate_table.to_csv(output_dir / "b1_hidden_substrate.csv", index=False)

    # -------------------------
    # Build similarity matrices
    # -------------------------
    spatial_W = compute_spatial_similarity(node_table, config.spatial_sigma_mm)
    temporal_W = compute_temporal_adjacency(node_table)

    spont_feature_cols = ["STD_p", "ICS_p", "Disp_p", "FI_burden", "LV_burden", "SpontScore"]
    paced_feature_cols = ["STD_paced", "ICS_paced", "LDH_paced", "LV_burden_paced", "PacedScore"]

    feat_W_sp = compute_feature_similarity(node_table, spont_feature_cols, config.feature_sigma)
    feat_W_pc = compute_feature_similarity(node_table, paced_feature_cols, config.feature_sigma)

    # Build sparse adjacency matrices for the two layers
    A_sp, spatial_sparse, temporal_sparse, feature_sparse_sp = build_sparse_layer_adjacency(
        spatial_W, temporal_W, feat_W_sp, config
    )
    A_pc, _, _, feature_sparse_pc = build_sparse_layer_adjacency(
        spatial_W, temporal_W, feat_W_pc, config
    )

    # -------------------------
    # Define inter-layer coupling
    # -------------------------
    # We use the mean nonzero intra-layer weight as an initial coupling scale.
    nonzero_sp = A_sp[A_sp > 0]
    nonzero_pc = A_pc[A_pc > 0]
    pooled = np.concatenate([nonzero_sp, nonzero_pc]) if (len(nonzero_sp) + len(nonzero_pc)) > 0 else np.array([1.0])
    omega = float(np.mean(pooled))

    # Build the full multiplex supra-adjacency matrix
    supra = build_supra_adjacency(A_sp, A_pc, omega)

    # -------------------------
    # Save adjacency matrices
    # -------------------------
    node_ids = node_table["placement_id"].to_numpy()
    pd.DataFrame(A_sp, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_spont_adjacency.csv")
    pd.DataFrame(A_pc, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_paced_adjacency.csv")

    supra_index = [f"s_{i}" for i in node_ids] + [f"p_{i}" for i in node_ids]
    pd.DataFrame(supra, index=supra_index, columns=supra_index).to_csv(output_dir / "b1_supra_adjacency.csv")

    # Save edge lists for both layers
    adjacency_to_edge_table(A_sp, node_ids).to_csv(output_dir / "b1_edge_table_spont.csv", index=False)
    adjacency_to_edge_table(A_pc, node_ids).to_csv(output_dir / "b1_edge_table_paced.csv", index=False)

    # Save adjacency components for transparency
    pd.DataFrame(spatial_sparse, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_spatial_component.csv")
    pd.DataFrame(temporal_sparse, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_temporal_component.csv")
    pd.DataFrame(feature_sparse_sp, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_feature_component_spont.csv")
    pd.DataFrame(feature_sparse_pc, index=node_ids, columns=node_ids).to_csv(output_dir / "b1_feature_component_paced.csv")

    # -------------------------
    # Compute graph summaries
    # -------------------------
    summary_sp, G_sp = summarize_graph(A_sp, "spontaneous")
    summary_pc, G_pc = summarize_graph(A_pc, "paced")

    summary_lines = []
    summary_lines.append("CODEB1 NETWORK SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"n_placements = {config.n_placements}")
    summary_lines.append(f"alpha_graph = {config.alpha_graph}")
    summary_lines.append(f"beta_graph  = {config.beta_graph}")
    summary_lines.append(f"gamma_graph = {config.gamma_graph}")
    summary_lines.append(f"k_spatial   = {config.k_spatial}")
    summary_lines.append(f"k_feature   = {config.k_feature}")
    summary_lines.append(f"omega       = {omega:.6f}")
    summary_lines.append("")
    summary_lines.append(summary_sp)
    summary_lines.append("")
    summary_lines.append(summary_pc)
    summary_lines.append("")
    summary_text = "\n".join(summary_lines)

    with open(output_dir / "b1_network_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # -------------------------
    # Save a short dataset card draft
    # -------------------------
    dataset_card_text = build_dataset_card_text(config, substrate_table)
    with open(output_dir / "b1_dataset_card_draft.txt", "w", encoding="utf-8") as f:
        f.write(dataset_card_text)

    # -------------------------
    # Create report-ready plots
    # -------------------------
    plot_patch_layout(node_table, output_dir / "b1_patch_layout.png")
    plot_network_overlay(
        node_table=node_table,
        W=A_sp,
        title="Spontaneous layer network overlay",
        score_col="SpontScore",
        out_path=output_dir / "b1_spont_network_overlay.png"
    )
    plot_network_overlay(
        node_table=node_table,
        W=A_pc,
        title="Paced layer network overlay",
        score_col="PacedScore",
        out_path=output_dir / "b1_paced_network_overlay.png"
    )
    plot_supra_heatmap(supra, output_dir / "b1_supra_matrix_heatmap.png")
    plot_feature_scatter_overview(node_table, output_dir / "b1_feature_scatter_overview.png")

    # -------------------------
    # Final console output
    # -------------------------
    print("\nCodeB1 outputs saved to:")
    print(output_dir)

    print("\nKey files:")
    print(" - b1_node_table.csv")
    print(" - b1_spont_adjacency.csv")
    print(" - b1_paced_adjacency.csv")
    print(" - b1_supra_adjacency.csv")
    print(" - b1_network_summary.txt")
    print(" - b1_dataset_card_draft.txt")

    print("\nSummary:")
    print(summary_text)