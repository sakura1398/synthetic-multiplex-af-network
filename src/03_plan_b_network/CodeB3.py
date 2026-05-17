# codeB3.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dataclasses import dataclass, replace
from pathlib import Path


print("RUNNING CODEB3: null model + sensitivity analysis for B26")


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class B3Config:
    # Output settings
    dpi: int = 220
    node_size: int = 170
    top_k: int = 8

    # Numerical stability
    eps: float = 1e-8

    # Base refined graph settings (same spirit as B26)
    spatial_sigma_mm: float = 18.0
    temporal_sigma_steps: float = 2.0
    phenotype_sigma_sp: float = 2.0
    phenotype_sigma_pc: float = 2.0

    k_spatial: int = 4
    k_temporal: int = 3
    k_phenotype: int = 3

    alpha_spatial: float = 0.45
    beta_temporal: float = 0.20
    gamma_phenotype: float = 0.35

    omega_mode: str = "mean_intralayer_weight"

    # Null-model settings
    n_null: int = 200
    random_seed: int = 123

    # Sensitivity settings for hybrid bridge
    bridge_boundary_weights: tuple = (1.0, 0.8, 0.6, 0.5)


# =========================================================
# 2) Utility functions
# =========================================================

def minmax_array(x, eps: float = 1e-8):
    """
    Min-max normalize a 1D array.
    """
    x = np.asarray(x, dtype=float)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + eps)


def row_standardize_df(df: pd.DataFrame, cols):
    """
    Standardize selected columns so that phenotype distance is not dominated by scale.
    """
    out = df[cols].copy().astype(float)
    for col in cols:
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        if sd < 1e-12:
            sd = 1.0
        out[col] = (out[col] - mu) / sd
    return out


def pairwise_squared_dist(X: np.ndarray):
    """
    Compute pairwise squared Euclidean distances between rows of X.
    """
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(diff ** 2, axis=2)


def keep_top_k_per_row(W: np.ndarray, k: int):
    """
    Keep only the top-k non-diagonal weights in each row.
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


def symmetrize_matrix(W: np.ndarray, mode: str = "max"):
    """
    Symmetrize a matrix using max or mean.
    """
    if mode == "max":
        return np.maximum(W, W.T)
    if mode == "mean":
        return 0.5 * (W + W.T)
    raise ValueError("mode must be 'max' or 'mean'")


def build_weighted_graph_from_adjacency(W: np.ndarray):
    """
    Build an undirected weighted graph from adjacency matrix.
    """
    return nx.from_numpy_array(W)


def add_distance_attribute(G: nx.Graph, eps: float = 1e-8):
    """
    Convert similarity weights into distances for shortest-path calculations.
    """
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.0)
        data["distance"] = 1.0 / (w + eps)


def safe_corr(x, y):
    """
    Safely compute Pearson correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    """
    Compute Spearman correlation using rank transformation.
    """
    sx = pd.Series(x).rank(method="average").to_numpy()
    sy = pd.Series(y).rank(method="average").to_numpy()
    return safe_corr(sx, sy)


def empirical_p_value_geq(null_values, observed_value):
    """
    One-sided empirical p-value for 'greater than or equal' significance.
    """
    null_values = np.asarray(null_values, dtype=float)
    return float((np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1))


def top_k_ids(df: pd.DataFrame, metric_col: str, k: int, ascending: bool = False):
    """
    Return top-k placement IDs by a metric.
    """
    return df.sort_values(metric_col, ascending=ascending)["placement_id"].astype(int).head(k).tolist()


def top_k_overlap(ids_a, ids_b):
    """
    Compute top-k set overlap ratio.
    """
    a = set(ids_a)
    b = set(ids_b)
    if len(a) == 0:
        return np.nan
    return float(len(a.intersection(b)) / len(a))


# =========================================================
# 3) Data loading
# =========================================================

def load_b1_node_table(input_dir: Path):
    """
    Load the node table produced by CodeB1.
    """
    return pd.read_csv(input_dir / "b1_node_table.csv")


# =========================================================
# 4) Similarity builders
# =========================================================

def build_spatial_similarity(node_table: pd.DataFrame, sigma_mm: float):
    """
    Build spatial similarity from patch coordinates.
    """
    XY = node_table[["center_x", "center_y"]].to_numpy(dtype=float)
    d2 = pairwise_squared_dist(XY)
    W = np.exp(-d2 / (2.0 * sigma_mm ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def build_temporal_similarity(node_table: pd.DataFrame, sigma_steps: float):
    """
    Build graded temporal similarity from placement order.
    """
    t = node_table["time_index"].to_numpy(dtype=float).reshape(-1, 1)
    dt2 = pairwise_squared_dist(t)
    W = np.exp(-dt2 / (2.0 * sigma_steps ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def build_phenotype_similarity_from_source(source_df: pd.DataFrame, cols, sigma_f: float):
    """
    Build phenotype similarity from a given source DataFrame.
    This allows null models to permute phenotype assignments while keeping geometry fixed.
    """
    Z = row_standardize_df(source_df, cols).to_numpy(dtype=float)
    d2 = pairwise_squared_dist(Z)
    W = np.exp(-d2 / (2.0 * sigma_f ** 2))
    np.fill_diagonal(W, 0.0)
    return W, Z


def build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W, config: B3Config):
    """
    Build the refined adjacency for one layer from spatial, temporal, and phenotype similarities.
    """
    W_spatial = symmetrize_matrix(keep_top_k_per_row(spatial_W, config.k_spatial), mode="max")
    W_temporal = symmetrize_matrix(keep_top_k_per_row(temporal_W, config.k_temporal), mode="max")
    W_pheno = symmetrize_matrix(keep_top_k_per_row(phenotype_W, config.k_phenotype), mode="max")

    A = (
        config.alpha_spatial * W_spatial
        + config.beta_temporal * W_temporal
        + config.gamma_phenotype * W_pheno
    )

    A = symmetrize_matrix(A, mode="max")
    np.fill_diagonal(A, 0.0)
    return A


def choose_omega(A_sp: np.ndarray, A_pc: np.ndarray, config: B3Config):
    """
    Choose inter-layer coupling from mean nonzero intra-layer weight.
    """
    if config.omega_mode != "mean_intralayer_weight":
        raise ValueError("Unsupported omega mode")

    nz_sp = A_sp[A_sp > 0]
    nz_pc = A_pc[A_pc > 0]
    pooled = np.concatenate([nz_sp, nz_pc]) if (len(nz_sp) + len(nz_pc)) > 0 else np.array([1.0])
    return float(np.mean(pooled))


def build_supra_adjacency(A_sp: np.ndarray, A_pc: np.ndarray, omega: float):
    """
    Build multiplex supra-adjacency matrix.
    """
    n = A_sp.shape[0]
    supra = np.zeros((2 * n, 2 * n), dtype=float)
    supra[:n, :n] = A_sp
    supra[n:, n:] = A_pc
    supra[:n, n:] = omega * np.eye(n)
    supra[n:, :n] = omega * np.eye(n)
    return supra


# =========================================================
# 5) Metric functions
# =========================================================

def weighted_boundary_score(A: np.ndarray, Z: np.ndarray, eps: float = 1e-8):
    """
    Compute phenotype boundary score:
    a node is high if it is connected to neighbors with different phenotype vectors.
    """
    n = A.shape[0]
    out = np.zeros(n, dtype=float)

    for i in range(n):
        weights = A[i]
        denom = np.sum(weights)

        if denom < eps:
            out[i] = 0.0
            continue

        diffs = np.sqrt(np.sum((Z - Z[i]) ** 2, axis=1))
        out[i] = float(np.sum(weights * diffs) / (denom + eps))

    return out


def supra_betweenness(A_supra: np.ndarray, eps: float):
    """
    Compute betweenness centrality on the supra-graph using inverse similarity as distance.
    """
    G = build_weighted_graph_from_adjacency(A_supra)
    add_distance_attribute(G, eps=eps)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
    return bc


# =========================================================
# 6) Main refined metric builder
# =========================================================

def compute_b26_style_metrics(node_table: pd.DataFrame, config: B3Config,
                              perm_sp=None, perm_pc=None):
    """
    Compute B26-style metrics.

    If perm_sp or perm_pc are provided, the phenotype assignments are shuffled
    while geometry and time remain unchanged.
    """
    sp_cols = ["STD_p", "ICS_p", "Disp_p", "FI_burden", "LV_burden"]
    pc_cols = ["STD_paced", "ICS_paced", "LDH_paced", "LV_burden_paced"]

    n = len(node_table)

    spatial_W = build_spatial_similarity(node_table, config.spatial_sigma_mm)
    temporal_W = build_temporal_similarity(node_table, config.temporal_sigma_steps)

    if perm_sp is None:
        sp_source = node_table.copy()
    else:
        sp_source = node_table.iloc[perm_sp].reset_index(drop=True)

    if perm_pc is None:
        pc_source = node_table.copy()
    else:
        pc_source = node_table.iloc[perm_pc].reset_index(drop=True)

    phenotype_W_sp, Z_sp = build_phenotype_similarity_from_source(
        sp_source, sp_cols, config.phenotype_sigma_sp
    )
    phenotype_W_pc, Z_pc = build_phenotype_similarity_from_source(
        pc_source, pc_cols, config.phenotype_sigma_pc
    )

    A_sp = build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W_sp, config)
    A_pc = build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W_pc, config)

    omega = choose_omega(A_sp, A_pc, config)
    A_supra = build_supra_adjacency(A_sp, A_pc, omega)

    bc_supra = supra_betweenness(A_supra, eps=config.eps)

    bet_sp = np.array([bc_supra[i] for i in range(n)], dtype=float)
    bet_pc = np.array([bc_supra[n + i] for i in range(n)], dtype=float)
    bet_mean = 0.5 * (bet_sp + bet_pc)
    bet_mean_norm = minmax_array(bet_mean, eps=config.eps)

    boundary_sp = minmax_array(weighted_boundary_score(A_sp, Z_sp, eps=config.eps), eps=config.eps)
    boundary_pc = minmax_array(weighted_boundary_score(A_pc, Z_pc, eps=config.eps), eps=config.eps)
    boundary_mean = 0.5 * (boundary_sp + boundary_pc)

    u_sp = node_table["u_sp"].to_numpy(dtype=float)
    u_pc = node_table["u_pc"].to_numpy(dtype=float)

    core_score = np.minimum(u_sp, u_pc)
    signed_discordance = u_sp - u_pc
    abs_discordance = np.abs(signed_discordance)

    out = pd.DataFrame({
        "placement_id": node_table["placement_id"].astype(int),
        "time_index": node_table["time_index"].astype(int),
        "center_x": node_table["center_x"].astype(float),
        "center_y": node_table["center_y"].astype(float),

        "u_sp": u_sp,
        "u_pc": u_pc,

        "core_score": core_score,
        "signed_discordance": signed_discordance,
        "abs_discordance": abs_discordance,

        "betweenness_supra_mean_refined": bet_mean,
        "betweenness_supra_mean_norm_refined": bet_mean_norm,

        "boundary_primary_bridge": boundary_mean,

        "mean_block_field": node_table["mean_block_field"].astype(float),
        "mean_vuln_field": node_table["mean_vuln_field"].astype(float),
        "mean_interaction": node_table["mean_interaction"].astype(float),
    })

    artifacts = {
        "A_sp": A_sp,
        "A_pc": A_pc,
        "A_supra": A_supra,
        "omega": omega,
        "boundary_mean": boundary_mean,
        "bet_mean_norm": bet_mean_norm,
    }

    return out, artifacts


# =========================================================
# 7) Null model analysis
# =========================================================

def run_null_model_analysis(node_table: pd.DataFrame, observed_df: pd.DataFrame,
                            config: B3Config):
    """
    Run null models by permuting phenotype assignments across placements
    while keeping geometry and time fixed.
    """
    rng = np.random.default_rng(config.random_seed)
    n = len(node_table)

    observed_top_boundary = top_k_ids(observed_df, "boundary_primary_bridge", config.top_k, ascending=False)

    rows = []

    obs_corr_boundary_interaction = safe_corr(
        observed_df["boundary_primary_bridge"], observed_df["mean_interaction"]
    )
    obs_corr_boundary_time = safe_corr(
        observed_df["boundary_primary_bridge"], observed_df["time_index"]
    )

    for trial in range(config.n_null):
        perm_sp = rng.permutation(n)
        perm_pc = rng.permutation(n)

        null_df, _ = compute_b26_style_metrics(
            node_table=node_table,
            config=config,
            perm_sp=perm_sp,
            perm_pc=perm_pc
        )

        null_top_boundary = top_k_ids(null_df, "boundary_primary_bridge", config.top_k, ascending=False)

        rows.append({
            "trial": trial,
            "corr_boundary_interaction": safe_corr(
                null_df["boundary_primary_bridge"], null_df["mean_interaction"]
            ),
            "corr_boundary_time": safe_corr(
                null_df["boundary_primary_bridge"], null_df["time_index"]
            ),
            "corr_topology_interaction": safe_corr(
                null_df["betweenness_supra_mean_refined"], null_df["mean_interaction"]
            ),
            "topk_overlap_with_observed_boundary": top_k_overlap(observed_top_boundary, null_top_boundary),
            "mean_boundary_score": float(null_df["boundary_primary_bridge"].mean()),
        })

    null_df_out = pd.DataFrame(rows)

    summary = {
        "observed_corr_boundary_interaction": obs_corr_boundary_interaction,
        "observed_corr_boundary_time": obs_corr_boundary_time,
        "empirical_p_boundary_interaction": empirical_p_value_geq(
            null_df_out["corr_boundary_interaction"].to_numpy(),
            obs_corr_boundary_interaction
        ),
        "empirical_p_boundary_time": empirical_p_value_geq(
            null_df_out["corr_boundary_time"].to_numpy(),
            obs_corr_boundary_time
        ),
    }

    return null_df_out, summary


# =========================================================
# 8) Sensitivity analysis
# =========================================================

def run_parameter_sensitivity(node_table: pd.DataFrame, observed_df: pd.DataFrame,
                              config: B3Config):
    """
    Evaluate how stable the boundary-primary bridge remains under parameter changes.
    """
    base_top_boundary = top_k_ids(observed_df, "boundary_primary_bridge", config.top_k, ascending=False)
    base_boundary = observed_df["boundary_primary_bridge"].to_numpy()

    variants = [
        ("base", {}),
        ("spatial_heavy", {"alpha_spatial": 0.60, "beta_temporal": 0.15, "gamma_phenotype": 0.25}),
        ("temporal_heavy", {"alpha_spatial": 0.30, "beta_temporal": 0.45, "gamma_phenotype": 0.25}),
        ("phenotype_heavy", {"alpha_spatial": 0.25, "beta_temporal": 0.15, "gamma_phenotype": 0.60}),
        ("local_spatial_sigma", {"spatial_sigma_mm": 12.0}),
        ("global_spatial_sigma", {"spatial_sigma_mm": 28.0}),
        ("short_temporal_sigma", {"temporal_sigma_steps": 1.0}),
        ("long_temporal_sigma", {"temporal_sigma_steps": 4.0}),
    ]

    rows = []

    for label, overrides in variants:
        cfg_variant = replace(config, **overrides)
        df_var, art_var = compute_b26_style_metrics(node_table, cfg_variant)

        top_boundary_var = top_k_ids(df_var, "boundary_primary_bridge", config.top_k, ascending=False)

        rows.append({
            "variant": label,
            "omega": art_var["omega"],
            "corr_boundary_interaction": safe_corr(
                df_var["boundary_primary_bridge"], df_var["mean_interaction"]
            ),
            "corr_boundary_time": safe_corr(
                df_var["boundary_primary_bridge"], df_var["time_index"]
            ),
            "corr_topology_interaction": safe_corr(
                df_var["betweenness_supra_mean_refined"], df_var["mean_interaction"]
            ),
            "topk_overlap_with_base": top_k_overlap(base_top_boundary, top_boundary_var),
            "spearman_boundary_vs_base": safe_spearman(
                base_boundary,
                df_var["boundary_primary_bridge"].to_numpy()
            ),
            "top_boundary_node": int(top_boundary_var[0]),
        })

    return pd.DataFrame(rows)


# =========================================================
# 9) Bridge-mix sensitivity
# =========================================================

def run_bridge_mix_sensitivity(observed_df: pd.DataFrame, config: B3Config):
    """
    Test how bridge interpretation changes when topology is mixed back into the metric.
    """
    boundary = observed_df["boundary_primary_bridge"].to_numpy(dtype=float)
    bet_norm = minmax_array(
        observed_df["betweenness_supra_mean_refined"].to_numpy(dtype=float),
        eps=config.eps
    )

    rows = []

    for w_boundary in config.bridge_boundary_weights:
        w_topology = 1.0 - w_boundary
        hybrid = w_boundary * boundary + w_topology * bet_norm

        rows.append({
            "boundary_weight": float(w_boundary),
            "topology_weight": float(w_topology),
            "corr_with_interaction": safe_corr(hybrid, observed_df["mean_interaction"]),
            "corr_with_time": safe_corr(hybrid, observed_df["time_index"]),
            "corr_with_block": safe_corr(hybrid, observed_df["mean_block_field"]),
        })

    return pd.DataFrame(rows)


# =========================================================
# 10) Plotting
# =========================================================

def plot_null_histograms(null_df: pd.DataFrame, null_summary: dict, out_path: Path, config: B3Config):
    """
    Plot null distributions with observed values marked as vertical lines.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.hist(null_df["corr_boundary_interaction"], bins=20)
    ax.axvline(null_summary["observed_corr_boundary_interaction"], linestyle="--")
    ax.set_title("Null: corr(boundary, interaction)")
    ax.set_xlabel("correlation")
    ax.set_ylabel("count")

    ax = axes[1]
    ax.hist(null_df["corr_boundary_time"], bins=20)
    ax.axvline(null_summary["observed_corr_boundary_time"], linestyle="--")
    ax.set_title("Null: corr(boundary, time)")
    ax.set_xlabel("correlation")
    ax.set_ylabel("count")

    ax = axes[2]
    ax.hist(null_df["topk_overlap_with_observed_boundary"], bins=10)
    ax.set_title("Null: top-k overlap with observed boundary")
    ax.set_xlabel("overlap ratio")
    ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_bars(sens_df: pd.DataFrame, out_path: Path, config: B3Config):
    """
    Plot sensitivity analysis results across parameter variants.
    """
    labels = sens_df["variant"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.bar(x - width / 2, sens_df["corr_boundary_interaction"], width=width, label="corr(boundary, interaction)")
    ax.bar(x + width / 2, sens_df["corr_boundary_time"], width=width, label="corr(boundary, time)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Sensitivity of boundary-primary bridge correlations")
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, sens_df["topk_overlap_with_base"], width=width, label="top-k overlap with base")
    ax.bar(x + width / 2, sens_df["spearman_boundary_vs_base"], width=width, label="rank correlation vs base")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Sensitivity of node ranking stability")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_mix_sensitivity(mix_df: pd.DataFrame, out_path: Path, config: B3Config):
    """
    Plot how bridge correlations change as topology is mixed back into the bridge metric.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(mix_df["boundary_weight"], mix_df["corr_with_interaction"], marker="o", label="corr with interaction")
    ax.plot(mix_df["boundary_weight"], mix_df["corr_with_time"], marker="o", label="corr with time")
    ax.plot(mix_df["boundary_weight"], mix_df["corr_with_block"], marker="o", label="corr with block")

    ax.set_title("Bridge-mix sensitivity")
    ax.set_xlabel("boundary weight in bridge metric")
    ax.set_ylabel("correlation")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 11) Summary writer
# =========================================================

def build_b3_summary(observed_df: pd.DataFrame,
                     null_df: pd.DataFrame,
                     null_summary: dict,
                     sens_df: pd.DataFrame,
                     mix_df: pd.DataFrame,
                     config: B3Config):
    """
    Build a compact text summary for B3.
    """
    top_boundary = top_k_ids(observed_df, "boundary_primary_bridge", config.top_k, ascending=False)
    top_core = top_k_ids(observed_df, "core_score", config.top_k, ascending=False)

    lines = []
    lines.append("CODEB3 NULL MODEL + SENSITIVITY SUMMARY")
    lines.append("=" * 68)
    lines.append("")

    lines.append(f"Observed top boundary-primary bridge nodes: {top_boundary}")
    lines.append(f"Observed top core nodes: {top_core}")
    lines.append("")

    lines.append("Observed boundary-primary bridge diagnostics:")
    lines.append(
        f"  corr(boundary_primary_bridge, mean_interaction) = "
        f"{null_summary['observed_corr_boundary_interaction']:.4f}"
    )
    lines.append(
        f"  corr(boundary_primary_bridge, time_index)       = "
        f"{null_summary['observed_corr_boundary_time']:.4f}"
    )
    lines.append(
        f"  empirical p-value vs null for interaction corr  = "
        f"{null_summary['empirical_p_boundary_interaction']:.4f}"
    )
    lines.append(
        f"  empirical p-value vs null for time corr         = "
        f"{null_summary['empirical_p_boundary_time']:.4f}"
    )
    lines.append("")

    lines.append("Null model distribution summary:")
    lines.append(
        f"  mean null corr(boundary, interaction) = "
        f"{null_df['corr_boundary_interaction'].mean():.4f}"
    )
    lines.append(
        f"  std  null corr(boundary, interaction) = "
        f"{null_df['corr_boundary_interaction'].std(ddof=0):.4f}"
    )
    lines.append(
        f"  mean null corr(boundary, time)        = "
        f"{null_df['corr_boundary_time'].mean():.4f}"
    )
    lines.append(
        f"  std  null corr(boundary, time)        = "
        f"{null_df['corr_boundary_time'].std(ddof=0):.4f}"
    )
    lines.append(
        f"  mean null top-k overlap               = "
        f"{null_df['topk_overlap_with_observed_boundary'].mean():.4f}"
    )
    lines.append("")

    lines.append("Sensitivity summary by parameter variant:")
    for _, row in sens_df.iterrows():
        lines.append(
            f"  {row['variant']}: "
            f"corr_int={row['corr_boundary_interaction']:.4f}, "
            f"corr_time={row['corr_boundary_time']:.4f}, "
            f"overlap={row['topk_overlap_with_base']:.4f}, "
            f"rank_corr={row['spearman_boundary_vs_base']:.4f}, "
            f"top_node={int(row['top_boundary_node'])}"
        )
    lines.append("")

    lines.append("Bridge-mix sensitivity:")
    for _, row in mix_df.iterrows():
        lines.append(
            f"  boundary_weight={row['boundary_weight']:.2f}, "
            f"topology_weight={row['topology_weight']:.2f}, "
            f"corr_int={row['corr_with_interaction']:.4f}, "
            f"corr_time={row['corr_with_time']:.4f}, "
            f"corr_block={row['corr_with_block']:.4f}"
        )
    lines.append("")

    return "\n".join(lines)


# =========================================================
# 12) Main
# =========================================================

if __name__ == "__main__":
    config = B3Config()

    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "codeB1_outputs"
    output_dir = script_dir / "codeB3_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load node table
    # -------------------------
    node_table = load_b1_node_table(input_dir)

    # -------------------------
    # Compute observed B26-style metrics
    # -------------------------
    observed_df, observed_artifacts = compute_b26_style_metrics(node_table, config)
    observed_df.to_csv(output_dir / "b3_observed_metrics.csv", index=False)

    # -------------------------
    # Null model analysis
    # -------------------------
    null_df, null_summary = run_null_model_analysis(node_table, observed_df, config)
    null_df.to_csv(output_dir / "b3_null_model_summary.csv", index=False)

    # -------------------------
    # Sensitivity analysis
    # -------------------------
    sens_df = run_parameter_sensitivity(node_table, observed_df, config)
    sens_df.to_csv(output_dir / "b3_parameter_sensitivity.csv", index=False)

    # -------------------------
    # Bridge-mix sensitivity
    # -------------------------
    mix_df = run_bridge_mix_sensitivity(observed_df, config)
    mix_df.to_csv(output_dir / "b3_bridge_mix_sensitivity.csv", index=False)

    # -------------------------
    # Save refined adjacency matrices
    # -------------------------
    node_ids = observed_df["placement_id"].astype(int).tolist()
    pd.DataFrame(
        observed_artifacts["A_sp"], index=node_ids, columns=node_ids
    ).to_csv(output_dir / "b3_observed_spont_adjacency.csv")

    pd.DataFrame(
        observed_artifacts["A_pc"], index=node_ids, columns=node_ids
    ).to_csv(output_dir / "b3_observed_paced_adjacency.csv")

    supra_index = [f"s_{i}" for i in node_ids] + [f"p_{i}" for i in node_ids]
    pd.DataFrame(
        observed_artifacts["A_supra"], index=supra_index, columns=supra_index
    ).to_csv(output_dir / "b3_observed_supra_adjacency.csv")

    # -------------------------
    # Figures
    # -------------------------
    plot_null_histograms(
        null_df=null_df,
        null_summary=null_summary,
        out_path=output_dir / "b3_null_histograms.png",
        config=config
    )

    plot_sensitivity_bars(
        sens_df=sens_df,
        out_path=output_dir / "b3_sensitivity_bars.png",
        config=config
    )

    plot_bridge_mix_sensitivity(
        mix_df=mix_df,
        out_path=output_dir / "b3_bridge_mix_sensitivity.png",
        config=config
    )

    # -------------------------
    # Summary text
    # -------------------------
    summary_text = build_b3_summary(
        observed_df=observed_df,
        null_df=null_df,
        null_summary=null_summary,
        sens_df=sens_df,
        mix_df=mix_df,
        config=config
    )

    with open(output_dir / "b3_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # -------------------------
    # Print console summary
    # -------------------------
    print("\nCodeB3 outputs saved to:")
    print(output_dir)

    print("\nKey files:")
    print(" - b3_observed_metrics.csv")
    print(" - b3_null_model_summary.csv")
    print(" - b3_parameter_sensitivity.csv")
    print(" - b3_bridge_mix_sensitivity.csv")
    print(" - b3_observed_spont_adjacency.csv")
    print(" - b3_observed_paced_adjacency.csv")
    print(" - b3_observed_supra_adjacency.csv")
    print(" - b3_null_histograms.png")
    print(" - b3_sensitivity_bars.png")
    print(" - b3_bridge_mix_sensitivity.png")
    print(" - b3_summary.txt")

    print("\nSummary:")
    print(summary_text)