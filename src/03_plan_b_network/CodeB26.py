# codeB26.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dataclasses import dataclass
from pathlib import Path


print("RUNNING CODEB26: boundary-primary AF-aware bridge refinement")


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class B26Config:
    # Figure settings
    dpi: int = 220
    node_size: int = 170
    annotate_top_n: int = 8
    top_k_report: int = 8

    # Numerical stability
    eps: float = 1e-8

    # Spatial similarity settings
    spatial_sigma_mm: float = 18.0
    k_spatial: int = 4

    # Temporal similarity settings
    temporal_sigma_steps: float = 2.0
    k_temporal: int = 3

    # Phenotype similarity settings
    phenotype_sigma_sp: float = 2.0
    phenotype_sigma_pc: float = 2.0
    k_phenotype: int = 3

    # Layer combination weights
    alpha_spatial: float = 0.45
    beta_temporal: float = 0.20
    gamma_phenotype: float = 0.35

    # Inter-layer coupling
    omega_mode: str = "mean_intralayer_weight"

    # New bridge mixture:
    # Boundary is now the primary signal.
    bridge_w_boundary: float = 0.80
    bridge_w_topology: float = 0.20


# =========================================================
# 2) Utility functions
# =========================================================

def minmax_array(x, eps: float = 1e-8):
    """
    Min-max normalize a 1D NumPy array.
    """
    x = np.asarray(x, dtype=float)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + eps)


def row_standardize_df(df: pd.DataFrame, cols):
    """
    Standardize selected columns so phenotype distances are not dominated by scale.
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
    Build a weighted undirected NetworkX graph from an adjacency matrix.
    """
    return nx.from_numpy_array(W)


def add_distance_attribute(G: nx.Graph, eps: float = 1e-8):
    """
    Convert similarity weights into distances for shortest-path metrics.
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


def top_nodes_by_metric(df: pd.DataFrame, metric_col: str, top_k: int, ascending: bool = False):
    """
    Return top-k rows sorted by a metric.
    """
    return df.sort_values(metric_col, ascending=ascending).head(top_k).copy()


# =========================================================
# 3) Loading
# =========================================================

def load_b1_node_table(input_dir: Path):
    """
    Load the CodeB1 node table.
    """
    return pd.read_csv(input_dir / "b1_node_table.csv")


# =========================================================
# 4) Similarity builders
# =========================================================

def build_spatial_similarity(node_table: pd.DataFrame, sigma_mm: float):
    """
    Build spatial similarity from patch center coordinates.
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


def build_phenotype_similarity(node_table: pd.DataFrame, cols, sigma_f: float):
    """
    Build phenotype similarity from raw Plan A descriptors.
    """
    Z = row_standardize_df(node_table, cols).to_numpy(dtype=float)
    d2 = pairwise_squared_dist(Z)
    W = np.exp(-d2 / (2.0 * sigma_f ** 2))
    np.fill_diagonal(W, 0.0)
    return W, Z


def build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W, config: B26Config):
    """
    Build refined layer adjacency from spatial, temporal, and phenotype similarities.
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

    return A, W_spatial, W_temporal, W_pheno


def choose_omega(A_sp: np.ndarray, A_pc: np.ndarray, config: B26Config):
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
    Build the multiplex supra-adjacency matrix.
    """
    n = A_sp.shape[0]
    supra = np.zeros((2 * n, 2 * n), dtype=float)
    supra[:n, :n] = A_sp
    supra[n:, n:] = A_pc
    supra[:n, n:] = omega * np.eye(n)
    supra[n:, :n] = omega * np.eye(n)
    return supra


# =========================================================
# 5) Metrics
# =========================================================

def weighted_node_strength(G: nx.Graph):
    """
    Compute weighted degree (node strength).
    """
    return dict(G.degree(weight="weight"))


def weighted_clustering(G: nx.Graph):
    """
    Compute weighted clustering coefficient.
    """
    return nx.clustering(G, weight="weight")


def weighted_degree(G: nx.Graph):
    """
    Compute unweighted degree.
    """
    return dict(G.degree())


def supra_betweenness(A_supra: np.ndarray, eps: float):
    """
    Compute supra-graph betweenness centrality using inverse similarity as distance.
    """
    G = build_weighted_graph_from_adjacency(A_supra)
    add_distance_attribute(G, eps=eps)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
    return bc, G


def weighted_boundary_score(A: np.ndarray, Z: np.ndarray, eps: float = 1e-8):
    """
    Compute a phenotype boundary score.

    A node gets a high boundary score if it is connected to neighbors
    with noticeably different phenotype vectors.
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


# =========================================================
# 6) Build B26 role table
# =========================================================

def build_b26_role_table(node_table: pd.DataFrame, config: B26Config):
    """
    Build the B26 node-role table.

    Main change from B25:
    - phenotype boundary becomes the primary bridge signal
    - topology becomes secondary support
    """
    sp_cols = ["STD_p", "ICS_p", "Disp_p", "FI_burden", "LV_burden"]
    pc_cols = ["STD_paced", "ICS_paced", "LDH_paced", "LV_burden_paced"]

    spatial_W = build_spatial_similarity(node_table, config.spatial_sigma_mm)
    temporal_W = build_temporal_similarity(node_table, config.temporal_sigma_steps)

    phenotype_W_sp, Z_sp = build_phenotype_similarity(node_table, sp_cols, config.phenotype_sigma_sp)
    phenotype_W_pc, Z_pc = build_phenotype_similarity(node_table, pc_cols, config.phenotype_sigma_pc)

    A_sp, _, _, _ = build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W_sp, config)
    A_pc, _, _, _ = build_refined_layer_adjacency(spatial_W, temporal_W, phenotype_W_pc, config)

    omega = choose_omega(A_sp, A_pc, config)
    A_supra = build_supra_adjacency(A_sp, A_pc, omega)

    G_sp = build_weighted_graph_from_adjacency(A_sp)
    G_pc = build_weighted_graph_from_adjacency(A_pc)

    strength_sp = weighted_node_strength(G_sp)
    strength_pc = weighted_node_strength(G_pc)

    degree_sp = weighted_degree(G_sp)
    degree_pc = weighted_degree(G_pc)

    clustering_sp = weighted_clustering(G_sp)
    clustering_pc = weighted_clustering(G_pc)

    bc_supra, G_supra = supra_betweenness(A_supra, eps=config.eps)

    n = len(node_table)
    bet_sp = np.array([bc_supra[i] for i in range(n)], dtype=float)
    bet_pc = np.array([bc_supra[n + i] for i in range(n)], dtype=float)
    bet_mean = 0.5 * (bet_sp + bet_pc)
    bet_mean_norm = minmax_array(bet_mean, eps=config.eps)

    boundary_sp_raw = weighted_boundary_score(A_sp, Z_sp, eps=config.eps)
    boundary_pc_raw = weighted_boundary_score(A_pc, Z_pc, eps=config.eps)

    boundary_sp = minmax_array(boundary_sp_raw, eps=config.eps)
    boundary_pc = minmax_array(boundary_pc_raw, eps=config.eps)
    boundary_mean = 0.5 * (boundary_sp + boundary_pc)

    # New primary bridge definition:
    # The main bridge interpretation is boundary-first.
    boundary_primary_bridge = boundary_mean

    # Secondary hybrid score:
    # Topology is now only a support term.
    hybrid_bridge_v2 = (
        config.bridge_w_boundary * boundary_mean
        + config.bridge_w_topology * bet_mean_norm
    )

    u_sp = node_table["u_sp"].to_numpy(dtype=float)
    u_pc = node_table["u_pc"].to_numpy(dtype=float)

    core_score = np.minimum(u_sp, u_pc)
    signed_discordance = u_sp - u_pc
    abs_discordance = np.abs(signed_discordance)

    time_idx = node_table["time_index"].to_numpy(dtype=float)
    recency_weight = minmax_array(time_idx, eps=config.eps)

    df = pd.DataFrame({
        "placement_id": node_table["placement_id"].astype(int),
        "time_index": node_table["time_index"].astype(int),
        "time_min": node_table["time_min"].astype(float),
        "center_x": node_table["center_x"].astype(float),
        "center_y": node_table["center_y"].astype(float),
        "angle_deg": node_table["angle_deg"].astype(float),

        "u_sp": u_sp,
        "u_pc": u_pc,
        "SpontScore": node_table["SpontScore"].astype(float),
        "PacedScore": node_table["PacedScore"].astype(float),

        "strength_sp_refined": [strength_sp[i] for i in range(n)],
        "strength_pc_refined": [strength_pc[i] for i in range(n)],

        "degree_sp_refined": [degree_sp[i] for i in range(n)],
        "degree_pc_refined": [degree_pc[i] for i in range(n)],

        "clustering_sp_refined": [clustering_sp[i] for i in range(n)],
        "clustering_pc_refined": [clustering_pc[i] for i in range(n)],

        "betweenness_sp_layer_refined": bet_sp,
        "betweenness_pc_layer_refined": bet_pc,
        "betweenness_supra_mean_refined": bet_mean,
        "betweenness_supra_mean_norm_refined": bet_mean_norm,

        "boundary_sp_refined": boundary_sp,
        "boundary_pc_refined": boundary_pc,
        "boundary_mean_refined": boundary_mean,

        "boundary_primary_bridge": boundary_primary_bridge,
        "hybrid_bridge_v2": hybrid_bridge_v2,

        "core_score": core_score,
        "signed_discordance": signed_discordance,
        "abs_discordance": abs_discordance,

        "recency_weight": recency_weight,

        "STD_p": node_table["STD_p"].astype(float),
        "ICS_p": node_table["ICS_p"].astype(float),
        "Disp_p": node_table["Disp_p"].astype(float),
        "FI_burden": node_table["FI_burden"].astype(float),
        "LV_burden": node_table["LV_burden"].astype(float),

        "STD_paced": node_table["STD_paced"].astype(float),
        "ICS_paced": node_table["ICS_paced"].astype(float),
        "LDH_paced": node_table["LDH_paced"].astype(float),
        "LV_burden_paced": node_table["LV_burden_paced"].astype(float),

        "mean_block_field": node_table["mean_block_field"].astype(float),
        "mean_lv_field": node_table["mean_lv_field"].astype(float),
        "mean_vuln_field": node_table["mean_vuln_field"].astype(float),
        "mean_interaction": node_table["mean_interaction"].astype(float),
    })

    artifacts = {
        "A_sp_refined": A_sp,
        "A_pc_refined": A_pc,
        "A_supra_refined": A_supra,
        "omega": omega,
    }

    return df, artifacts


# =========================================================
# 7) Summary
# =========================================================

def build_b26_summary(df: pd.DataFrame, artifacts: dict, config: B26Config):
    """
    Build a text summary for B26.
    """
    top_core = top_nodes_by_metric(df, "core_score", config.top_k_report, ascending=False)
    top_boundary_bridge = top_nodes_by_metric(df, "boundary_primary_bridge", config.top_k_report, ascending=False)
    top_hybrid = top_nodes_by_metric(df, "hybrid_bridge_v2", config.top_k_report, ascending=False)

    lines = []
    lines.append("CODEB26 BOUNDARY-PRIMARY BRIDGE SUMMARY")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"omega_refined = {artifacts['omega']:.6f}")
    lines.append("")

    lines.append("Top core nodes:")
    for _, row in top_core.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: core={row['core_score']:.4f}, "
            f"u_sp={row['u_sp']:.4f}, u_pc={row['u_pc']:.4f}"
        )
    lines.append("")

    lines.append("Top boundary-primary bridge nodes:")
    for _, row in top_boundary_bridge.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: boundary_bridge={row['boundary_primary_bridge']:.4f}, "
            f"bet={row['betweenness_supra_mean_refined']:.4f}"
        )
    lines.append("")

    lines.append("Top hybrid bridge v2 nodes:")
    for _, row in top_hybrid.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: hybrid_v2={row['hybrid_bridge_v2']:.4f}, "
            f"boundary={row['boundary_mean_refined']:.4f}, "
            f"bet={row['betweenness_supra_mean_refined']:.4f}"
        )
    lines.append("")

    lines.append("Correlation diagnostics:")
    lines.append(
        f"  corr(boundary_primary_bridge, mean_interaction) = {safe_corr(df['boundary_primary_bridge'], df['mean_interaction']):.4f}"
    )
    lines.append(
        f"  corr(hybrid_bridge_v2, mean_interaction)        = {safe_corr(df['hybrid_bridge_v2'], df['mean_interaction']):.4f}"
    )
    lines.append(
        f"  corr(topological_betweenness, mean_interaction) = {safe_corr(df['betweenness_supra_mean_refined'], df['mean_interaction']):.4f}"
    )
    lines.append(
        f"  corr(boundary_primary_bridge, time_index)       = {safe_corr(df['boundary_primary_bridge'], df['time_index']):.4f}"
    )
    lines.append(
        f"  corr(hybrid_bridge_v2, time_index)              = {safe_corr(df['hybrid_bridge_v2'], df['time_index']):.4f}"
    )
    lines.append(
        f"  corr(core_score, mean_block_field)              = {safe_corr(df['core_score'], df['mean_block_field']):.4f}"
    )
    lines.append(
        f"  corr(abs_discordance, mean_vuln_field)          = {safe_corr(df['abs_discordance'], df['mean_vuln_field']):.4f}"
    )
    lines.append("")

    return "\n".join(lines)


# =========================================================
# 8) Plotting
# =========================================================

def plot_single_metric_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path,
                           config: B26Config, cmap: str = "viridis",
                           annotate_top: bool = False, top_metric: str = None, ascending: bool = False):
    """
    Plot a single node-level metric map.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["center_x"], df["center_y"], c=df[value_col], s=config.node_size, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if annotate_top and top_metric is not None:
        top_df = top_nodes_by_metric(df, top_metric, config.annotate_top_n, ascending=ascending)
        label_ids = set(top_df["placement_id"].tolist())

        for _, row in df.iterrows():
            if int(row["placement_id"]) in label_ids:
                ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_comparison(df: pd.DataFrame, out_path: Path, config: B26Config):
    """
    Compare topological betweenness, boundary-primary bridge, and hybrid bridge v2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    sc = ax.scatter(
        df["center_x"], df["center_y"],
        c=df["betweenness_supra_mean_refined"],
        s=config.node_size,
        cmap="magma"
    )
    ax.set_title("Topological bridge")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(
        df["center_x"], df["center_y"],
        c=df["boundary_primary_bridge"],
        s=config.node_size,
        cmap="cividis"
    )
    ax.set_title("Boundary-primary bridge")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    sc = ax.scatter(
        df["center_x"], df["center_y"],
        c=df["hybrid_bridge_v2"],
        s=config.node_size,
        cmap="magma"
    )
    ax.set_title("Hybrid bridge v2")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_top_nodes_summary(df: pd.DataFrame, out_path: Path, config: B26Config):
    """
    Create a summary figure for the main node-role categories.
    """
    top_core = top_nodes_by_metric(df, "core_score", config.top_k_report, ascending=False)
    top_boundary = top_nodes_by_metric(df, "boundary_primary_bridge", config.top_k_report, ascending=False)
    top_hybrid = top_nodes_by_metric(df, "hybrid_bridge_v2", config.top_k_report, ascending=False)
    top_sp_dom = top_nodes_by_metric(df, "signed_discordance", config.top_k_report, ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.barh(top_core["placement_id"].astype(str), top_core["core_score"])
    ax.set_title("Top core nodes")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.barh(top_boundary["placement_id"].astype(str), top_boundary["boundary_primary_bridge"])
    ax.set_title("Top boundary-primary bridge nodes")
    ax.invert_yaxis()

    ax = axes[1, 0]
    ax.barh(top_hybrid["placement_id"].astype(str), top_hybrid["hybrid_bridge_v2"])
    ax.set_title("Top hybrid bridge v2 nodes")
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.barh(top_sp_dom["placement_id"].astype(str), top_sp_dom["signed_discordance"])
    ax.set_title("Top spontaneous-dominant nodes")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 9) Main
# =========================================================

if __name__ == "__main__":
    config = B26Config()

    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "codeB1_outputs"
    output_dir = script_dir / "codeB26_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    node_table = load_b1_node_table(input_dir)

    df, artifacts = build_b26_role_table(node_table, config)

    df.to_csv(output_dir / "b26_node_metrics.csv", index=False)

    node_ids = df["placement_id"].astype(int).tolist()
    pd.DataFrame(
        artifacts["A_sp_refined"], index=node_ids, columns=node_ids
    ).to_csv(output_dir / "b26_spont_adjacency_refined.csv")

    pd.DataFrame(
        artifacts["A_pc_refined"], index=node_ids, columns=node_ids
    ).to_csv(output_dir / "b26_paced_adjacency_refined.csv")

    supra_index = [f"s_{i}" for i in node_ids] + [f"p_{i}" for i in node_ids]
    pd.DataFrame(
        artifacts["A_supra_refined"], index=supra_index, columns=supra_index
    ).to_csv(output_dir / "b26_supra_adjacency_refined.csv")

    summary_text = build_b26_summary(df, artifacts, config)
    with open(output_dir / "b26_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    plot_single_metric_map(
        df=df,
        value_col="core_score",
        title="B26 core score map",
        out_path=output_dir / "b26_core_map.png",
        config=config,
        cmap="viridis",
        annotate_top=True,
        top_metric="core_score",
        ascending=False
    )

    plot_single_metric_map(
        df=df,
        value_col="boundary_primary_bridge",
        title="B26 boundary-primary bridge map",
        out_path=output_dir / "b26_boundary_primary_bridge_map.png",
        config=config,
        cmap="cividis",
        annotate_top=True,
        top_metric="boundary_primary_bridge",
        ascending=False
    )

    plot_single_metric_map(
        df=df,
        value_col="hybrid_bridge_v2",
        title="B26 hybrid bridge v2 map",
        out_path=output_dir / "b26_hybrid_bridge_v2_map.png",
        config=config,
        cmap="magma",
        annotate_top=True,
        top_metric="hybrid_bridge_v2",
        ascending=False
    )

    plot_single_metric_map(
        df=df,
        value_col="recency_weight",
        title="B26 recency weight map",
        out_path=output_dir / "b26_recency_map.png",
        config=config,
        cmap="plasma",
        annotate_top=False
    )

    plot_bridge_comparison(
        df=df,
        out_path=output_dir / "b26_bridge_comparison.png",
        config=config
    )

    plot_top_nodes_summary(
        df=df,
        out_path=output_dir / "b26_top_nodes_summary.png",
        config=config
    )

    print("\nCodeB26 outputs saved to:")
    print(output_dir)

    print("\nKey files:")
    print(" - b26_node_metrics.csv")
    print(" - b26_spont_adjacency_refined.csv")
    print(" - b26_paced_adjacency_refined.csv")
    print(" - b26_supra_adjacency_refined.csv")
    print(" - b26_summary.txt")
    print(" - b26_core_map.png")
    print(" - b26_boundary_primary_bridge_map.png")
    print(" - b26_hybrid_bridge_v2_map.png")
    print(" - b26_bridge_comparison.png")
    print(" - b26_top_nodes_summary.png")
    print(" - b26_recency_map.png")

    print("\nSummary:")
    print(summary_text)