# codeB2.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dataclasses import dataclass
from pathlib import Path


print("RUNNING CODEB2: multiplex node-role analysis")


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class B2Config:
    # Numerical stability for inverse-distance conversion
    eps_distance: float = 1e-8

    # Number of top nodes to report in text summaries and plots
    top_k_report: int = 8

    # Scatter plot style
    node_size: int = 160
    annotate_top_n: int = 8

    # Figure DPI
    dpi: int = 220


# =========================================================
# 2) Loading functions
# =========================================================

def load_b1_outputs(input_dir: Path):
    """
    Load CodeB1 outputs:
    - node table
    - spontaneous adjacency
    - paced adjacency
    - supra adjacency
    """
    node_table = pd.read_csv(input_dir / "b1_node_table.csv")

    A_sp = pd.read_csv(input_dir / "b1_spont_adjacency.csv", index_col=0).to_numpy(dtype=float)
    A_pc = pd.read_csv(input_dir / "b1_paced_adjacency.csv", index_col=0).to_numpy(dtype=float)
    A_supra = pd.read_csv(input_dir / "b1_supra_adjacency.csv", index_col=0).to_numpy(dtype=float)

    return node_table, A_sp, A_pc, A_supra


# =========================================================
# 3) Graph helpers
# =========================================================

def build_weighted_graph_from_adjacency(W: np.ndarray):
    """
    Build an undirected weighted NetworkX graph from an adjacency matrix.
    """
    return nx.from_numpy_array(W)


def add_distance_attribute(G: nx.Graph, eps: float = 1e-8):
    """
    Convert edge weights (similarity) into edge distances for shortest-path metrics.

    Higher similarity should correspond to shorter distance, so we use:
        distance = 1 / (weight + eps)
    """
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.0)
        data["distance"] = 1.0 / (w + eps)


def weighted_node_strength(G: nx.Graph):
    """
    Compute weighted degree (node strength) for all nodes.
    """
    return dict(G.degree(weight="weight"))


def unweighted_node_degree(G: nx.Graph):
    """
    Compute unweighted degree for all nodes.
    """
    return dict(G.degree())


def weighted_clustering(G: nx.Graph):
    """
    Compute weighted clustering coefficient for all nodes.
    """
    return nx.clustering(G, weight="weight")


def supra_betweenness_from_similarity_graph(A_supra: np.ndarray, eps: float = 1e-8):
    """
    Compute betweenness centrality on the multiplex supra-graph.

    Since adjacency weights represent similarity, shortest paths should be computed
    using inverse similarity as distance.
    """
    G = build_weighted_graph_from_adjacency(A_supra)
    add_distance_attribute(G, eps=eps)
    bc = nx.betweenness_centrality(G, weight="distance", normalized=True)
    return bc, G


# =========================================================
# 4) Metric construction
# =========================================================

def build_layer_metric_tables(node_table: pd.DataFrame, A_sp: np.ndarray, A_pc: np.ndarray, config: B2Config):
    """
    Compute layer-wise graph metrics for spontaneous and paced layers.
    """
    G_sp = build_weighted_graph_from_adjacency(A_sp)
    G_pc = build_weighted_graph_from_adjacency(A_pc)

    strength_sp = weighted_node_strength(G_sp)
    strength_pc = weighted_node_strength(G_pc)

    degree_sp = unweighted_node_degree(G_sp)
    degree_pc = unweighted_node_degree(G_pc)

    clustering_sp = weighted_clustering(G_sp)
    clustering_pc = weighted_clustering(G_pc)

    # Optional k-core numbers for assignment alignment
    kcore_sp = nx.core_number(nx.Graph(G_sp))
    kcore_pc = nx.core_number(nx.Graph(G_pc))

    spont_df = pd.DataFrame({
        "placement_id": node_table["placement_id"].astype(int),
        "time_index": node_table["time_index"].astype(int),
        "time_min": node_table["time_min"].astype(float),
        "center_x": node_table["center_x"].astype(float),
        "center_y": node_table["center_y"].astype(float),

        "u_sp": node_table["u_sp"].astype(float),
        "SpontScore": node_table["SpontScore"].astype(float),

        "strength_sp": [strength_sp[i] for i in range(len(node_table))],
        "degree_sp": [degree_sp[i] for i in range(len(node_table))],
        "clustering_sp": [clustering_sp[i] for i in range(len(node_table))],
        "kcore_sp": [kcore_sp[i] for i in range(len(node_table))],

        "STD_p": node_table["STD_p"].astype(float),
        "ICS_p": node_table["ICS_p"].astype(float),
        "Disp_p": node_table["Disp_p"].astype(float),
        "FI_burden": node_table["FI_burden"].astype(float),
        "LV_burden": node_table["LV_burden"].astype(float),

        "mean_block_field": node_table["mean_block_field"].astype(float),
        "mean_lv_field": node_table["mean_lv_field"].astype(float),
        "mean_vuln_field": node_table["mean_vuln_field"].astype(float),
        "mean_interaction": node_table["mean_interaction"].astype(float),
    })

    paced_df = pd.DataFrame({
        "placement_id": node_table["placement_id"].astype(int),
        "time_index": node_table["time_index"].astype(int),
        "time_min": node_table["time_min"].astype(float),
        "center_x": node_table["center_x"].astype(float),
        "center_y": node_table["center_y"].astype(float),

        "u_pc": node_table["u_pc"].astype(float),
        "PacedScore": node_table["PacedScore"].astype(float),

        "strength_pc": [strength_pc[i] for i in range(len(node_table))],
        "degree_pc": [degree_pc[i] for i in range(len(node_table))],
        "clustering_pc": [clustering_pc[i] for i in range(len(node_table))],
        "kcore_pc": [kcore_pc[i] for i in range(len(node_table))],

        "STD_paced": node_table["STD_paced"].astype(float),
        "ICS_paced": node_table["ICS_paced"].astype(float),
        "LDH_paced": node_table["LDH_paced"].astype(float),
        "LV_burden_paced": node_table["LV_burden_paced"].astype(float),

        "mean_block_field": node_table["mean_block_field"].astype(float),
        "mean_lv_field": node_table["mean_lv_field"].astype(float),
        "mean_vuln_field": node_table["mean_vuln_field"].astype(float),
        "mean_interaction": node_table["mean_interaction"].astype(float),
    })

    return spont_df, paced_df, G_sp, G_pc


def build_combined_role_table(node_table: pd.DataFrame, spont_df: pd.DataFrame, paced_df: pd.DataFrame,
                              A_supra: np.ndarray, config: B2Config):
    """
    Build a combined node-role table using:
    - spontaneous layer metrics
    - paced layer metrics
    - supra-graph betweenness
    - core score
    - signed discordance
    """
    n = len(node_table)

    bc_supra, G_supra = supra_betweenness_from_similarity_graph(A_supra, eps=config.eps_distance)

    # Layer-specific betweenness values from the supra-graph
    bet_sp_layer = np.array([bc_supra[i] for i in range(n)], dtype=float)
    bet_pc_layer = np.array([bc_supra[n + i] for i in range(n)], dtype=float)

    # Combined bridge score per physical patch
    bet_patch_mean = 0.5 * (bet_sp_layer + bet_pc_layer)
    bet_patch_max = np.maximum(bet_sp_layer, bet_pc_layer)

    combined_df = pd.DataFrame({
        "placement_id": node_table["placement_id"].astype(int),
        "time_index": node_table["time_index"].astype(int),
        "time_min": node_table["time_min"].astype(float),
        "center_x": node_table["center_x"].astype(float),
        "center_y": node_table["center_y"].astype(float),
        "angle_deg": node_table["angle_deg"].astype(float),

        "u_sp": node_table["u_sp"].astype(float),
        "u_pc": node_table["u_pc"].astype(float),

        "SpontScore": node_table["SpontScore"].astype(float),
        "PacedScore": node_table["PacedScore"].astype(float),

        "strength_sp": spont_df["strength_sp"].astype(float),
        "strength_pc": paced_df["strength_pc"].astype(float),

        "degree_sp": spont_df["degree_sp"].astype(float),
        "degree_pc": paced_df["degree_pc"].astype(float),

        "clustering_sp": spont_df["clustering_sp"].astype(float),
        "clustering_pc": paced_df["clustering_pc"].astype(float),

        "kcore_sp": spont_df["kcore_sp"].astype(int),
        "kcore_pc": paced_df["kcore_pc"].astype(int),

        "betweenness_sp_layer": bet_sp_layer,
        "betweenness_pc_layer": bet_pc_layer,
        "betweenness_supra_mean": bet_patch_mean,
        "betweenness_supra_max": bet_patch_max,

        "core_score": np.minimum(node_table["u_sp"].to_numpy(), node_table["u_pc"].to_numpy()),
        "signed_discordance": node_table["u_sp"].to_numpy() - node_table["u_pc"].to_numpy(),
        "abs_discordance": np.abs(node_table["u_sp"].to_numpy() - node_table["u_pc"].to_numpy()),

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

    return combined_df, G_supra


# =========================================================
# 5) Summary helpers
# =========================================================

def top_nodes_by_metric(df: pd.DataFrame, metric_col: str, top_k: int, ascending: bool = False):
    """
    Return the top-k nodes sorted by a metric.
    """
    return df.sort_values(metric_col, ascending=ascending).head(top_k).copy()


def safe_corr(x, y):
    """
    Compute Pearson correlation safely.
    Returns NaN if one vector is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_text_summary(combined_df: pd.DataFrame, config: B2Config):
    """
    Build a human-readable text summary of the main B2 findings.
    """
    top_core = top_nodes_by_metric(combined_df, "core_score", config.top_k_report, ascending=False)
    top_bridge = top_nodes_by_metric(combined_df, "betweenness_supra_mean", config.top_k_report, ascending=False)
    top_sp_dominant = top_nodes_by_metric(combined_df, "signed_discordance", config.top_k_report, ascending=False)
    top_pc_dominant = top_nodes_by_metric(combined_df, "signed_discordance", config.top_k_report, ascending=True)

    lines = []
    lines.append("CODEB2 NODE-ROLE SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Top core nodes (high in both spontaneous and paced):")
    for _, row in top_core.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: core={row['core_score']:.4f}, "
            f"u_sp={row['u_sp']:.4f}, u_pc={row['u_pc']:.4f}"
        )
    lines.append("")

    lines.append("Top bridge-like nodes (highest supra betweenness):")
    for _, row in top_bridge.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: bet={row['betweenness_supra_mean']:.4f}, "
            f"time_index={int(row['time_index'])}"
        )
    lines.append("")

    lines.append("Top spontaneous-dominant discordant nodes:")
    for _, row in top_sp_dominant.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: disc={row['signed_discordance']:.4f}, "
            f"u_sp={row['u_sp']:.4f}, u_pc={row['u_pc']:.4f}"
        )
    lines.append("")

    lines.append("Top paced-dominant discordant nodes:")
    for _, row in top_pc_dominant.iterrows():
        lines.append(
            f"  placement {int(row['placement_id'])}: disc={row['signed_discordance']:.4f}, "
            f"u_sp={row['u_sp']:.4f}, u_pc={row['u_pc']:.4f}"
        )
    lines.append("")

    # Simple correlations for interpretation
    lines.append("Correlation diagnostics:")
    lines.append(
        f"  corr(core_score, mean_block_field)       = {safe_corr(combined_df['core_score'], combined_df['mean_block_field']):.4f}"
    )
    lines.append(
        f"  corr(core_score, mean_vuln_field)        = {safe_corr(combined_df['core_score'], combined_df['mean_vuln_field']):.4f}"
    )
    lines.append(
        f"  corr(betweenness, mean_interaction)      = {safe_corr(combined_df['betweenness_supra_mean'], combined_df['mean_interaction']):.4f}"
    )
    lines.append(
        f"  corr(abs_discordance, mean_vuln_field)   = {safe_corr(combined_df['abs_discordance'], combined_df['mean_vuln_field']):.4f}"
    )
    lines.append(
        f"  corr(abs_discordance, time_index)        = {safe_corr(combined_df['abs_discordance'], combined_df['time_index']):.4f}"
    )
    lines.append(
        f"  corr(betweenness, time_index)            = {safe_corr(combined_df['betweenness_supra_mean'], combined_df['time_index']):.4f}"
    )
    lines.append("")

    return "\n".join(lines)


# =========================================================
# 6) Plotting functions
# =========================================================

def plot_two_layer_metric_maps(df: pd.DataFrame, col_sp: str, col_pc: str,
                               title_sp: str, title_pc: str, out_path: Path, config: B2Config):
    """
    Plot a side-by-side map for spontaneous and paced node metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sc = ax.scatter(df["center_x"], df["center_y"], c=df[col_sp], s=config.node_size)
    ax.set_title(title_sp)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(df["center_x"], df["center_y"], c=df[col_pc], s=config.node_size)
    ax.set_title(title_pc)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_metric_map(df: pd.DataFrame, value_col: str, title: str, out_path: Path,
                           config: B2Config, cmap: str = "viridis", annotate_top: bool = False,
                           top_metric: str = None, ascending: bool = False):
    """
    Plot a single-node metric map over the 2D domain.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["center_x"], df["center_y"], c=df[value_col], s=config.node_size, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if annotate_top and top_metric is not None:
        top_df = top_nodes_by_metric(df, top_metric, config.annotate_top_n, ascending=ascending)
        top_ids = set(top_df["placement_id"].tolist())
        for _, row in df.iterrows():
            if int(row["placement_id"]) in top_ids:
                ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_signed_discordance_map(df: pd.DataFrame, out_path: Path, config: B2Config):
    """
    Plot signed discordance using a diverging colormap centered around zero.
    """
    values = df["signed_discordance"].to_numpy()
    vmax = float(np.max(np.abs(values))) if len(values) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        df["center_x"], df["center_y"],
        c=df["signed_discordance"],
        s=config.node_size,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax
    )
    ax.set_title("Signed discordance map (spontaneous - paced)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    top_pos = top_nodes_by_metric(df, "signed_discordance", config.annotate_top_n, ascending=False)
    top_neg = top_nodes_by_metric(df, "signed_discordance", config.annotate_top_n, ascending=True)
    label_ids = set(top_pos["placement_id"].tolist() + top_neg["placement_id"].tolist())

    for _, row in df.iterrows():
        if int(row["placement_id"]) in label_ids:
            ax.text(row["center_x"], row["center_y"], str(int(row["placement_id"])), fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_top_nodes_summary(df: pd.DataFrame, out_path: Path, config: B2Config):
    """
    Create a 2x2 summary figure with bar plots for the main node-role categories.
    """
    top_core = top_nodes_by_metric(df, "core_score", config.top_k_report, ascending=False)
    top_bridge = top_nodes_by_metric(df, "betweenness_supra_mean", config.top_k_report, ascending=False)
    top_sp_dominant = top_nodes_by_metric(df, "signed_discordance", config.top_k_report, ascending=False)
    top_pc_dominant = top_nodes_by_metric(df, "signed_discordance", config.top_k_report, ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.barh(top_core["placement_id"].astype(str), top_core["core_score"])
    ax.set_title("Top core nodes")
    ax.invert_yaxis()

    ax = axes[0, 1]
    ax.barh(top_bridge["placement_id"].astype(str), top_bridge["betweenness_supra_mean"])
    ax.set_title("Top bridge-like nodes")
    ax.invert_yaxis()

    ax = axes[1, 0]
    ax.barh(top_sp_dominant["placement_id"].astype(str), top_sp_dominant["signed_discordance"])
    ax.set_title("Top spontaneous-dominant discordant nodes")
    ax.invert_yaxis()

    ax = axes[1, 1]
    ax.barh(top_pc_dominant["placement_id"].astype(str), top_pc_dominant["signed_discordance"])
    ax.set_title("Top paced-dominant discordant nodes")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 7) Main
# =========================================================

if __name__ == "__main__":
    config = B2Config()

    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "codeB1_outputs"
    output_dir = script_dir / "codeB2_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load CodeB1 outputs
    # -------------------------
    node_table, A_sp, A_pc, A_supra = load_b1_outputs(input_dir)

    # -------------------------
    # Compute layer-wise metrics
    # -------------------------
    spont_df, paced_df, G_sp, G_pc = build_layer_metric_tables(
        node_table=node_table,
        A_sp=A_sp,
        A_pc=A_pc,
        config=config
    )

    # -------------------------
    # Build combined node-role table
    # -------------------------
    combined_df, G_supra = build_combined_role_table(
        node_table=node_table,
        spont_df=spont_df,
        paced_df=paced_df,
        A_supra=A_supra,
        config=config
    )

    # -------------------------
    # Save tables
    # -------------------------
    spont_df.to_csv(output_dir / "b2_node_metrics_spont.csv", index=False)
    paced_df.to_csv(output_dir / "b2_node_metrics_paced.csv", index=False)
    combined_df.to_csv(output_dir / "b2_node_metrics_combined.csv", index=False)

    # -------------------------
    # Build and save text summary
    # -------------------------
    summary_text = build_text_summary(combined_df, config)
    with open(output_dir / "b2_metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # -------------------------
    # Create report-ready plots
    # -------------------------
    plot_two_layer_metric_maps(
        df=combined_df,
        col_sp="strength_sp",
        col_pc="strength_pc",
        title_sp="Spontaneous node strength",
        title_pc="Paced node strength",
        out_path=output_dir / "b2_strength_maps.png",
        config=config
    )

    plot_two_layer_metric_maps(
        df=combined_df,
        col_sp="clustering_sp",
        col_pc="clustering_pc",
        title_sp="Spontaneous clustering coefficient",
        title_pc="Paced clustering coefficient",
        out_path=output_dir / "b2_clustering_maps.png",
        config=config
    )

    plot_single_metric_map(
        df=combined_df,
        value_col="core_score",
        title="Core score map",
        out_path=output_dir / "b2_core_map.png",
        config=config,
        cmap="viridis",
        annotate_top=True,
        top_metric="core_score",
        ascending=False
    )

    plot_signed_discordance_map(
        df=combined_df,
        out_path=output_dir / "b2_signed_discordance_map.png",
        config=config
    )

    plot_single_metric_map(
        df=combined_df,
        value_col="betweenness_supra_mean",
        title="Bridge-like node map (supra betweenness)",
        out_path=output_dir / "b2_bridge_map.png",
        config=config,
        cmap="magma",
        annotate_top=True,
        top_metric="betweenness_supra_mean",
        ascending=False
    )

    plot_top_nodes_summary(
        df=combined_df,
        out_path=output_dir / "b2_top_nodes_summary.png",
        config=config
    )

    # -------------------------
    # Console output
    # -------------------------
    print("\nCodeB2 outputs saved to:")
    print(output_dir)

    print("\nKey files:")
    print(" - b2_node_metrics_spont.csv")
    print(" - b2_node_metrics_paced.csv")
    print(" - b2_node_metrics_combined.csv")
    print(" - b2_metrics_summary.txt")
    print(" - b2_strength_maps.png")
    print(" - b2_clustering_maps.png")
    print(" - b2_core_map.png")
    print(" - b2_signed_discordance_map.png")
    print(" - b2_bridge_map.png")
    print(" - b2_top_nodes_summary.png")

    print("\nSummary:")
    print(summary_text)