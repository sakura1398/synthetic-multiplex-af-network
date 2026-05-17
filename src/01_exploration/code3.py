import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path


# =========================================================
# 1) Configuration
# =========================================================

@dataclass
class DatasetConfig:
    # Global domain
    domain_width: float = 100.0   # mm
    domain_height: float = 80.0   # mm

    # Patch geometry
    patch_rows: int = 4
    patch_cols: int = 4
    electrode_spacing: float = 4.0   # mm

    # Dataset size
    n_samples: int = 1
    n_placements: int = 10

    # Time step between placements
    minutes_per_step: float = 25.0

    # Trajectory settings
    trajectory_mode: str = "zigzag"
    placement_jitter: float = 2.0      # mm
    angle_jitter_deg: float = 20.0     # degrees

    # Hidden substrate
    n_sources: int = 2
    base_cv: float = 0.7               # mm/ms
    min_cv: float = 0.20               # mm/ms

    # Raw feature model parameters
    sigma_A: float = 18.0
    sigma_I: float = 15.0

    lat_noise_std: float = 2.0         # ms
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
    disp_c1: float = 0.5   # variance of FI
    disp_c2: float = 0.5   # variance of A

    # Future-use parameters
    sigma_smooth: float = 1.0
    sigma_kernel: float = 6.0
    low_voltage_threshold: float = 0.5

    # PatchScore weights (for later)
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25

    # DIC coefficients (for later)
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
    """
    Smooth radial field in [0, strength].
    """
    sigma = max(radius / 2.0, 1e-6)
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return strength * np.exp(-d2 / (2.0 * sigma ** 2))


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

    df["d_source1"] = d1
    df["d_source2"] = d2
    df["S1"] = S1
    df["S2"] = S2
    df["block_field"] = block_field
    df["lv_field"] = lv_field
    df["vuln_field"] = vuln_field
    df["v_eff"] = v_eff
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

            fi_var = np.var(fi_vals) if len(fi_vals) > 0 else 0.0
            a_var = np.var(a_vals) if len(a_vals) > 0 else 0.0

            disp[r, c] = config.disp_c1 * fi_var + config.disp_c2 * a_var

    return disp


def compute_local_derived_features(patch_df: pd.DataFrame,
                                   config: DatasetConfig) -> pd.DataFrame:
    df = patch_df.copy()
    df = df.sort_values(["row", "col"]).reset_index(drop=True)

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
# 9) Assemble one sample
# =========================================================

def assemble_derived_features_for_one_sample(sample_id: int,
                                             config: DatasetConfig,
                                             rng: np.random.Generator):
    substrate = generate_hidden_substrate(config, rng)
    placement_df = generate_trajectory(config, rng)
    local_patch_df = build_local_patch_geometry(config)

    electrode_tables = []

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

        patch_df["sample_id"] = sample_id
        patch_df["placement_id"] = int(placement["placement_id"])
        patch_df["time_index"] = int(placement["time_index"])
        patch_df["time_min"] = float(placement["time_min"])

        electrode_tables.append(patch_df)

    electrode_df = pd.concat(electrode_tables, ignore_index=True)

    placement_df = placement_df.copy()
    placement_df["sample_id"] = sample_id

    sample_row = {"sample_id": sample_id}
    sample_row.update(substrate)
    sample_df = pd.DataFrame([sample_row])

    return sample_df, placement_df, electrode_df


# =========================================================
# 10) CSV export
# =========================================================

def export_derived_feature_csv(output_dir: str,
                               config: DatasetConfig,
                               sample_df: pd.DataFrame,
                               placement_df: pd.DataFrame,
                               electrode_df: pd.DataFrame):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([asdict(config)]).to_csv(out / "config_table.csv", index=False)
    sample_df.to_csv(out / "sample_table.csv", index=False)
    placement_df.to_csv(out / "placement_table.csv", index=False)
    electrode_df.to_csv(out / "electrode_derived_table.csv", index=False)


# =========================================================
# 11) Basic checks
# =========================================================

def run_basic_checks(placement_df: pd.DataFrame, electrode_df: pd.DataFrame):
    print("\n--- Basic checks ---")

    counts = electrode_df.groupby("placement_id").size()
    print("\nElectrodes per placement:")
    print(counts)

    print("\nLAT summary:")
    print(electrode_df["LAT"].describe())

    print("\nAmplitude summary:")
    print(electrode_df["A"].describe())

    print("\nFI summary:")
    print(electrode_df["FI"].describe())

    print("\nDF summary:")
    print(electrode_df["DF"].describe())

    print("\nInteraction summary:")
    print(electrode_df["interaction"].describe())

    print("\nDisp summary:")
    print(electrode_df["Disp"].describe())

    print("\nLAT gradient magnitude summary:")
    print(electrode_df["LAT_grad_mag"].describe())

    print("\nTime range (min):", electrode_df["time_min"].min(), "to", electrode_df["time_min"].max())
    print("Global X range:", electrode_df["x_global"].min(), "to", electrode_df["x_global"].max())
    print("Global Y range:", electrode_df["y_global"].min(), "to", electrode_df["y_global"].max())


# =========================================================
# 12) Main
# =========================================================

if __name__ == "__main__":
    config = DatasetConfig(
        n_samples=1,
        n_placements=10,
        random_seed=42
    )

    rng = np.random.default_rng(config.random_seed)

    sample_df, placement_df, electrode_df = assemble_derived_features_for_one_sample(
        sample_id=0,
        config=config,
        rng=rng
    )

    export_derived_feature_csv(
        output_dir="manual_check_derived_features",
        config=config,
        sample_df=sample_df,
        placement_df=placement_df,
        electrode_df=electrode_df
    )

    print("Derived-feature files saved to: manual_check_derived_features/")

    print("\nSample table:")
    print(sample_df.head())

    print("\nPlacement table:")
    print(placement_df.head())

    print("\nElectrode derived table:")
    print(electrode_df.head())

    run_basic_checks(placement_df, electrode_df)