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

    # Trajectory settings
    trajectory_mode: str = "zigzag"
    placement_jitter: float = 2.0      # mm
    angle_jitter_deg: float = 20.0     # degrees

    # Hidden substrate parameters
    n_sources: int = 2
    base_cv: float = 0.7               # mm/ms, for later use

    # Free parameters kept here for future steps
    sigma_A: float = 18.0
    sigma_I: float = 15.0
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
# 2) Utilities
# =========================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def rotation_matrix(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


# =========================================================
# 3) Hidden substrate generation
# =========================================================

def generate_hidden_substrate(config: DatasetConfig, rng: np.random.Generator) -> dict:
    """
    Build one synthetic hidden substrate.
    This is NOT the prediction target. It is only the data generator.
    """
    W = config.domain_width
    H = config.domain_height

    margin = 12.0

    substrate = {}

    # Source-like generators
    for k in range(1, config.n_sources + 1):
        substrate[f"source{k}_x"] = rng.uniform(margin, W - margin)
        substrate[f"source{k}_y"] = rng.uniform(margin, H - margin)

    # Slow / block-like region
    substrate["block_x"] = rng.uniform(margin, W - margin)
    substrate["block_y"] = rng.uniform(margin, H - margin)
    substrate["block_radius"] = rng.uniform(8.0, 15.0)
    substrate["block_strength"] = rng.uniform(0.3, 0.7)

    # Low-voltage region
    substrate["lv_x"] = rng.uniform(margin, W - margin)
    substrate["lv_y"] = rng.uniform(margin, H - margin)
    substrate["lv_radius"] = rng.uniform(6.0, 14.0)
    substrate["lv_strength"] = rng.uniform(0.2, 0.6)

    # Vulnerability / dispersion field
    substrate["vuln_x"] = rng.uniform(margin, W - margin)
    substrate["vuln_y"] = rng.uniform(margin, H - margin)
    substrate["vuln_radius"] = rng.uniform(8.0, 18.0)
    substrate["vuln_strength"] = rng.uniform(0.2, 0.8)

    return substrate


# =========================================================
# 4) Local patch geometry
# =========================================================

def build_local_patch_geometry(config: DatasetConfig) -> pd.DataFrame:
    """
    Build a 4x4 local electrode grid centered at (0,0).
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


# =========================================================
# 5) Patch placement on global domain
# =========================================================

def place_patch_on_global_domain(local_patch_df: pd.DataFrame,
                                 center_x: float,
                                 center_y: float,
                                 angle_deg: float) -> pd.DataFrame:
    """
    Rotate and translate the local patch into global coordinates.
    """
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
    """
    Generate a semi-structured zigzag trajectory.
    """
    n = config.n_placements
    W = config.domain_width
    H = config.domain_height

    patch_w = (config.patch_cols - 1) * config.electrode_spacing
    patch_h = (config.patch_rows - 1) * config.electrode_spacing

    margin_x = patch_w / 2 + 2.0
    margin_y = patch_h / 2 + 2.0

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
            "center_x": float(x),
            "center_y": float(y),
            "angle_deg": float(angle),
        })

    return pd.DataFrame(records)


# =========================================================
# 7) Assemble one sample (geometry only)
# =========================================================

def assemble_geometry_only_for_one_sample(sample_id: int,
                                          config: DatasetConfig,
                                          rng: np.random.Generator):
    """
    Build:
    - sample_df
    - placement_df
    - electrode_geometry_df
    """
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

        patch_df["sample_id"] = sample_id
        patch_df["placement_id"] = int(placement["placement_id"])
        patch_df["time_index"] = int(placement["time_index"])

        electrode_tables.append(patch_df)

    electrode_df = pd.concat(electrode_tables, ignore_index=True)

    placement_df = placement_df.copy()
    placement_df["sample_id"] = sample_id

    sample_row = {"sample_id": sample_id}
    sample_row.update(substrate)
    sample_df = pd.DataFrame([sample_row])

    return sample_df, placement_df, electrode_df


# =========================================================
# 8) CSV export for manual validation
# =========================================================

def export_geometry_csv(output_dir: str,
                        config: DatasetConfig,
                        sample_df: pd.DataFrame,
                        placement_df: pd.DataFrame,
                        electrode_df: pd.DataFrame):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([asdict(config)]).to_csv(out / "config_table.csv", index=False)
    sample_df.to_csv(out / "sample_table.csv", index=False)
    placement_df.to_csv(out / "placement_table.csv", index=False)
    electrode_df.to_csv(out / "electrode_geometry_table.csv", index=False)


# =========================================================
# 9) Main
# =========================================================

if __name__ == "__main__":
    config = DatasetConfig(
        n_samples=1,
        n_placements=10,
        random_seed=42
    )

    rng = np.random.default_rng(config.random_seed)

    sample_df, placement_df, electrode_df = assemble_geometry_only_for_one_sample(
        sample_id=0,
        config=config,
        rng=rng
    )

    export_geometry_csv(
        output_dir="manual_check_geometry",
        config=config,
        sample_df=sample_df,
        placement_df=placement_df,
        electrode_df=electrode_df
    )

    print("Geometry-only files saved to: manual_check_geometry/")
    print("\nSample table:")
    print(sample_df.head())

    print("\nPlacement table:")
    print(placement_df.head())

    print("\nElectrode geometry table:")
    print(electrode_df.head())