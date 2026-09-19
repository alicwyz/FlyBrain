"""Turn raw Codex neuron/connection CSVs into a GPU-friendly packed graph.

Reads three files from Data/raw (see download_codex.py, which fetches
exactly these):
  classification.csv        root_id, flow, super_class, class, sub_class,
                             hemilineage, side, nerve
  coordinates.csv            root_id, position ("x, y, z" in nanometers,
                             zero or more marked points per neuron), supervoxel_id
  connections_princeton.csv pre_root_id, post_root_id, neuropil, syn_count, nt_type

Output (written to paths.processed_dir):
  neurons.csv   one row per surviving neuron, in shader index order. Each
                neuron's incoming edges are baked in as esrc0..esrc{K-1} /
                ew0..ew{K-1} columns (source neuron index + signed weight,
                padded/capped to K -- see config.yaml's `gather` section).
  meta.json     dimensions, label maps, gather stats
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from common import find_column, load_config, require_column, resolve_path


def parse_positions(series: pd.Series) -> pd.DataFrame:
    """Parse Codex's freeform 'position' string ("(x, y, z)" or "x_y_z" or
    "x y z", nanometers) by pulling out the first three numbers -- robust
    to whichever delimiter/bracket style the export uses."""
    xs, ys, zs = [], [], []
    for val in series.astype(str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", val)
        if len(nums) >= 3:
            xs.append(float(nums[0])); ys.append(float(nums[1])); zs.append(float(nums[2]))
        else:
            xs.append(np.nan); ys.append(np.nan); zs.append(np.nan)
    return pd.DataFrame({"pos_x": xs, "pos_y": ys, "pos_z": zs})


def load_raw(raw_dir, position_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_csv = raw_dir / "classification.csv"
    coordinates_csv = raw_dir / "coordinates.csv"
    conn_csv = raw_dir / "connections_princeton.csv"
    missing = [p.name for p in (classification_csv, coordinates_csv, conn_csv) if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing} in {raw_dir}.\nRun download_codex.py first.")

    classification = pd.read_csv(classification_csv, low_memory=False)
    coordinates = pd.read_csv(coordinates_csv, low_memory=False)
    connections = pd.read_csv(conn_csv, low_memory=False)

    cls_id_col = require_column(classification.columns, "root_id", "pt_root_id", "id")
    classification = classification.rename(columns={cls_id_col: "root_id"})
    classification["root_id"] = classification["root_id"].astype(str)

    coord_id_col = require_column(coordinates.columns, "root_id", "pt_root_id", "id")
    pos_col = require_column(coordinates.columns, "position", "coordinates", "coord")
    parsed = parse_positions(coordinates[pos_col])
    parsed["root_id"] = coordinates[coord_id_col].astype(str)
    # A neuron can have zero, one, or several marked points -- average them
    # into one representative position per neuron.
    coord_agg = parsed.groupby("root_id", as_index=False)[["pos_x", "pos_y", "pos_z"]].mean()

    neurons = classification.merge(coord_agg, on="root_id", how="left")

    if position_cfg.get("center", True):
        for c in ("pos_x", "pos_y", "pos_z"):
            neurons[c] = neurons[c] - neurons[c].mean()

    if position_cfg.get("normalize", True):
        # Single uniform scale factor (not per-axis) so the brain's true
        # proportions are preserved -- POPs/shaders conventionally expect
        # roughly [-1, 1], so fit the largest extent (across all three
        # axes) to normalize_to and let the other two axes end up smaller.
        target = float(position_cfg.get("normalize_to", 1.0))
        max_abs = float(np.nanmax(np.abs(neurons[["pos_x", "pos_y", "pos_z"]].values)))
        factor = (target / max_abs) if max_abs > 0 else 1.0
        for c in ("pos_x", "pos_y", "pos_z"):
            neurons[c] = neurons[c] * factor
    else:
        scale = float(position_cfg.get("scale", 1.0))
        for c in ("pos_x", "pos_y", "pos_z"):
            neurons[c] = neurons[c] * scale

    n_missing = neurons["pos_x"].isna().sum()
    if n_missing:
        print(f"Note: {n_missing}/{len(neurons)} neurons have no marked "
              f"coordinate in coordinates.csv -- they'll get a small random "
              f"jitter position instead of a real one.", file=sys.stderr)

    return neurons, connections


def normalize_neurons(neurons: pd.DataFrame) -> pd.DataFrame:
    cols = neurons.columns
    id_col = require_column(cols, "root_id", "pt_root_id", "id")
    x_col = find_column(cols, "pos_x", "position_x", "soma_x", "nucleus_x", "x")
    y_col = find_column(cols, "pos_y", "position_y", "soma_y", "nucleus_y", "y")
    z_col = find_column(cols, "pos_z", "position_z", "soma_z", "nucleus_z", "z")
    side_col = find_column(cols, "side")
    super_col = find_column(cols, "super_class", "superclass")
    class_col = find_column(cols, "class", "cell_class")

    out = pd.DataFrame()
    out["root_id"] = neurons[id_col].astype(str)
    out["pos_x"] = neurons[x_col].astype(float) if x_col else np.nan
    out["pos_y"] = neurons[y_col].astype(float) if y_col else np.nan
    out["pos_z"] = neurons[z_col].astype(float) if z_col else np.nan
    out["side"] = neurons[side_col].astype(str) if side_col else "C"
    out["super_class"] = neurons[super_col].astype(str).str.lower() if super_col else "unknown"
    out["class"] = neurons[class_col].astype(str) if class_col else out["super_class"]

    # Fill only the rows actually missing a position (neurons with no
    # marked coordinate) with jitter around the rest of the cloud, instead
    # of nuking every position when the columns exist but some rows don't.
    n_missing = out["pos_x"].isna().sum()
    if n_missing:
        if out["pos_x"].notna().any():
            spread = float(np.nanstd(out[["pos_x", "pos_y", "pos_z"]].values)) or 100.0
        else:
            print("WARNING: no soma position data found at all -- filling with "
                  "a fully random layout. Visualization will still work but "
                  "won't reflect real anatomy. Check the neuron CSV's actual "
                  "column names if this is unexpected.", file=sys.stderr)
            print("Neuron CSV columns:", list(cols), file=sys.stderr)
            spread = 100.0
        rng = np.random.default_rng(0)
        mask = out["pos_x"].isna()
        for c in ("pos_x", "pos_y", "pos_z"):
            out.loc[mask, c] = rng.normal(0, spread, int(n_missing))
    return out


def normalize_connections(connections: pd.DataFrame) -> pd.DataFrame:
    cols = connections.columns
    pre_col = require_column(cols, "pre_root_id", "pre_pt_root_id", "pre_id", "source")
    post_col = require_column(cols, "post_root_id", "post_pt_root_id", "post_id", "target")
    syn_col = find_column(cols, "syn_count", "weight", "n_synapses", "count")
    nt_col = find_column(cols, "nt_type", "predicted_nt", "neurotransmitter")
    neuropil_col = find_column(cols, "neuropil", "region")

    out = pd.DataFrame()
    out["pre_root_id"] = connections[pre_col].astype(str)
    out["post_root_id"] = connections[post_col].astype(str)
    out["syn_count"] = connections[syn_col].astype(float) if syn_col else 1.0
    out["nt_type"] = connections[nt_col].astype(str).str.lower() if nt_col else "unknown"
    out["neuropil"] = connections[neuropil_col].astype(str) if neuropil_col else ""
    return out


def encode_categorical(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Stable string -> small-int encoding, POP attributes are numeric only.
    Returns (ids, labels) where labels[i] is the string for id i -- used so
    TouchDesigner-side node (a Select POP / Math POP / CHOP Execute) can
    pick input/output neurons interactively by id, without regenerating
    the CSV. See docs/touchdesigner_network_guide.md."""
    clean = series.fillna("unknown").astype(str)
    labels = sorted(clean.unique())
    label_to_id = {label: i for i, label in enumerate(labels)}
    ids = clean.map(label_to_id).values.astype(np.int32)
    return ids, labels


def build(cfg: dict):
    raw_dir = resolve_path(cfg, "raw_dir")
    max_in_degree_cap = int(cfg.get("gather", {}).get("max_in_degree", 64))
    processed_dir = resolve_path(cfg, "processed_dir", max_in_degree=max_in_degree_cap)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_neurons, raw_connections = load_raw(raw_dir, cfg.get("position", {}))
    neurons = normalize_neurons(raw_neurons)
    connections = normalize_connections(raw_connections)

    sub = cfg["subset"]
    if sub.get("neuropil_include"):
        allowed_pils = set(sub["neuropil_include"])
        touched_ids = set(connections.loc[connections["neuropil"].isin(allowed_pils), "pre_root_id"]) | \
                      set(connections.loc[connections["neuropil"].isin(allowed_pils), "post_root_id"])
        if not touched_ids:
            sys.exit(f"No connections found in neuropils {sorted(allowed_pils)} -- "
                      f"check the names against Codex's neuropil abbreviations "
                      f"(see the Neuropils app on codex.flywire.ai).")
        neurons = neurons[neurons["root_id"].isin(touched_ids)]
    if sub.get("super_class_include"):
        neurons = neurons[neurons["super_class"].isin(sub["super_class_include"])]
    if sub.get("max_neurons") and len(neurons) > sub["max_neurons"]:
        neurons = neurons.sample(n=sub["max_neurons"], random_state=0)
    neurons = neurons.reset_index(drop=True)
    if len(neurons) == 0:
        sys.exit("No neurons survived the subset filters in config.yaml -- widen them.")

    id_to_idx = {rid: i for i, rid in enumerate(neurons["root_id"])}
    connections = connections[
        connections["pre_root_id"].isin(id_to_idx) & connections["post_root_id"].isin(id_to_idx)
    ].copy()
    if sub.get("min_synapse_count"):
        connections = connections[connections["syn_count"] >= sub["min_synapse_count"]]

    nt_sign_cfg = {k.lower(): v for k, v in cfg["neurotransmitter_sign"].items()}
    unknown_sign = nt_sign_cfg.get("unknown", 0.5)
    sign = connections["nt_type"].map(nt_sign_cfg).fillna(unknown_sign).astype(float)

    w_cfg = cfg["weight"]
    magnitude = connections["syn_count"].astype(float)
    if w_cfg.get("log_compress", True):
        magnitude = np.log1p(magnitude)
    weight = sign.values * magnitude.values * float(w_cfg.get("global_scale", 1.0))

    pre_idx = connections["pre_root_id"].map(id_to_idx).values.astype(np.int32)
    post_idx = connections["post_root_id"].map(id_to_idx).values.astype(np.int32)

    # Sort by target (post) neuron so each neuron's incoming edges are
    # contiguous -- needed to group edges per neuron for the gather step
    # below.
    order = np.argsort(post_idx, kind="stable")
    pre_idx = pre_idx[order]
    post_idx = post_idx[order]
    weight = weight.astype(np.float32)[order]

    n = len(neurons)
    neuron_count = np.bincount(post_idx, minlength=n).astype(np.int32)
    neuron_offset = np.zeros(n, dtype=np.int32)
    np.cumsum(neuron_count[:-1], out=neuron_offset[1:]) if n > 1 else None

    # is_input/is_output/channel assignment lives in TouchDesigner (a Select
    # POP / Math POP keyed off super_class_id/class_id below) instead of
    # being baked into the CSV -- change which neurons are taps live,
    # without rerunning this script. See docs/touchdesigner_network_guide.md.
    super_class_id, super_class_labels = encode_categorical(neurons["super_class"])
    class_id, class_labels = encode_categorical(neurons["class"])
    # Per-neuron sample slot within its OWN super_class group (0-indexed,
    # stable/arbitrary-but-deterministic order). class_id only gives ~10-30
    # distinct values per super_class (many neurons share one), so driving
    # by class_id broadcasts the same input value to a whole class at once.
    # super_local_idx gives every individual neuron its own index instead,
    # so each can read a distinct SAMPLE from a single-channel driving
    # buffer (e.g. one audio sample per neuron) rather than sharing a
    # per-class CHANNEL value. See docs/touchdesigner_network_guide.md.
    super_local_idx = pd.Series(np.arange(n), index=neurons.index).groupby(
        pd.Series(super_class_id, index=neurons.index)
    ).cumcount().values.astype(np.int32)

    # Bake each neuron's own incoming edges into esrcK/ewK padded columns
    # on neurons.csv -> fixed-size POP point-attribute arrays. That's what
    # the recurrent GLSL update gathers over.
    #
    # Vectorized (no per-neuron Python loop -- this needs to hold up at
    # tens to hundreds of thousands of neurons): re-sort edges by
    # (post_idx, -|weight|) so within each neuron's contiguous block, the
    # strongest connections come first, then compute each edge's rank
    # within its block via a single subtraction against the repeated
    # offsets, and scatter rank < K straight into a dense (n, K) array.
    n_truncated = int((neuron_count > max_in_degree_cap).sum())

    strength_order = np.lexsort((-np.abs(weight), post_idx))
    pre_by_strength = pre_idx[strength_order]
    post_by_strength = post_idx[strength_order]
    w_by_strength = weight[strength_order]
    rank_in_group = np.arange(len(post_by_strength)) - np.repeat(neuron_offset, neuron_count)

    keep = rank_in_group < max_in_degree_cap
    rows = post_by_strength[keep]
    cols = rank_in_group[keep]
    esrc_padded = np.full((n, max_in_degree_cap), -1, dtype=np.int32)
    ew_padded = np.zeros((n, max_in_degree_cap), dtype=np.float32)
    esrc_padded[rows, cols] = pre_by_strength[keep]
    ew_padded[rows, cols] = w_by_strength[keep]

    neurons_out = neurons.copy()
    neurons_out.insert(0, "idx", np.arange(n))
    neurons_out["super_class_id"] = super_class_id
    neurons_out["class_id"] = class_id
    neurons_out["super_local_idx"] = super_local_idx
    # Simulation state columns TD needs but Codex doesn't provide -- initial
    # values for the leaky-rate update. is_output as a real backing column
    # (rather than a GLSL POP "attr" block created on demand) avoids a TD
    # gotcha: once a Feedback POP loop has run and its target has been
    # writing an attribute for a while, that attribute becomes a genuine
    # *inherited* input attribute on the next cook -- declaring it again via
    # an "attr" block at that point fails to compile. Backing it with a
    # column from the start sidesteps the timing-dependent trap.
    neurons_out["state"] = 0.0
    neurons_out["is_output"] = 0.0
    neurons_out["bias"] = 0.0
    neurons_out["inputgain"] = 1.0
    for k in range(max_in_degree_cap):
        neurons_out[f"esrc{k}"] = esrc_padded[:, k]
        neurons_out[f"ew{k}"] = ew_padded[:, k]
    neurons_out.to_csv(processed_dir / "neurons.csv", index=False)

    meta = {
        "n_neurons": int(n),
        "n_edges": int(len(pre_idx)),
        "super_class_labels": super_class_labels,
        "class_labels": class_labels,
        "avg_in_degree": float(neuron_count.mean()),
        "max_in_degree": int(neuron_count.max()) if n else 0,
        "gather_max_in_degree_cap": max_in_degree_cap,
        "n_neurons_truncated_by_cap": n_truncated,
    }
    with open(processed_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nWrote packed graph to {processed_dir}")
    return meta


def main():
    cfg = load_config()
    build(cfg)


if __name__ == "__main__":
    main()
