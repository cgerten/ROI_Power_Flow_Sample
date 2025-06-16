#!/usr/bin/env python3
"""
Script for an approximate "AC Base Case + OTDF" approach using pandapower's makeBdc and makeLODF functions,
with worst-case contingency analysis based on intact AC branch flows (for lines >=110 kV).
Additionally, it aggregates contingency data (intact flow, post-contingency loadings, etc.)
globally across all network files.
This version implements batch processing, vectorized lookups for transformer and line data,
and caches topology-dependent matrices so that they are computed only once.
It also enforces a fixed line orientation across all hours and, if a branch’s orientation
flips compared to the base hour, it adjusts (flips) the corresponding sensitivity values.
"""

import os
import re
import time
import gc
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.makeLODF import makeLODF, makeOTDF
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from numpy.linalg import LinAlgError
from numba import njit, prange

# ------------------ Global Configuration ------------------
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PKL_DIR = "Model/timeseries"  # Folder with .p files
BUS_DATA_CSV = "Model/bus_data.csv"
LINE_DATA_CSV = "Model/pandapower_line_data.csv"

PF_SETTINGS = {
    "algorithm": "nr",
    "max_iteration": 100,
    "tolerance_mva": 5e-3,
    "init": "dc",
    "enforce_q_lims": True,
    "calculate_voltage_angles": True,
    "logging": False,
    "voltage_depend_loads": False,
    "v_debug": True
}

OVERLOAD_THRESHOLD = 100  # not in per unit (1.0 means 100% loading)
BATCH_SIZE = 2          # Adjust as needed
DEBUG = False

def log(msg):
    if DEBUG:
        print(msg)

# ------------------ Read Name Mappings ------------------
BUS_NAMES_DF = pd.read_csv(BUS_DATA_CSV)
LINE_DATA_DF = pd.read_csv(LINE_DATA_CSV)
BUS_IDX_TO_NAME = dict(zip(BUS_NAMES_DF["index"], BUS_NAMES_DF["bus_names"]))

def numerical_sort(value):
    nums = re.findall(r'\d+', value)
    return int(nums[0]) if nums else 0

def normalize_line_name(line_name):
    return ' '.join(line_name.strip().split())

# ------------------ Preprocessing ------------------
def preprocess_network(net):
    """
    Normalize line/trafo names and ensure consistent bus indexing in pandapower.
    """
    if "name_normalized" not in net.line.columns:
        net.line["name_normalized"] = net.line["name"].apply(normalize_line_name)
    if "name_normalized" not in net.trafo.columns:
        net.trafo["name_normalized"] = net.trafo["name"].apply(normalize_line_name)
    if "from_bus_orig" not in net.line.columns:
        net.line["from_bus_orig"] = net.line["from_bus"]
    if "to_bus_orig" not in net.line.columns:
        net.line["to_bus_orig"] = net.line["to_bus"]
    pp.create_continuous_bus_index(net, start=0, store_old_index=True)

def orient_lines_by_flow(net):
    """
    Runs a DC power flow, then re-orients each line if its flow is negative.
    Re-runs DC to update flows.
    """
    pp.runpp(net, **PF_SETTINGS)
    for i in net.line.index:
        p_from = net.res_line.at[i, "p_from_mw"]
        if p_from < 0:
            old_from = net.line.at[i, "from_bus"]
            old_to   = net.line.at[i, "to_bus"]
            net.line.at[i, "from_bus"] = old_to
            net.line.at[i, "to_bus"]   = old_from
    pp.runpp(net, **PF_SETTINGS)

def build_hv_only_ppc(net):
    """
    Runs AC power flow, orients lines, and extracts ppc arrays (bus, branch)
    for HV buses (vn_kv >= 110) plus any slack buses.
    Additionally excludes pandapower lines where length_km < 1.
    Further filters to include only "meshed" HV buses.
    """
    # For the first file, we assume the orientation has been set by orient_lines_by_flow.
    pp.runpp(net, **PF_SETTINGS)
    
    ppc = net._ppc
    if "bus" not in ppc or "branch" not in ppc:
        print("[Error] Missing ppc bus or branch.")
        return None, None, None

    bus_array = ppc["bus"].copy()
    branch_array = ppc["branch"].copy()

    # Identify HV buses based on voltage (>= 110 kV)
    hv_buses = net.bus[net.bus["vn_kv"] >= 110].index.tolist()
    slack_buses = net.gen[net.gen["slack"] == True].bus.values
    for sb in slack_buses:
        if sb not in hv_buses:
            hv_buses.append(sb)

    # Filter to only "meshed" buses.
    line_from = net.line["from_bus"]
    line_to   = net.line["to_bus"]
    mask_from = line_from.isin(hv_buses)
    mask_to   = line_to.isin(hv_buses)
    lines_in_hv = net.line[mask_from & mask_to]
    line_connections = pd.concat([lines_in_hv["from_bus"], lines_in_hv["to_bus"]])
    line_counts = line_connections.value_counts()

    if "trafo" in net and not net.trafo.empty:
        trafo_counts = net.trafo["hv_bus"].value_counts()
    else:
        trafo_counts = pd.Series(dtype=float)

    connectivity = line_counts.add(trafo_counts, fill_value=0)
    meshed_hv_buses = [bus for bus in hv_buses if connectivity.get(bus, 0) >= 2]
    hv_buses = meshed_hv_buses

    bus_lookup = net._pd2ppc_lookups["bus"]
    hv_ppc_idx = [bus_lookup[b] for b in hv_buses if b in bus_lookup]

    mask_bus = np.isin(bus_array[:, 0].astype(int), hv_ppc_idx)
    bus_array = bus_array[mask_bus, :]

    from_b = np.real(branch_array[:, F_BUS]).astype(int)
    to_b   = np.real(branch_array[:, T_BUS]).astype(int)
    mask_hv_br = np.isin(from_b, hv_ppc_idx) & np.isin(to_b, hv_ppc_idx)

    line_start, line_end = net._pd2ppc_lookups["branch"]["line"]
    line_ppc_indices = range(line_start, line_end)
    line_pp_indices  = net.line.index.tolist()
    ppc_idx_to_line_idx = dict(zip(line_ppc_indices, line_pp_indices))
    short_line_mask = (net.line["length_km"] < 1.1)
    short_line_idx = set(net.line[short_line_mask].index)
    mask_short_line = np.zeros(branch_array.shape[0], dtype=bool)
    for ppc_idx in line_ppc_indices:
        if ppc_idx in ppc_idx_to_line_idx:
            pandapower_line_idx = ppc_idx_to_line_idx[ppc_idx]
            if pandapower_line_idx in short_line_idx:
                mask_short_line[ppc_idx] = True

    mask_br = mask_hv_br & (~mask_short_line)
    orig_branch_idx = np.where(mask_br)[0]
    branch_array = branch_array[mask_br, :]

    return bus_array, branch_array, orig_branch_idx

def reindex_bus_ids(bus_array, branch_array):
    """
    Re-maps bus IDs to a contiguous 0..(n-1) range and updates branch connections.
    """
    if bus_array is None or branch_array is None or len(bus_array) == 0:
        return bus_array, branch_array, None
    
    old_bus_ids = np.real(bus_array[:, 0]).astype(int)
    n_buses = len(old_bus_ids)
    new_ids = np.arange(n_buses)
    new_to_old = dict(zip(new_ids, old_bus_ids))
    old_to_new = {old: new for new, old in new_to_old.items()}
    
    for i in range(n_buses):
        old_id = old_bus_ids[i]
        if old_id in old_to_new:
            bus_array[i, 0] = old_to_new[old_id]
        else:
            print(f"[Error] Bus {old_id} not found in reindex mapping!")
    
    from_b = np.real(branch_array[:, F_BUS]).astype(int)
    to_b = np.real(branch_array[:, T_BUS]).astype(int)
    for i in range(len(branch_array)):
        old_from = from_b[i]
        old_to = to_b[i]
        new_from = old_to_new.get(old_from, -1)
        new_to = old_to_new.get(old_to, -1)
        if new_from == -1 or new_to == -1:
            print(f"[Error] Unmapped branch {old_from} -> {old_to}!")
        branch_array[i, F_BUS] = new_from
        branch_array[i, T_BUS] = new_to
    
    return bus_array, branch_array, new_to_old, old_to_new

def map_branch_indices(net, orig_branch_idx):
    """
    Map each HV branch in the subnetwork to its original pandapower index.
    For lines, return the net.line index; for transformers, return the net.trafo index.
    Also returns a list indicating branch type ("line", "trafo", or "unknown").
    """
    branch_lookup = net._pd2ppc_lookups["branch"]
    line_range = branch_lookup.get("line", (0, 0))
    line_start, line_end = line_range
    net_line_indices = net.line.index.to_numpy()
    
    trafo_range = branch_lookup.get("trafo", None)
    if trafo_range is not None:
        trafo_start, trafo_end = trafo_range
        net_trafo_indices = net.trafo.index.to_numpy()
    else:
        trafo_start, trafo_end = (0, 0)
        net_trafo_indices = np.array([])
    
    mapped = np.full(len(orig_branch_idx), -1, dtype=int)
    mapped_type = ["" for _ in range(len(orig_branch_idx))]
    for idx, ppc_idx in enumerate(orig_branch_idx):
        if line_start <= ppc_idx < line_end:
            mapped[idx] = net_line_indices[ppc_idx - line_start]
            mapped_type[idx] = "line"
        elif trafo_range is not None and trafo_start <= ppc_idx < trafo_end:
            mapped[idx] = net_trafo_indices[ppc_idx - trafo_start]
            mapped_type[idx] = "trafo"
        else:
            mapped[idx] = -1
            mapped_type[idx] = "unknown"
    return mapped, mapped_type

@njit
def clip_and_replace(matrix, clip_min, clip_max):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                matrix[i, j] = 0.0
            elif val < clip_min:
                matrix[i, j] = clip_min
            elif val > clip_max:
                matrix[i, j] = clip_max
    return matrix

@njit(parallel=True)
def clip_and_replace_parallel(matrix, clip_min, clip_max):
    m, n = matrix.shape
    for i in prange(m):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                matrix[i, j] = 0.0
            elif val < clip_min:
                matrix[i, j] = clip_min
            elif val > clip_max:
                matrix[i, j] = clip_max
    return matrix

def compute_base_loading_vectorized(net, branch_array, orig_branch_idx):
    """
    Vectorized computation of base loading for each branch.
    For lines, use net.line["max_i_ka"]; for transformers, use net.trafo["sn_mva"].
    Converts DataFrame indices to positional indices.
    """
    mapped_branch_idx, mapped_type = map_branch_indices(net, orig_branch_idx)
    mapped_type = np.array(mapped_type)
    mapped_branch_idx = np.array(mapped_branch_idx)
    n_br = branch_array.shape[0]
    base_loading = np.zeros(n_br, dtype=float)
    
    line_positions = {idx: pos for pos, idx in enumerate(net.line.index)}
    trafo_positions = {idx: pos for pos, idx in enumerate(net.trafo.index)}
    
    line_mask = (mapped_type == "line") & (mapped_branch_idx != -1)
    if line_mask.any():
        line_indices = np.array([line_positions[x] for x in mapped_branch_idx[line_mask]])
        line_values = net.res_line["loading_percent"].fillna(0.0).to_numpy()
        base_loading[line_mask] = line_values[line_indices]
    
    trafo_mask = (mapped_type == "trafo") & (mapped_branch_idx != -1)
    if trafo_mask.any():
        trafo_indices = np.array([trafo_positions[x] for x in mapped_branch_idx[trafo_mask]])
        trafo_values = net.res_trafo["loading_percent"].fillna(0.0).to_numpy()
        base_loading[trafo_mask] = trafo_values[trafo_indices] / 1.25  # Transformer allowed up to 125%
    
    return base_loading

def compute_base_loading_ia_vectorized(net, branch_array, orig_branch_idx):
    """
    Vectorized computation of base loading (current or percentage) for each branch.
    For lines, use net.res_line["i_ka"]; for transformers, use net.res_trafo["loading_percent"].
    Converts DataFrame indices to positional indices.
    """
    mapped_branch_idx, mapped_type = map_branch_indices(net, orig_branch_idx)
    mapped_type = np.array(mapped_type)
    mapped_branch_idx = np.array(mapped_branch_idx)
    n_br = branch_array.shape[0]
    base_loading_ia = np.zeros(n_br, dtype=float)
    
    line_positions = {idx: pos for pos, idx in enumerate(net.res_line.index)}
    trafo_positions = {idx: pos for pos, idx in enumerate(net.res_trafo.index)}
    
    line_mask = (mapped_type == "line") & (mapped_branch_idx != -1)
    if line_mask.any():
        line_indices = np.array([line_positions[x] for x in mapped_branch_idx[line_mask]])
        line_vals = net.res_line["i_ka"].fillna(0.0).to_numpy()
        base_loading_ia[line_mask] = line_vals[line_indices]
    
    trafo_mask = (mapped_type == "trafo") & (mapped_branch_idx != -1)
    if trafo_mask.any():
        trafo_indices = np.array([trafo_positions[x] for x in mapped_branch_idx[trafo_mask]])
        trafo_vals = net.res_trafo["loading_percent"].fillna(0.0).to_numpy()
        base_loading_ia[trafo_mask] = trafo_vals[trafo_indices]
    
    return base_loading_ia

def compute_PTDF_LODF_OTDF(net):
    """
    Builds an HV-only ppc subnetwork, reindexes bus IDs, then computes PTDF, LODF, and OTDF.
    Returns (H, L, OTDF, bus_array, branch_array, new_to_old, orig_branch_idx, old_to_new).
    """
    bus_array, branch_array, orig_branch_idx = build_hv_only_ppc(net)
    if bus_array is None or branch_array is None or len(bus_array) == 0 or len(branch_array) == 0:
        return None, None, None, None, None, None, None, None

    bus_array, branch_array, new_to_old, old_to_new = reindex_bus_ids(bus_array, branch_array)
    
    Bbus, Bf, _, _, _ = makeBdc(bus_array, branch_array)
    Bbus = Bbus.tocsc()
    Bf   = Bf.tocsc()

    reg_factor = 1e-4
    Bbus_dense = Bbus.toarray() + reg_factor * np.eye(Bbus.shape[0])
    try:
        cond_num = np.linalg.cond(Bbus_dense)
        if cond_num < 1e12:
            invBbus = np.linalg.solve(Bbus_dense, np.eye(Bbus_dense.shape[0]))
        else:
            print("High condition number; using pseudoinverse.")
            invBbus = np.linalg.pinv(Bbus_dense, rcond=1e-6)
    except LinAlgError as e:
        print(f"B matrix inversion error: {e}")
        invBbus = np.linalg.pinv(Bbus_dense, rcond=1e-6)
    
    PTDF_mat = Bf.dot(invBbus)
    H = np.real(PTDF_mat)
    H = clip_and_replace_parallel(H, -1, 1)
    
    L = makeLODF(branch_array, H)
    L = np.real(L)
    L = np.nan_to_num(L, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n_br = H.shape[0]
    outage_branches = np.arange(n_br)
    OTDF_raw = makeOTDF(H, L, outage_branches)
    OTDF_raw = np.real(OTDF_raw)
    OTDF_raw = np.nan_to_num(OTDF_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    OTDF = clip_and_replace_parallel(OTDF_raw, -1e6, 1e6)
    
    return H, L, OTDF, bus_array, branch_array, new_to_old, orig_branch_idx, old_to_new

# ------------------ Batch Processing ------------------
def process_batch(batch_files, cached_topo):
    """
    Processes a list of files sequentially and returns aggregated results for the batch.
    If cached_topo already contains topology-dependent matrices under key "topology",
    we enforce a fixed line orientation using the cached values, then run AC power flow
    to update injections.
    
    Additionally, we check the current orientation versus the cached orientation:
    if a branch's current "from_bus" differs from the cached "from_bus", we flip
    the corresponding row in L, OTDF, and the resulting loading matrix.
    
    Returns:
      batch_agg: dict mapping original bus id to list of (OTDF contribution, circuit_debug)
      batch_contingency: dict mapping branch name to list of (intact_flow, max_post_flow, worst_outage)
      sample mapping info and a sample network for bus mapping
    """
    batch_agg = {}
    batch_contingency = {}
    sample_new_to_old = None
    sample_net = None

    for f in batch_files:
        print(f"\nProcessing file: {f}")
        file_start = time.time()
        net_path = os.path.join(PKL_DIR, f)
        try:
            net = pp.from_pickle(net_path)
        except Exception as e:
            print(f"[Error] Could not load {f}: {e}")
            continue

        preprocess_network(net)
        orient_lines_by_flow(net)

        # Build debug dictionary for circuit info.
        circuit_info = {}
        for line_idx in net.line.index:
            ln_name = net.line.at[line_idx, "name"]
            fb_orig = net.line.at[line_idx, "from_bus_orig"]
            tb_orig = net.line.at[line_idx, "to_bus_orig"]
            circuit_info[line_idx] = f"'{ln_name.strip()}' (orig_from={fb_orig}, orig_to={tb_orig})"

        # Compute and cache topology-dependent matrices on the first file.
        if "topology" not in cached_topo:
            print("Computing topology-dependent matrices...")
            result = compute_PTDF_LODF_OTDF(net)
            if result[0] is None:
                print(f"[Warning] Could not compute PTDF/LODF/OTDF for file {f}; skipping.")
                continue
            # Cache the fixed orientation.
            cached_topo["line_from"] = net.line["from_bus"].copy()
            cached_topo["line_to"] = net.line["to_bus"].copy()
            cached_topo["topology"] = result
        else:
            result = cached_topo["topology"]

        H, L, OTDF, topo_bus_array, topo_branch_array, new_to_old, orig_branch_idx, old_to_new = result
        n_br, n_bus = H.shape

        sample_new_to_old = new_to_old
        sample_net = net

        # Compute injection-dependent base loadings.
        base_loading = compute_base_loading_vectorized(net, topo_branch_array, orig_branch_idx)
        base_loading_ia = compute_base_loading_ia_vectorized(net, topo_branch_array, orig_branch_idx)

        # Recalculate loading matrix with updated injections.
        loading_matrix = base_loading[:, None] + L * base_loading[None, :]
        loading_matrix = np.real(loading_matrix)
        loading_matrix_og = loading_matrix

        # Map branches.
        mapped_branch_idx, mapped_type = map_branch_indices(net, orig_branch_idx)

        # --- Compute per-branch flip factors (vectorized approach) ---
        # For each branch j, we want flip_factors[j] = -1 if the current orientation (from_bus)
        # differs from the cached base-hour orientation; otherwise, +1.
        flip_factors = np.ones(n_br)
        for j in range(n_br):
            branch_idx = mapped_branch_idx[j]
            if mapped_type[j] == "line" and branch_idx in net.line.index:
                current_from = net.line.at[branch_idx, "from_bus"]
                # Use .get() to avoid KeyError; default to current_from if missing.
                cached_from = cached_topo["line_from"].get(branch_idx, current_from)
                if current_from != cached_from:
                    flip_factors[j] = -1
                    # For debugging
                    # print(f"Branch {branch_idx} flip: cached_from={cached_from}, current_from={current_from}")

        # --- Apply flip factors in a single vectorized pass ---
        # Update L: Each element L[j,k] becomes flip_factors[j]*L[j,k]*flip_factors[k]
        L = flip_factors[:, np.newaxis] * L * flip_factors[np.newaxis, :]

        # Recalculate the loading matrix with updated L:
        # loading_matrix[j,k] = base_loading[j] + L[j,k]*base_loading[k]
        loading_matrix = base_loading[:, np.newaxis] + L * base_loading[np.newaxis, :]
        
        # --- Correct OTDF in a vectorized manner ---
        # At this point, OTDF has been computed as:
        # OTDF shape: (n_br*n_br, n_bus)
        # --- Correct OTDF with affected branch correction only ---
        print("OTDF shape before correction:", OTDF.shape)
        n_br = flip_factors.shape[0]  # number of branches

        # Since OTDF is stacked in blocks of shape (n_br, n_bus) for each outage,
        # each row's affected branch index equals (row index modulo n_br).
        total_rows = OTDF.shape[0]  # should equal n_br * n_br if outage_branches covers all branches
        affected_indices = np.arange(total_rows) % n_br  # affected branch for each row

        # Build a correction vector using only the affected branch flip factors.
        correction_vector = flip_factors[affected_indices]  # shape: (n_br*n_br,)

        # Apply the correction to each row.
        OTDF = OTDF * correction_vector[:, np.newaxis]
        print("OTDF correction applied based solely on affected branch.")

        # Recompute max loading values after correction.
        max_loading_values = loading_matrix.max(axis=1)
        max_loading_indices = loading_matrix.argmax(axis=1)

        max_loading_values_og = loading_matrix_og.max(axis=1)
        max_loading_indices_og = loading_matrix_og.argmax(axis=1)

        # Build per-file contingency data.
        for j in range(n_br):
            net_branch_idx = mapped_branch_idx[j]
            branch_type = mapped_type[j]
            if branch_type == "line" and net_branch_idx != -1:
                branch_name = net.line.at[net_branch_idx, "name"]
                intact_loading_pct = net.res_line.at[net_branch_idx, "loading_percent"]
                max_ia_loading = net.line.at[net_branch_idx, "max_i_ka"]
                i_star = max_loading_indices[j]
                max_loading_fraction = max_loading_values[j]
            elif branch_type == "trafo" and net_branch_idx != -1:
                branch_name = net.trafo.at[net_branch_idx, "name"]
                intact_loading_pct = net.res_trafo.at[net_branch_idx, "loading_percent"]
                max_ia_loading = net.trafo.at[net_branch_idx, "sn_mva"]
                i_star = max_loading_indices[j]
                max_loading_fraction = max_loading_values[j]
            else:
                branch_name = "Unknown"
                intact_loading_pct = 0.0
                max_ia_loading = 0.0
                i_star = max_loading_indices[j]
            outage_name = "Unknown"
            if i_star < n_br:
                o_net_branch_idx = mapped_branch_idx[i_star]
                o_branch_type = mapped_type[i_star]
                if o_net_branch_idx != -1:
                    if o_branch_type == "line":
                        outage_name = net.line.at[o_net_branch_idx, "name"]
                    elif o_branch_type == "trafo":
                        outage_name = net.trafo.at[o_net_branch_idx, "name"]
            batch_contingency.setdefault(branch_name, []).append(
                (intact_loading_pct, max_loading_fraction, outage_name)
            )

        # Update batch aggregator for OTDF contributions.
        threshold_idx = np.where(max_loading_values >= OVERLOAD_THRESHOLD)[0]
        for j in threshold_idx:
            net_branch_idx = mapped_branch_idx[j]
            branch_type = mapped_type[j]
            if net_branch_idx >= 0:
                if branch_type == "line":
                    circuit_debug = circuit_info.get(net_branch_idx, "UnknownCircuit")
                elif branch_type == "trafo":
                    circuit_debug = f"Transformer: {net.trafo.at[net_branch_idx, 'name']}"
                else:
                    circuit_debug = "UnknownCircuit"
            else:
                circuit_debug = "UnknownCircuit"
            i_star = max_loading_indices[j]
            row_idx = i_star * n_br + j
            Xj_unscaled = np.real(OTDF[row_idx, :])
            scaling_factor = max_loading_values[j]
            Xj = Xj_unscaled * scaling_factor
            for b_idx in range(n_bus):
                orig_bus_id = new_to_old[b_idx]
                batch_agg.setdefault(orig_bus_id, []).append((Xj[b_idx], circuit_debug))
        gc.collect()
        file_end = time.time()
        print(f"Total runtime for {f}: {file_end - file_start:.2f} seconds")
    return batch_agg, batch_contingency, sample_new_to_old, sample_net

BUS_DATA_CSV_index_fix   = "Model/bus_lodf_aggregator_June_Sept.csv"

def merge_global_aggregators():
    """
    Reads all intermediate aggregator CSV files and merges them into a single final CSV file,
    computing only the circuit/outage tied to the max value in each group.
    """

    # --- Load bus names mapping once ---
    bus_names = (
        pd.read_csv(BUS_DATA_CSV_index_fix, usecols=["Bus_ID","Bus_Name"])
          .rename(columns={"index":"Bus_ID", "bus_names":"Bus_Name"})
    )

    # --- Merge OTDF aggregator CSVs ---
    agg_files = sorted(f for f in os.listdir(RESULTS_DIR)
                       if f.startswith("agg_batch_") and f.endswith(".csv"))
    if not agg_files:
        print("No intermediate OTDF aggregation files found.")
    else:
        # 1) concat all the batch files
        df = pd.concat(
            (pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in agg_files),
            ignore_index=True
        )
        # 2) bring in Bus_Name
        df = df.merge(bus_names, on="Bus_ID", how="left")

        df2 = pd.concat(
            (pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in agg_files),
            ignore_index=True
        )
        # 2) bring in Bus_Name
        df2 = df2.merge(bus_names, on="Bus_ID", how="left")

        # 3) compute the five stats for OTDF_Contribution
        statss = (
            df.groupby(["Bus_ID","Bus_Name"], as_index=False)["scaled_OTDF_Contribution"]
              .agg(
                  Cum_sOTDF   = "sum",
                  Avg_sOTDF   = "mean",
                  Med_sOTDF   = "median",
                  Max_sOTDF   = "max",
                  Min_sOTDF   = "min"
              )
        )
        
        # Make sure contributions are floats
        df2["OTDF_Contribution"] = pd.to_numeric(df2["unscaled_OTDF_Contribution"], errors="coerce")
        # If you have a scaled column:
        if "unscaled_OTDF_Contribution" in df2.columns:
            df2["unscaled_OTDF_Contribution"] = pd.to_numeric(
                df2["unscaled_OTDF_Contribution"], errors="coerce"
            )

        # 2) compute the five stats for OTDF_Contribution
        stats = (
            df2.groupby(["Bus_ID","Bus_Name"], as_index=False)["unscaled_OTDF_Contribution"]
            .agg(
                Cum_OTDF = "sum",
                Avg_OTDF = "mean",
                Med_OTDF = "median",
                Max_OTDF = "max",
                Min_OTDF = "min"
            )
        )

        # 4) extract exactly the Circuit_Debug at the max-OTDF row
        idxmax = (
            df.groupby(["Bus_ID","Bus_Name"])["scaled_OTDF_Contribution"]
              .idxmax()
        )
        idxmin = (
            df.groupby(["Bus_ID","Bus_Name"])["scaled_OTDF_Contribution"]
              .idxmin()
        )
        idx2max = (
            df2.groupby(["Bus_ID","Bus_Name"])["unscaled_OTDF_Contribution"]
              .idxmax()
        )
        idx2min = (
            df2.groupby(["Bus_ID","Bus_Name"])["unscaled_OTDF_Contribution"]
              .idxmin()
        )
        circ = df.loc[idxmax, ["Bus_ID","Bus_Name","Circuit_Debug","Branch_Percentage","scaled_OTDF_Contribution"]]
        circ = circ.rename(columns={"Circuit_Debug":"Circuit_Involved_max",
                                    "Branch_Percentage":"Branch Loading max",
                                    "scaled_OTDF_Contribution": "Scaled OTDF_Contribution_max"})
        circm = df.loc[idxmin, ["Bus_ID","Bus_Name","Circuit_Debug","Branch_Percentage","scaled_OTDF_Contribution"]]
        circm = circm.rename(columns={"Circuit_Debug":"Circuit_Involved_min",
                                    "Branch_Percentage":"Branch Loading min",
                                    "scaled_OTDF_Contribution": "Scaled OTDF_Contribution_min"})
        circ2 = df2.loc[idx2max, ["Bus_ID","Bus_Name","Circuit_Debug","Branch_Percentage","unscaled_OTDF_Contribution"]]
        circ2 = circ2.rename(columns={"Circuit_Debug":"Circuit_Involved_max",
                                    "Branch_Percentage":"Branch Loading max",
                                    "unscaled_OTDF_Contribution": "Unscaled_OTDF_Contribution_max"})
        circ2m = df2.loc[idx2min, ["Bus_ID","Bus_Name","Circuit_Debug","Branch_Percentage","unscaled_OTDF_Contribution"]]
        circ2m = circ2m.rename(columns={"Circuit_Debug":"Circuit_Involved_min",
                                    "Branch_Percentage":"Branch Loading min",
                                    "unscaled_OTDF_Contribution": "Unscaled_OTDF_Contribution_min"})

        # 5) stitch them back together
        df_agg_final1 = statss.merge(circ, on=["Bus_ID","Bus_Name"], how="left")
        df_agg_final1 = df_agg_final1.merge(circm, on=["Bus_ID","Bus_Name"], how="left")
        df_agg_final2 = stats.merge(circ2, on=["Bus_ID","Bus_Name"], how="left")
        df_agg_final2 = df_agg_final2.merge(circ2m, on=["Bus_ID","Bus_Name"], how="left")
        df_agg_final = df_agg_final1.merge(df_agg_final2, on=["Bus_ID","Bus_Name"], how="left")

        # 6) write it out
        out1 = os.path.join(RESULTS_DIR, "global_agg_final.csv")
        df_agg_final.to_csv(out1, index=False)
        print(f"[Success] Final global OTDF aggregator saved to {out1}")

    # --- Merge contingency aggregator CSVs ---
    cont_files = sorted(f for f in os.listdir(RESULTS_DIR)
                        if f.startswith("contingency_batch_") and f.endswith(".csv"))
    if not cont_files:
        print("No intermediate contingency aggregation files found.")
    else:
        dfc = pd.concat(
            (pd.read_csv(os.path.join(RESULTS_DIR, f)) for f in cont_files),
            ignore_index=True
        )
        # 1) flow‐percent stats
        cont_stats = (
            dfc.groupby("Branch_Name", as_index=False)
               .agg(
                   Avg_Intact_Flow_Percent        = ("Intact_Flow",    "mean"),
                   Med_Intact_Flow_Percent        = ("Intact_Flow",    "median"),
                   Max_Intact_Flow_Percent        = ("Intact_Flow",    "max"),
                   Min_Intact_Flow_Percent        = ("Intact_Flow",    "min"),
                   Avg_Post_Contingency_Flow_Percent  = ("Max_Post_Flow", "mean"),
                   Med_Post_Contingency_Flow_Percent  = ("Max_Post_Flow", "median"),
                   Max_Post_Contingency_Flow_Percent  = ("Max_Post_Flow", "max"),
                   Min_Post_Contingency_Flow_Percent  = ("Max_Post_Flow", "min"),
                   Hour_Count                      = ("Max_Post_Flow", "size")
               )
        )

        # 2) pick the single Worst_Case_Outage at the row with the max post‐contingency flow
        idx2 = dfc.groupby("Branch_Name")["Max_Post_Flow"].idxmax()
        idx2m = dfc.groupby("Branch_Name")["Max_Post_Flow"].idxmin()
        outages = dfc.loc[idx2, ["Branch_Name","Worst_Circuit"]]
        outages = outages.rename(columns={"Worst_Circuit":"Worst_Circuit_max"})
        outagesm = dfc.loc[idx2m, ["Branch_Name","Worst_Circuit"]]
        outagesm = outagesm.rename(columns={"Worst_Circuit":"Worst_Circuit_min"})


        # 3) merge and write
        df_cont_final = cont_stats.merge(outages, on="Branch_Name", how="left")
        df_cont_final_m = df_cont_final.merge(outagesm, on="Branch_Name", how="left")
        out2 = os.path.join(RESULTS_DIR, "global_contingency_agg_final.csv")
        df_cont_final_m.to_csv(out2, index=False)
        print(f"[Success] Final global contingency aggregator saved to {out2}")


# ------------------ Main Routine with Batch Processing ------------------
def main():
    # After batch processing, now merge the intermediate CSVs to form the final aggregated CSV.
    merge_global_aggregators()

if __name__ == "__main__":
    main()
