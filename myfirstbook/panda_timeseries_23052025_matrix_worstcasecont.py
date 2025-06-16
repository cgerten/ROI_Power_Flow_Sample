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
PKL_DIR = "Model/timeseries/shortlist"  # Folder with .p files
BUS_DATA_CSV = "Model/bus_data.csv"
LINE_DATA_CSV = "Model/line_data.csv"

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
OTDF_filter = 0.01 # filter out OTDF values that are less than this value or greater than this negative value
test_solar_size = 10 # example solar project specified by user to see spilled vs utilized energy over the simulation
test_wind_size = 10 # example wind project specified by user to see spilled vs utilized energy over the simulation
BATCH_SIZE = 1          # Adjust as needed
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

def flush_batch_aggregators(batch_idx, batch_agg, batch_contingency):
    """
    Writes the batch aggregator data to Parquet files and returns empty dictionaries.
    """
    # Flush OTDF aggregator for the batch.
    if batch_agg:
        df_batch = pd.DataFrame([
            {"Bus_ID": bus, "scaled_OTDF_Contribution": contrib, "Circuit_Debug": debug,"Branch Loading":cont_loading,"Percent Reduction": pct_reduction}
            for bus, entries in batch_agg.items()
            for contrib, debug, cont_loading, pct_reduction in entries
        ])
        agg_file = os.path.join(RESULTS_DIR, f"agg_batch_{batch_idx:03d}.parquet")
        df_batch["Branch Loading"] = pd.to_numeric(df_batch["Branch Loading"],
                                                errors="coerce").fillna(0.0)

        df_batch.to_parquet(agg_file, index=False)
        print(f"Flushed batch OTDF aggregator to {agg_file}")
    else:
        print(f"No OTDF aggregator data for batch {batch_idx}")

    # Flush contingency aggregator for the batch.
    if batch_contingency:
        df_cont = pd.DataFrame([
            {"Branch_Name": branch, "Intact_Flow": intact, "Max_Post_Flow": post, "Worst_Circuit": circuit}
            for branch, entries in batch_contingency.items()
            for intact, post, circuit in entries
        ])
        cont_file = os.path.join(RESULTS_DIR, f"contingency_batch_{batch_idx:03d}.parquet")
        df_cont.to_parquet(cont_file, index=False)
        print(f"Flushed batch contingency aggregator to {cont_file}")
    else:
        print(f"No contingency aggregator data for batch {batch_idx}")

    return {}, {}


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
    
    After processing each file, we explicitly delete large objects and call gc.collect()
    to free memory.
    
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
        solar_types = [23]
        wind_types = [17,18,19,20,21,22] #onshore wind
        mask_solar = (net.gen['type'].isin(solar_types)) & (net.gen['in_service'] == True)
        all_solar_pw_re = net.gen.loc[mask_solar, 'p_mw'].sum()
        all_solar_max_pw_re = net.gen.loc[mask_solar, 'max_p_mw'].sum()
        mask_wind = (net.gen['type'].isin(wind_types)) & (net.gen['in_service'] == True)
        all_onshore_pw_re = net.gen.loc[mask_wind, 'p_mw'].sum()
        all_onshore_max_pw_re = net.gen.loc[mask_wind, 'max_p_mw'].sum()
        solar_cf = all_solar_pw_re / all_solar_max_pw_re
        wind_cf = all_onshore_pw_re / all_onshore_max_pw_re


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
        loading_matrix_og = loading_matrix.copy()

        # Map branches.
        mapped_branch_idx, mapped_type = map_branch_indices(net, orig_branch_idx)

        # --- Compute per-branch flip factors (currently using a loop) ---
        flip_factors = np.ones(n_br)
        for j in range(n_br):
            branch_idx = mapped_branch_idx[j]
            if mapped_type[j] == "line" and branch_idx in net.line.index:
                current_from = net.line.at[branch_idx, "from_bus"]
                cached_from = cached_topo["line_from"].get(branch_idx, current_from)
                if current_from != cached_from:
                    flip_factors[j] = -1

        # --- Apply flip factors in a vectorized pass ---
        L = flip_factors[:, np.newaxis] * L * flip_factors[np.newaxis, :]
        loading_matrix = base_loading[:, None] + L * base_loading[np.newaxis, :]
        
        # --- Correct OTDF: affected branch correction only ---
        print("OTDF shape before correction:", OTDF.shape)
        total_rows = OTDF.shape[0]  # expected shape: n_br*n_br x n_bus
        affected_indices = np.arange(total_rows) % n_br
        correction_vector = flip_factors[affected_indices]
        OTDF = OTDF * correction_vector[:, np.newaxis]
        print("OTDF correction applied based solely on affected branch.")

        # Recompute max loading values.
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

        # --- Optimized Aggregation Loop for OTDF Contributions ---
        threshold_idx = np.where(max_loading_values >= OVERLOAD_THRESHOLD)[0]
        #threshold_idx = np.where(max_loading_values >= 1)[0]
        contrib_records = []  # list to accumulate tuples of (orig_bus_id, contribution, circuit_debug)
        p = net.res_bus["p_mw"].to_numpy()
        q = net.res_bus["q_mvar"].to_numpy()
        bus_injection = np.sqrt(p**2 + q**2)


        for j in threshold_idx:
        #for j in range(n_br):
            net_branch_idx = mapped_branch_idx[j]
            branch_type = mapped_type[j]
            if net_branch_idx >= 0:
                if branch_type == "line":
                    circuit_debug = circuit_info.get(net_branch_idx, "UnknownCircuit")
                    circuit_rating = net.line.at[net_branch_idx, 'max_i_ka']*net.line.at[net_branch_idx, 'kv_from']*1.732
                    #branch_loading = max_loading_values[j]
                elif branch_type == "trafo":
                    circuit_debug = f"Transformer: {net.trafo.at[net_branch_idx, 'name']}"
                    circuit_rating = net.trafo.at[net_branch_idx, 'sn_mva']
                    #branch_loading = max_loading_values[j]
                else:
                    circuit_debug = "UnknownCircuit"
            else:
                circuit_debug = "UnknownCircuit"

            i_star = max_loading_indices[j]
            row_idx = i_star * n_br + j
            Xj_unscaled = np.real(OTDF[row_idx, :])
            scaling_factor = max_loading_values[j]
            mva_loading_over = ((scaling_factor/100) * circuit_rating) - circuit_rating
            nz = np.where(np.abs(Xj_unscaled) >= OTDF_filter)[0]

            # Accumulate contributions for all buses at once
            for b_idx in nz:
                orig_bus_id = new_to_old[b_idx]
                mva_reduction = Xj_unscaled[b_idx] * mva_loading_over
                inj = bus_injection[orig_bus_id]
                pct_of_injection = (mva_reduction / inj)*100 if inj != 0 else (mva_reduction / 1)*100 #assumes buses with no injection inject at least 1 MVA
                branch_loading = max_loading_values[j]
                sc = Xj_unscaled[b_idx] * scaling_factor
                cd = circuit_debug
                contrib_records.append((orig_bus_id, sc, cd,branch_loading,pct_of_injection))
        
        # Convert the records to a DataFrame for efficient grouping.
        if contrib_records:
            df_contrib = pd.DataFrame(contrib_records, columns=["Bus_ID", "Scaled Contribution","Circuit_Debug","Branch Loading","Percent Reduction"])
            # Group the contributions by Bus_ID.
            for bus, group in df_contrib.groupby("Bus_ID"):
                # Convert the group DataFrame to a list of tuples.
                batch_agg.setdefault(bus, []).extend(list(group[["Scaled Contribution", "Circuit_Debug","Branch Loading","Percent Reduction"]].itertuples(index=False, name=None)))

        
        gc.collect()
        file_end = time.time()
        print(f"Total runtime for {f}: {file_end - file_start:.2f} seconds")
        
        # Explicitly delete large objects to free memory.
        del net, base_loading, base_loading_ia, loading_matrix, loading_matrix_og, H, L, OTDF
        gc.collect()
    
    return batch_agg, batch_contingency, sample_new_to_old, sample_net

def merge_global_aggregators():
    """
    Reads intermediate Parquet files and merges them into final outputs.
    Writes both Parquet and CSV for OTDF and contingency aggregations.
    """
    #debug
    print(f"Begin merge global aggregator")
    # --- Load bus names mapping ---
    BUS_DATA_CSV_index_fix = "Model/bus_lodf_aggregator_June_Sept.csv"
    bus_names = pd.read_csv(BUS_DATA_CSV_index_fix, usecols=["Bus_ID", "Bus_Name"] )

    # --- Merge OTDF aggregator Parquets ---
    agg_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("agg_batch_") and f.endswith(".parquet")
    )
    if not agg_files:
        print("No intermediate OTDF aggregation files found.")
    else:
        per_file_dfs = []
        for fn in agg_files:
            # Filename format is "agg_batch_{batch_idx:03d}.parquet"
            # e.g. "agg_batch_003.parquet" → batch_idx = 3
            batch_str = fn.split("_")[2].split(".")[0]   # "003"
            hour = int(batch_str)                       # 3
            single_df = pd.read_parquet(os.path.join(RESULTS_DIR, fn))
            single_df["Hour"] = hour
            per_file_dfs.append(single_df)

        #Concatenate all of those "augmented with Hour" DataFrames
        df = pd.concat(per_file_dfs, ignore_index=True)
        df["Branch Loading"] = pd.to_numeric(df["Branch Loading"], errors="coerce").fillna(0.0)

        #Merge in Bus_Name from the static bus_names table
        df = df.merge(bus_names, on="Bus_ID", how="left")

        #Compute the basic OTDF statistics per (Bus_ID, Bus_Name):
        statss = (
            df
            .groupby(["Bus_ID", "Bus_Name"], as_index=False)["scaled_OTDF_Contribution"]
            .agg(
                Cum_sOTDF   = "sum",
                Avg_sOTDF   = "mean",
                Med_sOTDF   = "median",
                Max_sOTDF   = "max",
                Min_sOTDF   = "min"
            )
        )
        #keep only the rows that actually achieve the group max
        mx = df.groupby(["Bus_ID", "Bus_Name"])["scaled_OTDF_Contribution"].transform("max")
        candidates = df[df["scaled_OTDF_Contribution"] == mx]

        #if there are several, keep the one with the largest Branch Loading (> 0 first)
        candidates = candidates.sort_values(
                        ["Bus_ID", "Bus_Name", "Branch Loading"],
                        ascending=[True, True, False], kind="mergesort")

        idx_max_otdf = candidates.groupby(["Bus_ID", "Bus_Name"], sort=False).head(1).index

        circ_max = (
            df.loc[idx_max_otdf,
                   ["Bus_ID", "Bus_Name", "Circuit_Debug", "Branch Loading", "Percent Reduction",
                    "scaled_OTDF_Contribution"]]
              .rename(columns={
                  "Circuit_Debug":            "Circuit_Involved_max",
                  "scaled_OTDF_Contribution": "Scaled_OTDF_Contribution_max",
                  "Branch Loading":           "Branch_Loading_max",
                  "Percent Reduction":        "Percent_Reduction_max"
              })
        )
        #Similarly, for each bus find the single row where scaled_OTDF_Contribution is minimal
        #idx_min_otdf = df.groupby(["Bus_ID", "Bus_Name"])["scaled_OTDF_Contribution"].idxmin()
        #idx_min_otdf = idx_min_otdf.dropna().astype(int)
        #keep only the rows that actually achieve the group max
        mx_min = df.groupby(["Bus_ID", "Bus_Name"])["scaled_OTDF_Contribution"].transform("min")
        candidates_min = df[df["scaled_OTDF_Contribution"] == mx_min]

        #if there are several, keep the one with the smallest Branch Loading (> 0 first)
        candidates_min = candidates_min.sort_values(
                        ["Bus_ID", "Bus_Name", "Branch Loading"],
                        ascending=[True, True, False], kind="mergesort")

        idx_min_otdf = candidates_min.groupby(["Bus_ID", "Bus_Name"], sort=False).head(1).index
        circ_min = (
            df.loc[idx_min_otdf,
                   ["Bus_ID", "Bus_Name", "Circuit_Debug", "Branch Loading", "Percent Reduction",
                    "scaled_OTDF_Contribution"]]
              .rename(columns={
                  "Circuit_Debug":            "Circuit_Involved_min",
                  "scaled_OTDF_Contribution": "Scaled_OTDF_Contribution_min",
                  "Branch Loading":           "Branch_Loading_min",
                  "Percent Reduction":        "Percent_Reduction_min"
              })
        )


        #Merge into one final DataFrame
        df_agg_final = (
            statss
            .merge(circ_max,              on=["Bus_ID", "Bus_Name"], how="left")
            .merge(circ_min,              on=["Bus_ID", "Bus_Name"], how="left")
        )
        parquet_path = os.path.join(RESULTS_DIR, "global_agg_final.parquet")
        csv_path     = os.path.join(RESULTS_DIR, "global_agg_final.csv")
        df_agg_final.to_parquet(parquet_path, index=False)
        df_agg_final.to_csv(csv_path, index=False)
        print(f"[Success] Final global OTDF aggregator saved to {parquet_path} and {csv_path}")

    # --- Merge contingency aggregator Parquets ---
    cont_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("contingency_batch_") and f.endswith(".parquet")
    )
    if not cont_files:
        print("No intermediate contingency aggregation files found.")
    else:
        dfc = pd.concat(
            (pd.read_parquet(os.path.join(RESULTS_DIR, f)) for f in cont_files),
            ignore_index=True
        )
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
        idx2 = dfc.groupby("Branch_Name")["Max_Post_Flow"].idxmax()
        outages = dfc.loc[idx2, ["Branch_Name","Worst_Circuit"]]
        outages = outages.rename(columns={"Worst_Circuit":"Worst_Circuit_max"})
        idx2m = dfc.groupby("Branch_Name")["Max_Post_Flow"].idxmin()
        outagesm = dfc.loc[idx2m, ["Branch_Name","Worst_Circuit"]]
        outagesm = outagesm.rename(columns={"Worst_Circuit":"Worst_Circuit_min"})

        df_cont_final = (
            cont_stats
            .merge(outages, on=["Branch_Name"], how="left")
            .merge(outagesm, on=["Branch_Name"], how="left")
        )
        parquet_path = os.path.join(RESULTS_DIR, "global_contingency_agg_final.parquet")
        csv_path     = os.path.join(RESULTS_DIR, "global_contingency_agg_final.csv")
        df_cont_final.to_parquet(parquet_path, index=False)
        df_cont_final.to_csv(csv_path, index=False)
        print(f"[Success] Final global contingency aggregator saved to {parquet_path} and {csv_path}")

# ------------------ Main Routine with Batch Processing ------------------
def main():
    pkl_files = sorted([f for f in os.listdir(PKL_DIR) if f.endswith(".p")], key=numerical_sort)
    batches = [pkl_files[i:i+BATCH_SIZE] for i in range(0, len(pkl_files), BATCH_SIZE)]
    
    global_agg = {}             # Aggregator for bus OTDF contributions (global)
    global_contingency_agg = {} # Aggregator for branch contingency data (global)
    sample_new_to_old = None    # Mapping info from last processed batch
    sample_net = None           # Sample network for bus mapping
    overall_start = time.time()
    
    # Dictionary to cache topology-dependent matrices and fixed line orientation.
    cached_topo = {}
    
    for batch_idx, batch in enumerate(batches):
        print(f"\nProcessing batch {batch_idx+1} of {len(batches)} (Files {batch[0]} ... {batch[-1]})")
        batch_agg, batch_contingency, sample_new_to_old, sample_net = process_batch(batch, cached_topo)
        
        # Update global aggregators:
        for bus_id, values in batch_agg.items():
            global_agg.setdefault(bus_id, []).extend(values)
        for branch_name, values in batch_contingency.items():
            global_contingency_agg.setdefault(branch_name, []).extend(values)
        
        # Flush the current batch results to intermediate files and clear batch_agg/batch_contingency.
        batch_agg, batch_contingency = flush_batch_aggregators(batch_idx, batch_agg, batch_contingency)
        
        # Optionally, every 10 batches, flush the global aggregator to disk and clear it.
        #if (batch_idx + 1) % 10 == 0:
        #    df_global_temp = pd.DataFrame([
        #        {"Bus_ID": bus, "scaled_OTDF_Contribution": scaled_contrib,"unscaled_OTDF_Contribution": unscaled_contrib, "Branch_Percentage":branch_percentage,"Circuit_Debug": debug}
        #        for bus, entries in global_agg.items() for scaled_contrib,unscaled_contrib,branch_percentage, debug in entries
        #    ])
        #    global_file = os.path.join(RESULTS_DIR, f"global_agg_until_batch_{batch_idx+1:03d}.csv")
        #    df_global_temp.to_csv(global_file, index=False)
        #    print(f"Flushed global OTDF aggregator up to batch {batch_idx+1} to {global_file}")
        #    global_agg.clear()
        
        gc.collect()
    
    overall_end = time.time()
    print(f"Total runtime: {overall_end - overall_start:.2f} seconds")
    
    # After batch processing, now merge the intermediate CSVs to form the final aggregated CSV.
    merge_global_aggregators()


if __name__ == "__main__":

    main()
