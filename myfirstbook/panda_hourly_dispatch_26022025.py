#!/usr/bin/env python3
"""
Refactored Economic Dispatch and Hourly Simulation Script
-----------------------------------------------------------
- Loads network, load, and generator data.
- For each hour, scales load and sets wind/solar outputs per capacity factors.
- The remaining load gap (load - renewables) is sequentially filled:
    UPWARD (if generation is insufficient): battery discharge, then interconnectors, then conventional.
    DOWNWARD (if generation is too high): reduce conventional generation (non-min units), then interconnectors, then battery charging.
- Uses proportional adjustment (adjust_dispatch_proportional) that updates the gap sequentially.
- Iterates dispatch adjustments until the slack (from the slack generator) is within ±40 MW.
- Uses AC power flow.
- Saves hourly network states and writes summary results.

This finalized version has been moved to individual python file on 26022025

"""

import os, time, copy, re, warnings
import pandas as pd, numpy as np, pandapower as pp
import matplotlib.pyplot as plt

# --- Configurable Parameters and Directories ---
DATA_DIR = "Model/"
RESULTS_DIR = "Results/"
os.makedirs(RESULTS_DIR, exist_ok=True)
SLACK_TOL = 40          # MW tolerance for slack generator output
MAX_ITER = 25           # Maximum iterations per hour
BESS_DURATION = 4       # in hours

# --- Load Base Data ---
net = pp.from_pickle("adjusted_network.p")

# Update functions for transformers, buses, lines, generators, and reactive equipment
def update_load_from_csv(net, csv_file):
    load_df = pd.read_csv(csv_file)
    for index, row in load_df.iterrows():
        matched_loads = net.load[(net.load.name == row['name'])]
        if not matched_loads.empty:
            load_index = matched_loads.index[0]
            for col in row.index:
                net.load.at[load_index, col] = row[col]
        else:
            print(f"Warning: load named {row['name']} not found in the network.")
            pp.create_load(net, name=row['name'], bus=row['bus'], p_mw=row['p_mw'], q_mvar=row['q_mvar'], 
                            in_service=row['in_service'],scaling=row['scaling'])

# Assume update_load_from_csv is defined externally.
update_load_from_csv(net, os.path.join(DATA_DIR, 'pp_load_data_WP2031.csv'))
gen_orig = pd.read_csv(os.path.join(DATA_DIR, 'pp_generator_data.csv'))
gen_orig['type'] = gen_orig['type'].astype(int)
gen_orig['max_p_mw'] = gen_orig['max_p_mw'].astype(float)
gen_orig['min_p_mw'] = gen_orig['min_p_mw'].astype(float)

#Setting interconnectors to start at zero
mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
net.gen.loc[mask_int, 'p_mw'] = 0

min_conv_units=["  'Huntstown'", "     'PBEGG6'", "   'AGH_CCGT'", "Knockfinglas"]
mask_min_gas = (net.gen['type'] == 9) & (net.gen['name'].isin(min_conv_units))
mak_all_gas = (net.gen['type'] == 9)
min_gas_mw_level = net.gen.loc[mask_min_gas, 'min_p_mw'].sum()
max_gas_mw_level = net.gen.loc[mak_all_gas, 'max_p_mw'].sum()

# For battery storage (type 14)
total_bess_max_mw = gen_orig.loc[(gen_orig['type'] == 14) & (gen_orig['in_service'] == True), 'max_p_mw'].sum()
total_bess_max_mwh = total_bess_max_mw * BESS_DURATION
bess_energy = 0  # initial energy in MWh

# Load hourly timeseries data (columns: "Wind CF", "Solar CF", "IE Demand factor of Peak", etc.)
time_series_data = pd.read_csv(os.path.join(DATA_DIR, "Full_Year_hourly_sample_week.csv"))

# --- Utility Functions ---

# Define power flow attempts as a list of tuples (initialization method, algorithm)
power_flow_attempts = [
    #('flat', 'nr', True),   # Newton-Raphson with flat start, qlims set to true
    #('flat', 'nr', False),  # Newton-Raphson with flat start, qlims set to false
    ('dc', 'nr', True),     # Newton-Raphson with DC start, qlims set to true (most likely to solve using this)
    #('dc', 'nr', False),    # Newton-Raphson with DC start, qlims set to false
]  

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

def update_lines_from_csv(net, csv_file, max_i_column='max_i_ka'):
    line_df = pd.read_csv(csv_file)
    for index, row in line_df.iterrows():
        line_indices = net.line[net.line.index == row['line_id']].index
        if not line_indices.empty:
            line_index = line_indices[0]
            for col in row.index:
                net.line.at[line_index, col] = row[col]
        else:
            print(f"Warning: Line with line_id {row['line_id']} not found in the network.")
            pp.create_line_from_parameters(net, from_bus=row['from_bus'], to_bus=row['to_bus'], index=row['line_id'],
                                           length_km=row['length_km'], r_ohm_per_km=row['r_ohm_per_km'], 
                                           x_ohm_per_km=row['x_ohm_per_km'], c_nf_per_km=row['c_nf_per_km'], 
                                           max_i_ka=row[max_i_column], name=row['name'], in_service=row['in_service'], 
                                           max_loading_percent=row['max_loading_percent'])

def orient_lines_by_flow(net):
    """
    Runs a DC power flow, then re-orients each line if its flow is negative.
    Re-runs DC to update flows.
    """
    for i in net.line.index:
        p_from = net.res_line.at[i, "p_from_mw"]
        if p_from < 0:
            old_from = net.line.at[i, "from_bus"]
            old_to   = net.line.at[i, "to_bus"]
            net.line.at[i, "from_bus"] = old_to
            net.line.at[i, "to_bus"]   = old_from

def attempt_power_flow(net, attempts=1):
    # Below is to display generation vs load balance for ROI
    total_load = net.load.p_mw.sum()
    total_generation = net.gen.loc[net.gen['in_service'] == True, 'p_mw'].sum()
    print(f"Total Load: {total_load} MW, Total Generation: {total_generation} MW")
    
    for init_method, algorithm, q_lims in attempts:
        try:
            pp.runpp(net, init=init_method, max_iteration=100, calculate_voltage_angles=True,
                     enforce_q_lims=q_lims, tolerance_mva=5e-3, algorithm=algorithm, logging=False,
                     voltage_depend_loads=False, v_debug=True)
            if net.converged:
                print(f"Power flow calculation successful with {init_method} start and {algorithm} algorithm, with q lims set to {q_lims}")
                return True
            else:
                print("Power flow did not converge, checking mismatches...")
        except Exception as e:
            print(f"Power flow calculation failed for {init_method} start {algorithm}, with q lims set to {q_lims}: {e}")
            #Below optional to see diagnostic if powerflow fails to converge
            #diagnostic_results = pp.diagnostic(net, report_style='detailed')
            #print(diagnostic_results)
    return False

def normalize_line_name(line_name):
    return ' '.join(line_name.strip().split())

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def compute_slack(net):
    slack_indices = net.gen.index[net.gen.slack == True]
    total_slack = sum(net.res_gen.at[idx, "p_mw"] for idx in slack_indices)
    return total_slack

# --- Bulk Dispatch Functions ---
def dispatch_renewables(net, wind_cf, solar_cf, curtail_re = False):
    wind_types = [17,18,19,20,21,22]
    all_re_types = [16,17,18,19,20,21,22,23]
    mask_wind = (net.gen['type'].isin(wind_types)) & (net.gen['in_service'] == True)
    mask_solar = (net.gen['type'] == 23) & (net.gen['in_service'] == True)
    mask_offshore = (net.gen['type'] == 16) & (net.gen['in_service'] == True)
    mask_wind_negative = (net.gen['type'].isin(wind_types)) & (net.gen['in_service'] == True) & (net.gen['p_mw'] <= 0)
    if curtail_re:
        mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)
        mask_all_re_negative = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True) & (net.gen['p_mw'] <= 0)
        all_max_re = net.gen.loc[mask_all_re, 'max_p_mw'].sum()
        net.gen.loc[mask_all_re, 'p_mw'] += ((net.gen.loc[mask_all_re, 'max_p_mw'] / all_max_re) * curtail_re)
        net.gen.loc[mask_all_re_negative, 'p_mw'] = 0
        print(f"Renewables curtailed by: {curtail_re}")
        all_onshore_pw_re_min = net.gen.loc[mask_wind_negative, 'p_mw'].sum()
        if all_onshore_pw_re_min < 0:
            print("All onshore below zero:", all_onshore_pw_re_min)
    else:
        net.gen.loc[mask_wind, 'p_mw'] = wind_cf * net.gen.loc[mask_wind, 'max_p_mw']
        net.gen.loc[mask_solar, 'p_mw'] = solar_cf * net.gen.loc[mask_solar, 'max_p_mw']
        if wind_cf > 0.2:
            net.gen.loc[mask_offshore, 'p_mw'] = 0.2 * net.gen.loc[mask_offshore, 'max_p_mw']
        else:
            net.gen.loc[mask_offshore, 'p_mw'] = wind_cf * net.gen.loc[mask_offshore, 'max_p_mw']
            
        # Zone‐based caps
        net.gen.loc[(net.gen['zone'] == 1) & (wind_cf > 0.65) & mask_wind, 'p_mw'] = 0.65 * net.gen.loc[(net.gen['zone'] == 1) & mask_wind, 'max_p_mw']
        net.gen.loc[(net.gen['zone'] == 2) & (wind_cf > 0.8) & mask_wind, 'p_mw'] = 0.8 * net.gen.loc[(net.gen['zone'] == 2) & mask_wind, 'max_p_mw']

    mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)    
    renewable_gen = net.gen.loc[mask_all_re, 'p_mw'].sum()
    
    #debug
    print(f"Amount of renewables dispatched: {renewable_gen}")
    return net

def dispatch_battery(net, load_gap, bess_energy):
    """
    Adjust battery (type 14) dispatch subject to energy capacity:
      - If load_gap > 0 (need additional generation): Discharge up to available battery energy.
      - If load_gap < 0 (excess generation): Charge battery up to remaining capacity.
    Updates net.gen and returns updated bess_energy.
    """
    mask_bess = (net.gen['type'] == 14) & (net.gen['in_service'] == True)
    total_bess_power = net.gen.loc[mask_bess, 'max_p_mw'].sum()
    total_bess_power_min = net.gen.loc[mask_bess, 'min_p_mw'].sum()
    total_bess_current = net.gen.loc[mask_bess, 'p_mw'].sum()
    #debug
    print(f"Total current BESS generation level: {total_bess_current}")
    #If the batteries are already dispatched from the last gap, they shouldn't double their efforts
    if load_gap > 0 and (abs(load_gap - total_bess_current) > 0.001):
        available_discharge = min(bess_energy,total_bess_power - total_bess_current)
        discharge = min(load_gap, available_discharge)
        if available_discharge > 0:
            ratio = net.gen.loc[mask_bess, 'max_p_mw'] / total_bess_power
            net.gen.loc[mask_bess, 'p_mw'] += ratio * discharge
        bess_energy -= discharge
        load_gap -= discharge
        print(f"Batteries discharged {discharge:.2f} MW; new bess_energy: {bess_energy:.2f} MWh")
    elif load_gap < 0 and (abs(load_gap - total_bess_current) > 0.001):
        available_charge = total_bess_max_mwh - bess_energy
        # For charging, maximum charging power is assumed as total_bess_power.
        available_charging = min(available_charge, total_bess_power + total_bess_current)
        charge = min(abs(load_gap), available_charging)
        if available_charging > 0:
            ratio = net.gen.loc[mask_bess, 'max_p_mw'] / total_bess_power
            net.gen.loc[mask_bess, 'p_mw'] -= ratio * charge
        bess_energy += charge
        load_gap += charge
        print(f"Batteries charged {charge:.2f} MW; new bess_energy: {bess_energy:.2f} MWh")
    return net, bess_energy

def dispatch_interconnectors(net, load_gap):
    mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
    int_gen = net.gen.loc[mask_int, 'p_mw'].sum()
    if load_gap > 0 and (abs(load_gap - int_gen) > 0.001):
        avail_int = (net.gen.loc[mask_int, 'max_p_mw'] - net.gen.loc[mask_int, 'p_mw']).clip(lower=0).sum()
        add_int = min(load_gap, avail_int)
        if avail_int > 0:
            ratio_int = net.gen.loc[mask_int, 'max_p_mw'] / avail_int
            net.gen.loc[mask_int, 'p_mw'] += ratio_int * add_int
        load_gap -= add_int
        print(f"Interconnectors add {add_int:.2f} MW; remaining gap: {load_gap:.2f} MW")
    elif load_gap < 0 and (abs(load_gap - int_gen) > 0.001):
        avail_int = (net.gen.loc[mask_int, 'p_mw'] - net.gen.loc[mask_int, 'min_p_mw']).clip(lower=0).sum()
        red_int = min(abs(load_gap), avail_int)
        if avail_int > 0:
            ratio_int = abs(net.gen.loc[mask_int, 'min_p_mw']) / avail_int
            net.gen.loc[mask_int, 'p_mw'] -= ratio_int * red_int
        load_gap += red_int
        print(f"Interconnectors reduce {red_int:.2f} MW; remaining gap: {load_gap:.2f} MW")
    
    #debug
    int_gen = net.gen.loc[mask_int, 'p_mw'].sum()
    print(f"Interconnector dispatch in int: {int_gen}")
    return net

def dispatch_conventional(net, load_gap, min_conv_units=["  'Huntstown'", "     'PBEGG6'", "   'AGH_CCGT'", "Knockfinglas"]):
    mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
    quant_gas = net.gen[mask_gas].shape[0]
    if load_gap > 0:
        avail_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0).sum()
        if avail_gas == 0 and quant_gas == 4:
            mask_removable_gas = (net.gen['type'] == 9) & (~net.gen['name'].isin(min_conv_units))
            if not mask_removable_gas.empty:
                net.gen.loc[mask_removable_gas, 'in_service'] = True
                print(f"Some conventional units added to service to improve differential.")
                mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
                avail_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0).sum()
        add_gas = min(load_gap, avail_gas)
        if avail_gas > 0:
            ratio_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0) / avail_gas
            net.gen.loc[mask_gas, 'p_mw'] += ratio_gas * add_gas
        load_gap -= add_gas
        print(f"Conventional add {add_gas:.2f} MW; remaining gap: {load_gap:.2f} MW")
    else:
        mask_tot = mask_gas #& (~net.gen['name'].isin(min_conv_units))
        avail_gas_down = (net.gen.loc[mask_tot, 'p_mw'] - net.gen.loc[mask_tot, 'min_p_mw']).clip(lower=0).sum()
        if avail_gas_down == 0 and quant_gas > 4:
            mask_nonmin = mask_gas & (~net.gen['name'].isin(min_conv_units))
            net.gen.loc[mask_nonmin, 'in_service'] = False
            print(f"Some conventional units removed from service to improve differential.")
            mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
            avail_gas_down = (net.gen.loc[mask_gas, 'p_mw'] - net.gen.loc[mask_gas, 'min_p_mw']).clip(lower=0).sum()
        red_gas = min(abs(load_gap), avail_gas_down)
        if avail_gas_down > 0:
            ratio_gas = (net.gen.loc[mask_gas, 'p_mw'] - net.gen.loc[mask_gas, 'min_p_mw']).clip(lower=0) / avail_gas_down
            net.gen.loc[mask_gas, 'p_mw'] -= ratio_gas * red_gas
        load_gap += red_gas
        print(f"Conventional reduce {red_gas:.2f} MW; remaining gap: {load_gap:.2f} MW")
        
    #debug
    gas_gen = net.gen.loc[mask_gas, 'p_mw'].sum()
    print(f"Conventional dispatch in dispatch conv: {gas_gen}")
    return net

def bulk_dispatch(net, wind_cf, solar_cf, bess_energy,bulk_gap = 0):
    """
    Performs bulk dispatch update:
      1. Dispatch wind and solar (renewables).
      2. Compute gap = total load - renewable generation.
      3. Dispatch battery, then interconnectors, then conventional generators.
      4. Run AC power flow.
    Returns updated network and bess_energy.
    """
    wind_types = [16,17,18,19,20,21,22]
    mask_wind = (net.gen['type'].isin(wind_types)) & (net.gen['in_service'] == True)
    mask_solar = (net.gen['type'] == 23) & (net.gen['in_service'] == True)
    renewable_gen = net.gen.loc[mask_wind, 'p_mw'].sum() + net.gen.loc[mask_solar, 'p_mw'].sum()
    total_load = net.load["p_mw"].sum()
    if bulk_gap:
        gap = bulk_gap
    else:
        gap = total_load - (renewable_gen + min_gas_mw_level)
    print(f"Bulk dispatch: Gap after renewables: {gap} MW")
    
    # Step 2: Battery dispatch (if gap > 0, discharge; if gap < 0, charge)
    net, bess_energy = dispatch_battery(net, gap, bess_energy)
    mask_bess = (net.gen['type'] == 14) & (net.gen['in_service'] == True)
    battery_gen = net.gen.loc[mask_bess, 'p_mw'].sum()
    gap = total_load - (renewable_gen + battery_gen + min_gas_mw_level)
    print(f"Bulk dispatch: Gap after battery: {gap:.2f} MW")
    
    # Step 3: Interconnector dispatch
    net = dispatch_interconnectors(net, gap)
    mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
    int_gen = net.gen.loc[mask_int, 'p_mw'].sum()
    #debug
    print(f"Interconnector dispatch in bulk disp: {int_gen}")
    total_gen = net.gen.loc[net.gen.in_service == True, "p_mw"].sum()
    #gap = total_load - (renewable_gen + battery_gen + int_gen)
    gap = total_load - total_gen
    print(f"Bulk dispatch: Gap after interconnectors: {gap:.2f} MW")
    
    # Step 4: Conventional dispatch
    net = dispatch_conventional(net, gap)
    total_gen = net.gen.loc[net.gen.in_service == True, "p_mw"].sum()
    #gap = total_load - (renewable_gen + battery_gen + int_gen)
    gap = total_load - total_gen
    current_gas_mw = net.gen.loc[(net.gen['type'] == 9) & (net.gen['in_service'] == True), 'p_mw'].sum()
    
    # Finalize dispatch by running AC power flow
    #pp.runpp(net, algorithm='nr', tolerance_mva=5e-3, max_iteration=100, logging=False)
    return net, bess_energy

def adjust_dispatch_proportional(net, gap, bess_energy, min_conv_units=["  'Huntstown'", "     'PBEGG6'", "   'AGH_CCGT'", "Knockfinglas"], curtailed_action=0):
    """
    Adjust generator dispatch sequentially based on the gap.
    UPWARD (gap > 0): First use batteries, then interconnectors, then conventional.
    DOWNWARD (gap < 0): First reduce conventional (non-min) generation, then interconnectors, then batteries.
    The function updates the remaining gap as it proceeds.
    Returns updated net.
    """
    mask_bess = (net.gen['type'] == 14) & (net.gen['in_service'] == True)
    total_bess_current = net.gen.loc[mask_bess, 'p_mw'].sum()
    
    all_re_types = [16,17,18,19,20,21,22,23]
    mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)
    all_mw_re = net.gen.loc[mask_all_re, 'p_mw'].sum()
    
    if gap > 0:
        # UPWARD: Need extra generation.
        # 1a. undo curtailed renewables
        if curtailed_action > 0:
            net = dispatch_renewables(net, wind_cf, solar_cf, curtail_re=gap)
            all_re_types = [16,17,18,19,20,21,22,23]
            mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)
            all_mw_re_change = all_mw_re - net.gen.loc[mask_all_re, 'p_mw'].sum()
            gap += all_mw_re_change
            print(f"Proportional Upward: Renewables curtail undone by adding {abs(all_mw_re_change):.2f} MW; gap remaining {gap:.2f} MW")
            
        # 1b. Batteries:
        mask_bess = (net.gen['type'] == 14) & (net.gen['in_service'] == True)
        available_discharge = min(bess_energy, net.gen.loc[mask_bess, 'max_p_mw'].sum() - total_bess_current)
        add_bess = min(gap, available_discharge)
        if available_discharge > 0:
            ratio = net.gen.loc[mask_bess, 'max_p_mw'] / net.gen.loc[mask_bess, 'max_p_mw'].sum()
            net.gen.loc[mask_bess, 'p_mw'] += ratio * add_bess
        gap -= add_bess
        bess_energy -= add_bess
        print(f"Proportional Upward: Batteries add {add_bess:.2f} MW; gap remaining {gap:.2f} MW")
        
        # 2. Interconnectors:
        if gap > 0:
            mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
            avail_int = (net.gen.loc[mask_int, 'max_p_mw'] - net.gen.loc[mask_int, 'p_mw']).clip(lower=0).sum()
            add_int = min(gap, avail_int)
            if avail_int > 0:
                ratio_int = (net.gen.loc[mask_int, 'max_p_mw'] - net.gen.loc[mask_int, 'p_mw']).clip(lower=0) / avail_int
                net.gen.loc[mask_int, 'p_mw'] += ratio_int * add_int
            gap -= add_int
            print(f"Proportional Upward: Interconnectors add {add_int:.2f} MW; gap remaining {gap:.2f} MW")
            #debug
            int_gen = net.gen.loc[mask_int, 'p_mw'].sum()
            print(f"Interconnector dispatch in int: {int_gen}")
        
        # 3. Conventional:
        if gap > 0:
            mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
            quant_gas = net.gen[mask_gas].shape[0]
            avail_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0).sum()
            if avail_gas == 0 and quant_gas == 4:
                mask_removable_gas = (net.gen['type'] == 9) & (~net.gen['name'].isin(min_conv_units))
                if not mask_removable_gas.empty:
                    net.gen.loc[mask_removable_gas, 'in_service'] = True
                    print(f"Some conventional units added to service to improve differential.")
                    #debug
                    mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
                    print(f"Quantity of gas gens: {net.gen[mask_gas].shape[0]}")
                    avail_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0).sum()
            add_gas = min(gap, avail_gas)
            if avail_gas > 0:
                ratio_gas = (net.gen.loc[mask_gas, 'max_p_mw'] - net.gen.loc[mask_gas, 'p_mw']).clip(lower=0) / avail_gas
                net.gen.loc[mask_gas, 'p_mw'] += ratio_gas * add_gas
            gap -= add_gas
            print(f"Proportional Upward: Conventional add {add_gas:.2f} MW; gap remaining {gap:.2f} MW")
    
    else:
        # DOWNWARD: Need to reduce generation.
        gap_abs = abs(gap)
        # 1. Reduce conventional generation (only non-min units):
        mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
        quant_gas = net.gen[mask_gas].shape[0]
        mask_nonmin = mask_gas & (~net.gen['name'].isin(min_conv_units))
        avail_gas_down = (net.gen.loc[mask_nonmin, 'p_mw'] - net.gen.loc[mask_nonmin, 'min_p_mw']).clip(lower=0).sum()
        if avail_gas_down == 0 and quant_gas > 4:
            net.gen.loc[mask_nonmin, 'in_service'] = False
            print(f"Some conventional units removed from service to improve differential.")
            mask_gas = (net.gen['type'] == 9) & (net.gen['in_service'] == True)
            avail_gas_down = (net.gen.loc[mask_gas, 'p_mw'] - net.gen.loc[mask_gas, 'min_p_mw']).clip(lower=0).sum()
        red_gas = min(gap_abs, avail_gas_down)
        #debug
        print(f"red_gas after minimum applied before proportional downward: {red_gas}")
        if avail_gas_down > 0:
            ratio_gas = (net.gen.loc[mask_gas, 'p_mw'] - net.gen.loc[mask_gas, 'min_p_mw']).clip(lower=0) / avail_gas_down
            net.gen.loc[mask_gas, 'p_mw'] -= ratio_gas * red_gas
        gap_abs -= red_gas
        print(f"Proportional Downward: Conventional reduced by {red_gas:.2f} MW; gap remaining {gap_abs:.2f} MW")
        
        # 2. Then reduce interconnectors:
        if gap_abs > 0:
            mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
            avail_int_down = (net.gen.loc[mask_int, 'p_mw'] - net.gen.loc[mask_int, 'min_p_mw']).clip(lower=0).sum()
            red_int = min(gap_abs, avail_int_down)
            if avail_int_down > 0:
                ratio_int = (net.gen.loc[mask_int, 'p_mw'] - net.gen.loc[mask_int, 'min_p_mw']).clip(lower=0) / avail_int_down
                net.gen.loc[mask_int, 'p_mw'] -= ratio_int * red_int
            gap_abs -= red_int
            print(f"Proportional Downward: Interconnectors reduced by {red_int:.2f} MW; gap remaining {gap_abs:.2f} MW")
            #debug
            int_gen = net.gen.loc[mask_int, 'p_mw'].sum()
            print(f"Interconnector dispatch in int: {int_gen}")
        
        # 3. Finally, adjust batteries (increase charging).
        if gap_abs > 0:
            mask_bess = (net.gen['type'] == 14) & (net.gen['in_service'] == True)
            avail_bess_down = min(total_bess_max_mwh - bess_energy, net.gen.loc[mask_bess, 'max_p_mw'].sum() + total_bess_current)
            red_bess = min(gap_abs, avail_bess_down)
            if avail_bess_down > 0:
                ratio_bess = net.gen.loc[mask_bess, 'max_p_mw'] / net.gen.loc[mask_bess, 'max_p_mw'].sum()
                net.gen.loc[mask_bess, 'p_mw'] -= ratio_bess * red_bess
            gap_abs -= red_bess
            bess_energy += red_bess
            print(f"Proportional Downward: Batteries reduced by {red_bess:.2f} MW; gap remaining {gap_abs:.2f} MW")
        
        #If other steps fail, reduce renewables
        #if (gap_abs == abs(gap)) & (gap_abs > 0):
        #    net = dispatch_renewables(net, wind_cf, solar_cf, curtail_re = gap)
            
    
    return net,bess_energy

def economic_dispatch_loop(net, wind_cf, solar_cf, bess_energy, max_iter=25, slack_tol=40):
    """
    Iteratively adjust dispatch until the slack (from the slack generator) is within ±slack_tol.
    In each iteration, bulk_dispatch is called; then the gap is computed and the proportional adjustment
    (adjust_dispatch_proportional) is applied.
    Returns updated net, bess_energy, and convergence flag.
    """
    previous_gap = 0
    net = dispatch_renewables(net, wind_cf, solar_cf)
    all_re_types = [16,17,18,19,20,21,22,23]
    mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)
    all_mw_re = net.gen.loc[mask_all_re, 'p_mw'].sum()
    gap = 0
    for iteration in range(max_iter):
        mask_all_re = net.gen['type'].isin(all_re_types) & (net.gen['in_service'] == True)
        all_mw_re_update = net.gen.loc[mask_all_re, 'p_mw'].sum()
        curtailed_action = all_mw_re - all_mw_re_update
        
        if attempt_power_flow(net,power_flow_attempts):
            slack = compute_slack(net)
            print(f"Iteration {iteration}: Slack = {slack:.2f} MW")
            gap = slack
            if abs(slack) <= slack_tol:
                print("Convergence achieved: Slack within target range.")
                return net, bess_energy, True
            net,bess_energy = adjust_dispatch_proportional(net, gap, bess_energy, curtailed_action=curtailed_action)
        else:
            net, bess_energy = bulk_dispatch(net, wind_cf, solar_cf, bess_energy,bulk_gap=gap)
            total_load = net.load["p_mw"].sum()
            total_gen = net.gen.loc[net.gen.in_service == True, "p_mw"].sum()
            gap = total_load - total_gen
            print(f"Iteration {iteration}: Overall gap = {gap:.2f} MW")
        
        # Reduce renewables if other steps aren't enough
        if (gap < 0) & (previous_gap == gap):
            curtail_re = gap
            net = dispatch_renewables(net, wind_cf, solar_cf, curtail_re=curtail_re)
        
        previous_gap = gap

        
    print("Convergence not achieved within maximum iterations.")
    return net, bess_energy, False

# --- Main Simulation Loop ---
results = []
for hour, ts_row in time_series_data.iterrows():
    print(f"\nProcessing Hour {hour} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    start_time = time.time()
    if hour == 0:
        net = copy.deepcopy(net)
    else:
        update_load_from_csv(net, os.path.join(DATA_DIR, 'pp_load_data_WP2031.csv'))
    
    wind_cf = ts_row["Wind CF"]
    solar_cf = ts_row["Solar CF"]
    load_scaling = ts_row["IE Demand factor of Winter Peak"]
    season = ts_row["Season"]

    #season change line ratings
    if season == "winter":
        net.line['max_i_ka'] = net.line['wp_max_i_ka']

    mask_load = (net.load['scale'] == 1)
    
    net.load.loc[mask_load,"p_mw"] *= load_scaling
    net.load.loc[mask_load,"q_mvar"] *= load_scaling
    
    mask_bess = (net.gen['type'] == 14)
    net.gen.loc[mask_bess, 'p_mw'] = 0  # Reset battery dispatch
    
    #Setting interconnectors to start at zero
    mask_int = net.gen['name'].isin(['Celtic', "   'EASTWEST'", "  'Greenlink'"])
    net.gen.loc[mask_int, 'p_mw'] = 0
    
    net, bess_energy, converged = economic_dispatch_loop(net, wind_cf, solar_cf, bess_energy, max_iter=25, slack_tol=40)
    if not converged:
        print(f"Hour {hour} did not converge. Skipping results.")
        continue
    #debug
    current_bess_energy_ratio = bess_energy / total_bess_max_mwh 
    print(f"After converged, BESS energy capacity ratio (1.0 is full): {current_bess_energy_ratio}")
    
    #pp.runpp(net, **PF_SETTINGS)
    #orient_lines_by_flow(net)
    #pp.runpp(net, **PF_SETTINGS)
    pp.to_pickle(net, f"Full_Year/adjusted_network_{hour}.p")
    slack = compute_slack(net)
    total_gen = net.gen.loc[net.gen.in_service == True, "p_mw"].sum()
    total_load = net.load["p_mw"].sum()
    
    results.append({
        "hour": hour,
        "wind_cf": wind_cf,
        "solar_cf": solar_cf,
        "bess_energy": bess_energy,
        "load_scaling": load_scaling,
        "slack": slack,
        "total_generation": total_gen,
        "total_load": total_load
    })
    
    end_time = time.time()
    print(f"Time taken for hour {hour}: {end_time - start_time:.2f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "time_series_results.csv"), index=False)
print("\nHourly time-series simulation complete. Results saved.")
