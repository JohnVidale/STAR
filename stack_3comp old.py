#%% make plot of the brute stack of 3 components on all arrays, aligned with P or S time
# John Vidale 5/2025
from obspy import read, Stream, UTCDateTime
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd  # Added to read the CSV file
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
import csv

os.chdir('/Users/vidale/Documents/Research/STAR')

# Read the event list CSV and select event "ci40789071"
event_file = "Lists/List_all.csv"
df = pd.read_csv(event_file)

# does all 3 channels
array_select = [1, 2, 3, 4, 5] # Select the arrays to process
# array_select = [2]
comp_list = ["*Z", "*1", "*2"]
lat_center = [  33.609800,   33.483522,   33.327171,   33.373449,   33.473353] # use element 41 for center
lon_center = [-116.454500, -116.151843, -116.366194, -116.62345, -116.646267]
align_phase        = "P" # "P" for P-wave, "S" for S-wave alignment
save_figs          = True
be_choosy          = True   # discard traces that are often bad
moveout_correction = False   # correct for predicted slowness moveout
static_correction  = True   # apply static corrections to traces

# Define offsets (in seconds) relative to the picked phase arrival time
start_time = -4  # seconds relative to pick, for start of analysis window
end_time   = 20   # seconds relative to pick, for end   of analysis window

min_freq   = 0.5 # Frequency range for bandpass filter
max_freq   = 2

vertical_offset = 1.2  # For visual separation among components
plot_mag        = 2.0 # Vertical amplification factor for plot

# Hypocenter, the CSV has columns "id", "time", "latitude", "longitude", and "depth"
evid = "ci40183482" # "ci40789071"
evt = df[df['id'] == evid].iloc[0]
origin_time_str = evt['time']  # e.g., "2024-11-07T08:39:06"
event_time  = UTCDateTime(origin_time_str)
event_lat   = float(evt['latitude'])
event_lon   = float(evt['longitude'])
event_depth = float(evt['depth'])  # Event depth in km
print(f"Event {evid} with origin time {event_time} lat={event_lat}, lon={event_lon}, depth={event_depth} km")

# Load the TauPyModel for travel time calculations
# model = TauPyModel(model="iasp91")
model = TauPyModel(model="sp6")

# Load station metadata
stations_file = "Lists/STAR_stations.csv"
stations_df = pd.read_csv(stations_file)

# Load station statics
# data_file = "Lists/output3_" + evid + ".csv"
data_file = "Lists/statics.csv"
data_df = pd.read_csv(data_file)

# Define the directory to save figures
save_dir = "Plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List to store figure handles
fig_list = []

# Create one common figure with one subplot per array (vertical stack)
narr = len(array_select)
fig, axs = plt.subplots(nrows=narr, ncols=1, figsize=(10, narr * 3))
if narr == 1:
    axs = [axs]

#%% Loop over arrays
for idx, i in enumerate(array_select):
    ax = axs[idx]  # subplot for current array
    print(f"Processing Array {i}")
    # ----------------------------
    center_deg = locations2degrees(event_lat, event_lon, lat_center[i-1], lon_center[i-1])
    print(f"    Epicentral distance: {center_deg*111:.2f} km")

    # Compute travel times using the first station.
    travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                          distance_in_degree=center_deg,
                                          phase_list=["p", "P", "s", "S", "PvmP", "SvmS", "PcP", "ScS"])
    print("    Central station travel times and slowness:")
    for tt in reversed(travel_times):
        slowness_sk = tt.ray_param / 111.19
        print(f"      Phase: {tt.phase.name.upper()}, Travel Time: {tt.time:.2f} s, Slowness: {slowness_sk:.4f} s/km")
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
            p_arrival_time = event_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = tt.time
            s_arrival_time = event_time + s_traveltime
        if tt.phase.name.upper() == "PVMP":
            pmp_traveltime = tt.time
            pmp_arrival_time = event_time + pmp_traveltime
        if tt.phase.name.upper() == "SVMS":
            sms_traveltime = tt.time
            sms_arrival_time = event_time + sms_traveltime
        if tt.phase.name.upper() == "PCP":
            pcp_traveltime = tt.time
            pcp_arrival_time = event_time + pcp_traveltime
        if tt.phase.name.upper() == "SCS":
            scs_traveltime = tt.time
            scs_arrival_time = event_time + scs_traveltime

    # Define the fixed window relative to the chosen arrival.
    if align_phase == "P":
        abs_pick   = p_arrival_time + start_time
        rel_pick   = p_arrival_time
        s_time_rel = s_traveltime - p_traveltime
        p_time_rel = 0
        pmp_time_rel = pmp_traveltime - p_traveltime
        sms_time_rel = sms_traveltime - p_traveltime
    elif align_phase == "S":
        abs_pick   = s_arrival_time + start_time
        rel_pick   = s_arrival_time
        s_time_rel = 0
        p_time_rel = p_traveltime - s_traveltime
        pmp_time_rel = pmp_traveltime - s_traveltime
        sms_time_rel = sms_traveltime - s_traveltime
    # elif align_phase == "PMP":
    #     abs_pick   = pmp_arrival_time + start_time
    #     rel_pick   = pmp_arrival_time
    #     s_time_rel = 0
    #     p_time_rel = p_traveltime - pmp_traveltime
    # elif align_phase == "SMS":
    #     abs_pick   = sms_arrival_time + start_time
    #     rel_pick   = sms_arrival_time
    #     s_time_rel = 0
    #     p_time_rel = p_traveltime - sms_traveltime
    else:
        print(f"Did not find chosen arrival")
        continue

    # Read the data file for Array i.
    hr = abs_pick.hour
    day = abs_pick.julday
    year = abs_pick.year
    year_str = f"{year:04d}"
    day_str  = f"{day:03d}"
    hr_str   = f"{hr:02d}"
    input_file = f"/Volumes/STAR2/Array{i}/{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
    if not os.path.exists(input_file):
        print(f"STAR seismogram-hour file not found, skipping: {input_file}")
        continue

    st = read(input_file)
    print(f"    File read: {input_file}")

    # Keep only traces with data in the fixed time window.
    hour_traces = Stream()
    for tr in st:
        # Define station_id from the current trace as a string
        station_id = int(tr.stats.station)

        ## Reject problematic stations
        data_info = data_df[data_df['station'] == int(station_id)]
        if data_info.empty:
            print(f"Station {station_id} not in station file!")
            exit(-1)
        select = bool(data_info.iloc[0]['select'])
        if be_choosy and select == False:
            continue
        
        if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + (end_time - start_time):
            hour_traces.append(tr)
        # print(f"hour_traces len is {len(hour_traces)}")


    if len(hour_traces) == 0:
        print(f"No data for time window in file: {input_file}")
        continue

    # Slice the hour traces to the fixed time window.
    st_window = hour_traces.slice(starttime=abs_pick, endtime=abs_pick + end_time - start_time)

    # Dictionaries to store final (averaged) stacks and vertical offsets.
    comp_stacks = {}
    comp_offsets = {}

    # Loop over components and process each (do not normalize each trace).
    for comp in comp_list:
        st_comp = st_window.select(channel=comp)
        num_traces = len(st_comp)
        if num_traces == 0:
            print(f"No traces found for component {comp} in Array {i}")
            continue

        # Process each trace: detrend, taper, filter.
        for tr in st_comp:
            tr.detrend(type="demean")
            tr.taper(max_percentage=0.01, type="cosine")
            tr.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=4, zerophase=False)

        sample_rate = st_comp[0].stats.sampling_rate
        nstack = int(sample_rate * (end_time - start_time))
        stack_shifted = np.zeros(nstack)
        count = 0

        # Loop over traces and sum shifted traces.
        for tr in st_comp:
            unshifted = tr.data[:nstack]

            #     Find moveout correction
            # Define station_id from the current trace as a string
            # station_id = str(tr.stats.station)
            station_id = tr.stats.station
            dt = tr.stats.delta

            station_info = stations_df[stations_df['station'] == int(station_id)]
            if station_info.empty:
                print(f"Station {station_id} not found in stations file!")
                exit(-1)
                continue  # or handle the missing case appropriately
            else:
                station_meta = station_info.iloc[0]

            station_lat = float(station_meta['latitude'])
            station_lon = float(station_meta['longitude'])

            station_deg = locations2degrees(event_lat, event_lon, station_lat, station_lon)
            # print(f"  Station {station_id}  Epicentral distance: {station_deg*111:.2f} km")

            travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                                distance_in_degree=station_deg,
                                                phase_list=["p", "P", "s", "S", "PmP", "SmS"])
            # print(f"    1st station = {travel_times} s")

            # Extract the travel times and arrival times for P and S phases
            # Travel times are checked in reverse order so that the first arrival can be chosen.
            for tt in reversed(travel_times):
                # slowness_sk = tt.ray_param / 111.19
                if tt.phase.name.upper() == "P":
                    p_diff = p_traveltime - tt.time
                if tt.phase.name.upper() == "S":
                    s_diff = s_traveltime - tt.time

            if   align_phase == "P":
                shift = p_diff
            elif align_phase == "S":
                shift = s_diff

            # Define station_id from the current trace as a string
            station_id = int(tr.stats.station)

            ## apply P or S static corrections from the CSV file
            if static_correction:
                data_info = data_df[data_df['station'] == int(station_id)]
                if data_info.empty:
                    print(f"Station {station_id} not in station file!")
                    exit(-1)
                if   align_phase == "P":
                    static_corr = int(data_info.iloc[0]['p_static'])
                elif align_phase == "S":
                    static_corr = int(data_info.iloc[0]['s_static'])
                if moveout_correction:
                    shift = shift + (static_corr/100)
                else:
                    shift = static_corr/100
                # print(f"    Static correction for {station_id}: {static_corr} s")

            if moveout_correction or static_correction:
                shifted = np.roll(unshifted, int(-shift*100))
                stack_shifted += shifted
            else:
                stack_shifted += unshifted
            count += 1

        if count > 0:
            stack_shifted /= count  # Average the stack (without normalization yet)
            print(f"Component {comp}: {count} traces stacked.")
            # Store the stack for later global normalization.
            comp_stacks[comp] = stack_shifted

            # Set a vertical offset for visual separation.
            if comp == "*Z":
                comp_offsets[comp] = 0
            elif comp == "*1":
                comp_offsets[comp] = vertical_offset
            elif comp == "*2":
                comp_offsets[comp] = 2 * vertical_offset
        else:
            print(f"No selected traces for stacking for component {comp} in Array {i}")

    # Compute a global normalization factor across all components (normalize after stacking).
    global_max = 0
    for comp in comp_stacks:
        this_max = np.max(np.abs(comp_stacks[comp]))
        if this_max > global_max:
            global_max = this_max
    if global_max == 0:
        global_max = 1

    # Build a common time axis (relative to pick).
    t_stack = np.linspace(start_time, end_time, nstack)

    # Plot each component's stack after normalizing by the same global factor.
    for comp in comp_stacks:
        normalized_stack = comp_stacks[comp] / global_max
        ax.plot(t_stack, normalized_stack * plot_mag + comp_offsets[comp], label=f"Shifted Stack {comp}")

    # Add vertical lines for predicted arrivals.
    if align_phase == "P":
        if 0 > start_time and 0 < end_time:
            ax.axvline(x=0, color="magenta", linestyle="--", lw=1, label="P predicted")
        if s_time_rel > start_time and s_time_rel < end_time:
            ax.axvline(x=s_time_rel, color="cyan", linestyle="--", lw=1, label="S predicted")
        if pmp_time_rel < end_time:
            ax.axvline(x=pmp_time_rel, color="green", linestyle="--", lw=1, label="PmP predicted")
        if sms_time_rel < end_time:
            ax.axvline(x=sms_time_rel, color="orange", linestyle="--", lw=1, label="SmS predicted")
    elif align_phase == "S":
        if 0 > start_time and 0 < end_time:
            ax.axvline(x=0, color="cyan", linestyle="--", lw=1, label="S predicted")
        if p_time_rel > start_time and p_time_rel < end_time:
            ax.axvline(x=p_time_rel, color="magenta", linestyle="--", lw=1, label="P predicted")
        if pmp_time_rel < end_time and pmp_time_rel > start_time:
            ax.axvline(x=pmp_time_rel, color="green", linestyle="--", lw=1, label="PmP predicted")
        if sms_time_rel < end_time and sms_time_rel > start_time:
            ax.axvline(x=sms_time_rel, color="orange", linestyle="--", lw=1, label="SmS predicted")

    # Set fixed vertical axis limits.
    ax.set_ylim(-1.5, 3.9)
    
    ax.set_xlabel("Time (s) relative to pick")
    ax.set_ylabel("Normalized Amplitude")
    if i == 1:
        ax.set_title(f"Array {i}: {origin_time_str}")
    else:
        ax.set_title(f"Array {i}")
    ax.legend(loc="upper right", fontsize="small")

fig.tight_layout()
save_path = os.path.join(save_dir, f"E{origin_time_str}.png")
fig.savefig(save_path)
print(f"    Saved figure for Array {i} to {save_path}")
plt.show()
