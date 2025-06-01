#%% plots all traces and compares shifted and unshifted stack for one component
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
plot_all_traces = True

channel     = "*Z" # "*1" for N, "*2" for E, "*Z" for Z
# array_select = [1]
array_select = [1, 2, 3, 4, 5] # Select the arrays to process

# Define offsets (in seconds) relative to the picked phase arrival time
align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
start_time = -20   # seconds relative to pick, for start of analysis window
end_time   = 20   # seconds relative to pick, for end   of analysis window

min_freq   = 2 # Frequency range for bandpass filter
max_freq   = 8

# Hypocenter, the CSV has columns "id", "time" (ISO formatted), "latitude", "longitude", and "depth"
evid = "ci40852279" #"ci40789071"
evt = df[df['id'] == evid].iloc[0]
origin_time_str = evt['time']  # e.g., "2024-11-07T08:39:06"
event_time = UTCDateTime(origin_time_str)
event_lat = float(evt['latitude'])
event_lon = float(evt['longitude'])
event_depth = float(evt['depth'])  # Event depth in km
print(f"Event {evid} with origin time {event_time} lat={event_lat}, lon={event_lon}, depth={event_depth} km")

# Load the TauPyModel for travel time calculations
model = TauPyModel(model="iasp91")

# Load station metadata
stations_file = "Lists/STAR_stations.csv"
stations_df = pd.read_csv(stations_file)

# Load station statics
# data_file = "Lists/output3_" + evid + ".csv"
data_file = "Lists/Pstatics.csv"
data_df = pd.read_csv(data_file)

# Define the directory to save figures
save_dir = "Plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List to store figure handles
fig_list = []

#%% Loop over arrays
for i in array_select:
    print(f"Processing Array {i}")
    # Compute epicentral distance in degrees between event and the first station of the array
    # Stations are so dense we'll assume they are colocated after aligning the chosen phase
    first_station_id = i * 10000 + 1
    first_station_meta = stations_df[stations_df['station'] == first_station_id].iloc[0]
    # print(" 1st station:", first_station_meta)
    station_lat = float(first_station_meta['latitude'])
    station_lon = float(first_station_meta['longitude'])
    deg = locations2degrees(event_lat, event_lon, station_lat, station_lon)
    print(f"    Epicentral distance: {deg*111:.2f} km")

    # Compute travel times using the first station
    travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                          distance_in_degree=deg,
                                          phase_list=["p", "P", "s", "S"])
    print(f"    1st station = {travel_times}")

    # Extract the travel times and arrival times for P and S phases
    for tt in travel_times:
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
            p_arrival_time = event_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = tt.time
            s_arrival_time = event_time + s_traveltime

    # Define the fixed window relative to the chosen arrival
    if   align_phase == "P":
        abs_pick   = p_arrival_time + start_time
        rel_pick   = p_arrival_time
        s_time_rel = s_traveltime - p_traveltime
        p_time_rel = 0
    elif align_phase == "S":
        abs_pick   = s_arrival_time + start_time
        rel_pick   = s_arrival_time
        s_time_rel = 0
        p_time_rel = p_traveltime - s_traveltime
    else:
        print(f"Did not find chosen arrival")
        continue

    # Read the data file for this array
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
    if len(st) > 0:
        first_station_trace = str(st[0].stats.station)
        print("First station:", first_station_trace)

    print(f"    File read: {input_file}")

    # Keep only traces with data in the fixed time window
    overlapping_traces = Stream()
    for tr in st:
        # print(f"Read trace with station: {tr.stats.station}")
        if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + end_time - start_time:
            overlapping_traces.append(tr)
    if len(overlapping_traces) == 0:
        print(f"No data for time window in traces in file: {input_file}")
        continue

    # Slice these overlapping traces to the fixed time window
    st_window = overlapping_traces.slice(starttime = abs_pick, endtime = abs_pick + end_time - start_time)

    # Which component, e.g. "*Z", "*1", or "*2"
    st_comp = st_window.select(channel=channel)
    num_traces = len(st_comp)
    print(f"    Array {i}: {num_traces} traces in {channel} channel")
    if num_traces == 0:
        print(f"No component traces in Array {i} in file {input_file}")
        continue

    # Demean, taper, and filter each trace after slicing the window
    for tr in st_comp:
        tr.detrend(type="demean")
        tr.taper(max_percentage=0.05, type="cosine")
        tr.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=4, zerophase=False)
        tr.data = tr.data / np.max(np.abs(tr.data))
    
    # Get parameters of data from first trace
    sample_rate = st_comp[0].stats.sampling_rate
    npts = min(tr.stats.npts for tr in st_comp)
    t_axis = st_comp[0].times()[:npts]

    # ----- Compute and plot stacks for unshifted and shifted traces -----
    nstack = int(sample_rate * (end_time - start_time))
    stack_unshifted = np.zeros(nstack)
    stack_shifted   = np.zeros(nstack)
    count = 0
    for tr in st_comp:
        # Get the unshifted, cut segment from the trace
        unshifted = tr.data[:nstack]
        
        # Retrieve lag information (from your data_df)
        lag = 0
        # data_info = data_df[data_df['station'] == int(tr.stats.station)]
        # if data_info.empty:
        #     select = True
        # else:
        #     lag = int(data_info.iloc[0]['lag3'])
        #     select = bool(data_info.iloc[0]['select'])
        # if not select:
        #     continue

        # Check if lag is within reasonable bounds; otherwise use zero
        if lag > 10 or lag < -10:
            lag = 0

        # Compute the shifted version of the cut trace.
        shifted = np.roll(unshifted, -lag)
        
        # Accumulate the stacks
        stack_unshifted += unshifted
        stack_shifted   += shifted
        count += 1

    if count > 0:
        # Compute the average stack for each
        stack_unshifted /= count
        stack_shifted   /= count

        # Build a time axis relative to the pick for the stack plot.
        # Here we assume the analysis window runs from start_time to end_time.
        t_stack = np.linspace(start_time, end_time, nstack)

        # Create a new figure for the two stacks.
        fig_stack, ax_stack = plt.subplots(figsize=(10, 6))
        ax_stack.plot(t_stack, stack_unshifted, label="Unshifted Stack", color="blue")
        ax_stack.plot(t_stack, stack_shifted, label="Shifted Stack", color="red")
        ax_stack.set_xlabel("Time (s) relative to pick")
        ax_stack.set_ylabel("Amplitude")
        ax_stack.set_title(f"Array {i}: Unshifted vs. Shifted Stacks")
        ax_stack.legend(loc="upper right", fontsize="small")
        
        # Optionally, save the new stack figure.
        stack_fig_path = os.path.join(save_dir, f"Array{i}_Stacks.png")
        fig_stack.savefig(stack_fig_path)
        print(f"    Saved stacked figure for Array {i} to {stack_fig_path}")
        # plt.close(fig_stack)  # close the figure if needed
        
    else:
        print(f"    No selected traces for stacking in Array {i}")

    if plot_all_traces:
        # --- Define subplot layout for individual traces ---
        chunk_size = 10
        num_groups = math.ceil(num_traces / chunk_size)
        total_subplots = num_groups

        n_cols = 4
        n_rows = math.ceil(total_subplots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        offset_unit = 1.0  # vertical spacing for individual traces

        # Define the pick reference so that zero corresponds to the pick
        pick_ref = p_traveltime

        for j, chunk_start in enumerate(range(0, num_traces, chunk_size)): # make each subplot
            ax = axes[j]
            chunk = st_comp[chunk_start:chunk_start + chunk_size]
            extra_offset = 0

            # --- Plotting the individual station traces ---
            for idx, tr in enumerate(chunk):
                data = tr.data

                # Define station_id from the current trace as a string
                station_id = str(tr.stats.station)

                ## Get lag data
                data_info = data_df[data_df['station'] == int(station_id)]
                if data_info.empty:
                    lag = 0
                    select = True
                else:
                    lag = int(data_info.iloc[0]['lag3'])
                    select = bool(data_info.iloc[0]['select'])

                if select == False:
                    continue
                
                if lag >10 or lag < -10:
                    lag = int(0)

                aligned_data = np.roll(data, -lag)
                cut_data = aligned_data[0:int(sample_rate * (end_time - start_time))]

                times = tr.times()[:len(cut_data)]
                offset = - (idx + 1 + extra_offset) * offset_unit
                
                # Now look up station metadata using station_id to calculate travel time for each station, not currently used
                station_info = stations_df[stations_df['station'] == int(station_id)]
                st_lat = float(station_info.iloc[0]['latitude'])
                st_lon = float(station_info.iloc[0]['longitude'])
                elev =   float(station_info.iloc[0]['elevation'])  # elevation in meters
                
                station_label = station_id[-2:]
                
                # --- Plotting each individual trace: subtract the chosen alignment time ---
                ax.plot(times + start_time, cut_data + offset, label=station_label)
                
            # Mark the P and S arrivals
            if start_time <= p_time_rel <= end_time:
                ax.axvline(x=p_time_rel, color="magenta", linestyle="--", lw=1)
            if start_time <= s_time_rel <= end_time:
                ax.axvline(x=s_time_rel, color="cyan", linestyle="--", lw=1)
                
            ax.set_title(f"Traces {chunk_start+1} to {min(chunk_start+chunk_size, num_traces)}")
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.legend(loc="upper right", fontsize="small")

        for idx in range(total_subplots, len(axes)):
            axes[idx].axis('off')

        # Later, when labeling the x-axis, update the label
        axes[-1].set_xlabel("Time (s) relative to pick")
        # Include event date and time in the overall plot title
        plt.suptitle(f"Array {i} component {channel[1]}  Stacks + aligned arrivals\nEvent: {event_time.isoformat()}", fontsize=14)
        plt.tight_layout(pad=2.0)   
        plt.subplots_adjust(top=0.90)
        
        common_ylim = axes[0].get_ylim()
        for j in range(total_subplots):
            axes[j].set_ylim(common_ylim)
        
        save_path = os.path.join(save_dir, f"Array{i}_component_seismograms.png")
        fig.savefig(save_path)
        print(f"    Saved figure for Array {i} to {save_path}")
        fig_list.append(fig)

plt.show()