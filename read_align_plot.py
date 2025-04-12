#%% Read, align, reject bad traces, stack, and plot
# JV 3/29/2025, fixed by Ruoyan 4/1/2025

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
event_file = "Lists/List_tight.csv"
df = pd.read_csv(event_file)

# array_select = [5] # Select the arrays to process
channel     ="*Z" # Change to "*1" for N, "*2" for E, "*Z" for Z
array_select = [1, 2, 3, 4, 5] # Select the arrays to process
# array_select = [1] # Select the arrays to process

# Define offsets (in seconds) relative to the picked phase arrivel time
align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
start_time = -1   # seconds for start of analysis window
end_time   = 5   # seconds for end   of analysis window
duration = end_time - start_time
short_win_pre  =  0.25 # seconds for short cross-correlation window pre-pick
short_win_post =  0.25 # seconds for short cross-correlation window post-pick
min_freq   = 4 # Frequency range for bandpass filter
max_freq   = 40
plot_all_trace_traveltimes = False

# Assumes the CSV has columns "id", "time" (ISO formatted), "latitude", "longitude", and "depth"
evid = "ci40789071"
evt = df[df['id'] == evid].iloc[0]
origin_time_str = evt['time']  # e.g., "2024-11-07T08:39:06"
event_time = UTCDateTime(origin_time_str)
event_lat = float(evt['latitude'])
event_lon = float(evt['longitude'])
event_depth = float(evt['depth'])  # Event depth in km
print(f"Using event {evid} with origin time {event_time}")
print(f"Event location: lat={event_lat}, lon={event_lon}, depth={event_depth} km")

# Use the custom velocity model instead of iasp91
# model = TauPyModel(model="Lists/custom_model")
# Create a TauPy model (if you want to compute travel times later)
model = TauPyModel(model="iasp91")

# Calculated travel times
stations_file = "Lists/STAR_stations.csv"
stations_df = pd.read_csv(stations_file)

results1 = []  # List to accumulate station correlation results
results2 = []  # List to accumulate station correlation results
results3 = []  # List to accumulate station correlation results

# Define the directory to save figures
save_dir = "Plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List to store figure handles
fig_list = []

#%% Loop over arrays 1 to 5 (here only processing Array 1 for brevity)
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
    print(f"    1st station = {travel_times} s")

    # Extract the travel times and arrival times for P and S phases
    for tt in travel_times:
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
            p_arrival_time = event_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = tt.time
            s_arrival_time = event_time + s_traveltime

    # Redefine the fixed window relative to the P arrival of this array.
    # (For example, start 5 seconds before and end 15 seconds after P arrival)
    if   align_phase == "P":
        abs_pick         = p_arrival_time + start_time
        arrival_time     = p_arrival_time
        arrival_time_rel = p_traveltime
    elif align_phase == "S":
        abs_pick         = s_arrival_time + start_time
        arrival_time     = s_arrival_time
        arrival_time_rel = s_traveltime
    else:
        print(f"align phase improper")
        continue
    arrival_time_rel = start_time - p_traveltime

    # For plotting, compute the arrival times relative to the new fixed_start:

    # Read the data file for this array

    hr = abs_pick.hour
    day = abs_pick.julday
    year = abs_pick.year
    year_str = f"{year:04d}"
    day_str  = f"{day:03d}"
    hr_str   = f"{hr:02d}"
    input_file = f"/Volumes/STAR2/Array{i}/{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
    tr_name = f"Array {i} {year_str} day {day_str} hour {hr_str}"
    
    if not os.path.exists(input_file):
        print(f"File not found, skipping: {tr_name}")
        continue

    st = read(input_file)
    if len(st) > 0:
        first_station_trace = str(st[0].stats.station)
        # print("First station name from traces:", first_station_trace)

    print(f"    File read: {tr_name}")

    # Keep only traces with data in the fixed time window
    overlapping_traces = Stream()
    for tr in st:
        # print(f"Read trace with station: {tr.stats.station}")
        if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + duration:
            overlapping_traces.append(tr)
    if len(overlapping_traces) == 0:
        print(f"No data for time window in traces in file: {input_file}")
        continue

    # Slice these overlapping traces to the fixed time window
    st_window = overlapping_traces.slice(starttime=abs_pick, endtime=abs_pick + duration)

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
    npts = min(tr.stats.npts for tr in st_comp)

    # Extract the first trace to be the initial correlation target
    first_tr    = st_comp[0]
    sample_rate = st_comp[0].stats.sampling_rate
    ref = first_tr.data[:npts]
    t_axis = first_tr.times()[:npts]
    # Compute the plotting time range (common for both arrivals)

    if align_phase == "P":
        t_min = (start_time + p_traveltime)
    if align_phase == "S":
        t_min = (start_time + s_traveltime)
    t_max = t_min + duration

    pre_samples  = int(sample_rate * short_win_pre)
    post_samples = int(sample_rate * short_win_post)

    # --- Compute an aligned stack correlating with the first trace ---
    aligned_stack = np.zeros(npts)
    for tr in st_comp:
        d = tr.data[:npts] # Use only the first npts samples

        # limit to times that exist in case correlation window is too large
        win_start = int(max(0,    sample_rate * (-start_time - short_win_pre )))
        win_end   = int(min(npts, sample_rate * (-start_time + short_win_post)))

        # print(f"    win_start {win_start} win_end {win_end} short_win_pre {sample_rate * short_win_pre} short_win_post {sample_rate * short_win_post}")
        # print(f"    start_time {start_time} npts {npts}")
        ref_window = ref[win_start:win_end]
        d_window   = d[  win_start:win_end]
        # Compute unnormalized cross-correlation on the normalized segments
        corr = np.correlate(d_window, ref_window, mode="full")
        lag1 = np.argmax(corr) - (len(d_window) - 1)

        # apply correlation shifts and stack
        # aligned_data = np.roll(d, -lag1)
        aligned_data = np.roll(d, 0)
        aligned_stack += aligned_data

        # save 1st round of correlation shifts
        station_id = str(tr.stats.station) 
        result_dict1 = {
            "station": station_id,
            "lag1": lag1            }
            # "corr": np.max(corr)            }
        results1.append(result_dict1)

    # Normalize the aligned stack
    aligned_stack = aligned_stack / np.max(np.abs(aligned_stack))

    # --- Compute a "good" aligned stack omitting traces with low correlation ---
    selected_aligned_stack = np.zeros(npts)
    for tr in st_comp:
        d = tr.data[:npts]

        win_start = int(max(0,    sample_rate * (-start_time - short_win_pre )))
        win_end   = int(min(npts, sample_rate * (-start_time + short_win_post)))

        ref_window = aligned_stack[win_start:win_end]
        d_window   = d[  win_start:win_end]
        corr = np.correlate(d_window, ref_window, mode="full")
        lag2 = np.argmax(corr) - (len(d_window) - 1)
        # aligned_data = np.roll(d, -lag2)
        aligned_data = np.roll(d, 0)

        # save 1st round of correlation shifts
        station_id = str(tr.stats.station)
        result_dict2 = {
            "station": station_id,
            "lag2": lag2            }
        results2.append(result_dict2)
        
        # Compute correlation on the aligned window
        aligned_window = aligned_data[win_start:win_end]
        r_window = np.dot(aligned_window, ref_window) / (np.linalg.norm(aligned_window)* np.linalg.norm(ref_window))
        # and whole trace correlation
        r_whole =  np.dot(aligned_data,   ref)        / (np.linalg.norm(aligned_data)  * np.linalg.norm(ref))
        
        if r_window >= 0.3 and r_whole >= 0.1:
            selected_aligned_stack += aligned_data
        else:
            print(f"    Rejected trace {tr.stats.station} with window r {r_window:.2f} and whole r {r_whole:.2f}")
            
    # Normalize the selected aligned stack
    selected_aligned_stack = selected_aligned_stack / np.max(np.abs(selected_aligned_stack))

    # --- Define subplot layout for individual traces ---
    chunk_size = 10
    num_groups = math.ceil(num_traces / chunk_size)
    total_subplots = num_groups

    n_cols = 4
    n_rows = math.ceil(total_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    offset_unit = 1.0  # vertical spacing for individual traces

    p_label_val = p_traveltime
    s_label_val = s_traveltime

    for j, chunk_start in enumerate(range(0, num_traces, chunk_size)): # make each subplot
        ax = axes[j]
        chunk = st_comp[chunk_start:chunk_start + chunk_size]
        extra_offset = 0
        # --- Plotting the stacked traces (in the first subplot of each chunk) ---
        if j == 0:
            stack_offset = 1.0
            # # Shift the stacked trace time axis by subtracting the chosen alignment time
            # ax.plot(t_axis + start_time + p_traveltime, ref, color="red", lw=1, label="orig")
            # ax.plot(t_axis + start_time + p_traveltime, aligned_stack - stack_offset, color="green", lw=1, label="align")
            # ax.plot(t_axis + start_time + p_traveltime, selected_aligned_stack - 2*stack_offset, color="black", lw=1, label="good")
            # extra_offset = 2 * stack_offset

        # --- Plotting the individual station traces ---
        for idx, tr in enumerate(chunk):
            data = tr.data
            
            x = tr.data[:npts]
            x_window = x[win_start:win_end]
            ref_window = ref[win_start:win_end]
            
            corr = np.correlate(x_window, ref_window, mode="full")
            lag3 = np.argmax(corr) - (len(x_window) - 1)
            # aligned_data = np.roll(data, -lag3)
            aligned_data = np.roll(data, 0)
            aligned_window = aligned_data[win_start:win_end]
            if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
                r = 0
            else:
                r       = np.dot(aligned_window, ref_window) / (np.linalg.norm(aligned_window) * np.linalg.norm(ref_window))
            if np.linalg.norm(aligned_data)   == 0 or np.linalg.norm(ref)        == 0:
                r_whole = 0
            else:
                r_whole = np.dot(aligned_data,   ref)        / (np.linalg.norm(aligned_data)   * np.linalg.norm(ref))
            times = tr.times()[:len(aligned_data)]
            offset = - (idx + 1 + extra_offset) * offset_unit
            
            # Define station_id from the current trace as a string
            station_id = str(tr.stats.station)
            
            # Now look up station metadata using station_id, not currently used
            # calculate travel time for each station
            station_info = stations_df[stations_df['station'] == int(station_id)]
            st_lat = float(station_info.iloc[0]['latitude'])
            st_lon = float(station_info.iloc[0]['longitude'])
            elev = float(station_info.iloc[0]['elevation'])  # elevation in meters
            station_label = station_id[-2:]
            # Compute epicentral distance (in degrees) for this trace
            deg_trace = locations2degrees(event_lat, event_lon, st_lat, st_lon)
            arrival_delay = (p_arrival_i - event_time) if 'p_arrival_i' in locals() and p_arrival_i is not None else ""
            
            # Append results for the current trace:
            # Compute arrival delay in seconds (difference between hypocentral time and arrival time)
            result_dict3 = {
                "station": station_id,
                "distance_km": f"{deg_trace * 111.0:.4f}",
                "arrival_delay": f"{arrival_delay:.4f}" if arrival_delay != "" else "",
                "lag3": lag3            }
            results3.append(result_dict3)

            # --- Plotting each individual trace: subtract the chosen alignment time ---
            ax.plot(t_axis + start_time + p_traveltime, aligned_data + offset, label=station_label)
            ax.text((t_axis + start_time + p_traveltime)[0], aligned_data[0] + offset, f"{r:.2f}",
                    fontsize=8, color="blue", verticalalignment="top")
            ax.text((t_axis + start_time + p_traveltime)[-1], aligned_data[-1] + offset, f"{r_whole:.2f}",
                    fontsize=8, color="magenta", verticalalignment="top", horizontalalignment="left")
            
        # --- Mark the arrival times as vertical lines in each subplot ---

        # Plot the P arrival only if it lies within this range
        if t_min <= p_traveltime <= t_max:
            ax.axvline(x=p_traveltime, color="magenta", linestyle="--", lw=1)

        # Similar check for the S arrival
        if t_min <= s_traveltime <= t_max:
            ax.axvline(x=s_traveltime, color="cyan", linestyle="--", lw=1)
            
        ax.set_title(f"Traces {chunk_start+1} to {min(chunk_start+chunk_size, num_traces)}")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize="small")

    for idx in range(total_subplots, len(axes)):
        axes[idx].axis('off')

    # Later, when labeling the x-axis, update the label
    axes[-1].set_xlabel("Time (s) relative to P arrival")
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

    # After the stacks have been computed, add a separate frame for the three stacked traces:

    fig_stack, ax_stack = plt.subplots(figsize=(10, 6))
    stack_offset = 1.0  # same offset used in your main plot

    # Plot the original stack (target trace)
    ax_stack.plot(t_axis + start_time + p_traveltime, ref, color="red", lw=1, label="Original stack")
    # Plot the overall aligned stack (after first round of alignment)
    ax_stack.plot(t_axis + start_time + p_traveltime, aligned_stack - stack_offset, color="green", lw=1, label="Aligned stack")
    # Plot the "good" aligned stack (selected traces)
    ax_stack.plot(t_axis + start_time + p_traveltime, selected_aligned_stack - 2 * stack_offset, color="black", lw=1, label="Good stack")

    # Plot the P arrival only if it lies within this range
    if t_min <= p_traveltime <= t_max:
        ax_stack.axvline(x=p_traveltime, color="magenta", linestyle="--", lw=1)

    # Similar check for the S arrival
    if t_min <= s_traveltime <= t_max:
        ax_stack.axvline(x=s_traveltime, color="cyan", linestyle="--", lw=1)
            

    ax_stack.set_xlabel("Time (s) relative to earthquake origin")
    ax_stack.set_ylabel("Normalized Amplitude")
    ax_stack.set_title("Stacked Traces")
    ax_stack.legend(loc="upper right", fontsize="small")

    # Optionally save the figure:
    stack_fig_path = os.path.join(save_dir, f"Stacked_Traces_Array{i}.png")
    fig_stack.savefig(stack_fig_path)
    print(f"Saved stacked traces figure to {stack_fig_path}")

    # Define the cross-correlation window indices (based on the reference trace)
    win_start = int(max(0,    sample_rate * (-start_time - short_win_pre)))
    win_end   = int(min(npts, sample_rate * (-start_time + short_win_post)))

    # Extract the portions of the traces within this window
    ref_window_plot = ref[win_start:win_end]
    aligned_stack_window = aligned_stack[win_start:win_end]
    selected_aligned_stack_window = selected_aligned_stack[win_start:win_end]

    # For plotting, compute the corresponding time axis for the window
    t_axis_window = t_axis[win_start:win_end] + start_time + p_traveltime

    # Create a separate figure for the three aligned stacks in the short cross-correlation window
    fig_short, ax_short = plt.subplots(figsize=(10, 6))
    ax_short.plot(t_axis_window, ref_window_plot, color="red", lw=1, label="Original stack window")
    ax_short.plot(t_axis_window, aligned_stack_window, color="green", lw=1, label="Aligned stack window")
    ax_short.plot(t_axis_window, selected_aligned_stack_window, color="black", lw=1, label="Good stack window")

    ax_short.set_xlabel("Time (s) relative to P arrival")
    ax_short.set_ylabel("Normalized Amplitude")
    ax_short.set_title("Stacked Traces in Short Cross-correlation Window")
    ax_short.legend(loc="upper right", fontsize="small")

    # Optionally save the figure
    save_path_short = os.path.join(save_dir, f"Short_Window_Stacked_Traces_Array{i}.png")
    fig_short.savefig(save_path_short)
    print(f"Saved short window stacked traces figure to {save_path_short}")

    # plt.show()

output_file1 = "/Users/vidale/Documents/Research/STAR/Lists/output1_" + evid + ".csv"
with open(output_file1, "w", newline="") as f:
    fieldnames = ["station", "lag1"]
    # fieldnames = ["station", "lag1", "corr"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results1:
        writer.writerow(rec)

# print(f"Written station correlation results to {output_file1}")    

output_file2 = "/Users/vidale/Documents/Research/STAR/Lists/output2_" + evid + ".csv"
with open(output_file2, "w", newline="") as f:
    fieldnames = ["station", "lag2"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results2:
        writer.writerow(rec)

# print(f"Written station correlation results to {output_file2}")    

output_file3 = "/Users/vidale/Documents/Research/STAR/Lists/output3_" + evid + ".csv"
with open(output_file3, "w", newline="") as f:
    fieldnames = ["station", "distance_km", "arrival_delay", "lag3"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results3:
        writer.writerow(rec)

# print(f"Written station correlation results to {output_file3}")    
plt.show()
