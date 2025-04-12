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

channel     ="*Z" # Change to "*1" for N, "*2" for E, "*Z" for Z
align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
array_select = [5] # Select the arrays to process
# array_select = [1, 2, 3, 4, 5] # Select the arrays to process

# Define offsets (in seconds) relative to the event origin for the fixed window
start_time = -3   # seconds for start of analysis window
end_time   =  3   # seconds for end   of analysis window
short_win_pre  =  2 # seconds for short cross-correlation window pre-pick
short_win_post =  2 # seconds for short cross-correlation window post-pick
# Frequency range for bandpass filter
min_freq   = 2
max_freq   = 20

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

fixed_start = event_time + start_time
fixed_end   = event_time + end_time
plot_all_trace_traveltimes = False

#%% Calculated travel times
stations_file = "Lists/STAR_stations.csv"
stations_df = pd.read_csv(stations_file)

results1 = []  # List to accumulate station correlation results
results2 = []  # List to accumulate station correlation results
results3 = []  # List to accumulate station correlation results

#%% Extract hour, day (julian day), and year from fixed_start
hr = fixed_start.hour
day = fixed_start.julday
year = fixed_start.year

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
    first_station_id = i * 10000 + 1
    first_station_meta = stations_df[stations_df['station'] == first_station_id].iloc[0]
    # print(" 1st station:", first_station_meta)
    station_lat = float(first_station_meta['latitude'])
    station_lon = float(first_station_meta['longitude'])
    deg = locations2degrees(event_lat, event_lon, station_lat, station_lon)
    print(f"    Epicentral distance for Array {i}: {deg*111:.2f} km")

    # Compute travel times for both P and S phases using the first station
    travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                          distance_in_degree=deg,
                                          phase_list=["p", "s"])
                
    # Print out the computed travel times for the first station
    print(f"    1st station = {travel_times} s")

    p_traveltime = None
    s_traveltime = None
    for tt in travel_times:
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
        elif tt.phase.name.upper() == "S":
            s_traveltime = tt.time
    if p_traveltime is None:
        print(f"P time not found, skipping array {i}")
        continue
    if s_traveltime is None:
        print(f"S time not found, skipping array {i}")
        continue
    # Define P arrival time using the travel time computed for this array
    p_arrival = event_time + p_traveltime
    s_arrival = event_time + s_traveltime

    # Redefine the fixed window relative to the P arrival of this array.
    # (For example, start 5 seconds before and end 15 seconds after P arrival)
    if align_phase.upper() == "P":
        fixed_start = p_arrival + start_time
        fixed_end   = p_arrival + end_time
    else:
        fixed_start = s_arrival + start_time
        fixed_end   = s_arrival + end_time

    # For plotting, compute the arrival times relative to the new fixed_start:
    p_arrival_rel = p_arrival - fixed_start
    s_arrival_rel = s_arrival - fixed_start

    year_str = f"{year:04d}"
    day_str  = f"{day:03d}"
    hr_str   = f"{hr:02d}"
    input_file = f"/Volumes/STAR2/Array{i}/{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
    # input_file = f"{i}_{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
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
        if tr.stats.endtime >= fixed_start and tr.stats.starttime <= fixed_end:
            overlapping_traces.append(tr)
    if len(overlapping_traces) == 0:
        print(f"No data for time window in traces in file: {input_file}")
        continue

    # Slice these overlapping traces to the fixed time window
    st_window = overlapping_traces.slice(starttime=fixed_start, endtime=fixed_end)

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
    pre_samples  = int(sample_rate * short_win_pre)
    post_samples = int(sample_rate * short_win_post)

    # --- Compute an aligned stack correlating with the first trace ---
    aligned_stack = np.zeros(npts)
    for tr in st_comp:
        d = tr.data[:npts] # Use only the first npts samples

        # Find pick_idx based on the computed arrival time:
        if align_phase.upper() == "P":
            pick_idx = int((p_arrival - fixed_start) * sample_rate)
        else:  # For S alignment, use s_arrival instead:
            pick_idx = int((s_arrival - fixed_start) * sample_rate)

        # limit to times that exist in case correlation window is too large
        win_start = max(0, pick_idx - pre_samples)
        win_end   = min(npts, pick_idx + post_samples)

        ref_window = ref[win_start:win_end]
        d_window   = d[       win_start:win_end]
        # Compute unnormalized cross-correlation on the normalized segments
        corr = np.correlate(d_window, ref_window, mode="full")
        lag1 = np.argmax(corr) - (len(d_window) - 1)

        # apply correlation shifts and stack
        aligned_data = np.roll(d, -lag1)
        aligned_stack += aligned_data

        # save 1st round of correlation shifts
        station_id = str(tr.stats.station) 
        result_dict1 = {
            "station": station_id,
            "lag1": lag1            }
        results1.append(result_dict1)

    # Normalize the aligned stack
    aligned_stack = aligned_stack / np.max(np.abs(aligned_stack))

    # --- Compute a "good" aligned stack omitting traces with low correlation ---
    selected_aligned_stack = np.zeros(npts)
    for tr in st_comp:
        d = tr.data[:npts]
        win_start = max(0, pick_idx - pre_samples)
        win_end   = min(npts, pick_idx + post_samples)
        ref_window = ref[win_start:win_end]
        d_window = d[win_start:win_end]
        corr = np.correlate(d_window, ref_window, mode="full")
        lag2 = np.argmax(corr) - (len(d_window) - 1)
        aligned_data = np.roll(d, -lag2)

        # save 1st round of correlation shifts
        station_id = str(tr.stats.station)
        result_dict2 = {
            "station": station_id,
            "lag2": lag2            }
        results2.append(result_dict2)
        
        # Compute correlation on the aligned window
        aligned_window = aligned_data[win_start:win_end]
        r_window = np.dot(aligned_window, ref_window) / \
                    (np.linalg.norm(aligned_window)*np.linalg.norm(ref_window))
        # and whole trace correlation
        r_whole = np.dot(aligned_data, ref) / \
                    (np.linalg.norm(aligned_data)*np.linalg.norm(ref))
        
        if r_window >= 0.3 and r_whole >= 0.1:
            selected_aligned_stack += aligned_data
        # else:
        #     print(f"    Rejected trace {tr.stats.station} with window r {r_window:.2f} and whole r {r_whole:.2f}")
            
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

    # Determine the time shift for alignment based on the chosen phase
    if align_phase.upper() == "P":
        align_time = p_arrival_rel  # p_arrival_rel was computed relative to fixed_start
        # When aligning on P: P wave appears at zero; S wave is offset by:
        p_label_val = 0
        s_label_val = s_arrival_rel - p_arrival_rel
    else:
        align_time = s_arrival_rel
        # When aligning on S: S wave appears at zero; P wave appears at a negative time:
        s_label_val = 0
        p_label_val = p_arrival_rel - s_arrival_rel

    for j, chunk_start in enumerate(range(0, num_traces, chunk_size)): # make each subplot
        ax = axes[j]
        chunk = selected_aligned_stack[chunk_start:chunk_start + chunk_size]
        extra_offset = 0
        # --- Plotting the stacked traces (in the first subplot of each chunk) ---
        if j == 0:
            stack_offset = 1.0
            # Shift the stacked trace time axis by subtracting the chosen alignment time
            ax.plot(t_axis - align_time, ref, color="red", lw=1, label="orig")
            ax.plot(t_axis - align_time, aligned_stack - stack_offset, color="green", lw=1, label="align")
            ax.plot(t_axis - align_time, selected_aligned_stack - 2*stack_offset, color="black", lw=1, label="good")
            extra_offset = 2 * stack_offset
        for idx, tr in enumerate(chunk):
            data = tr.data
            
            # Choose the window center based on align_phase:
            if align_phase.upper() == "P" and p_arrival_rel is not None:
                # Convert P arrival relative to fixed_start to sample index
                p_idx = int(p_arrival_rel * sample_rate)
                win_center = p_idx
            elif align_phase.upper() == "S" and s_arrival_rel is not None:
                # Convert S arrival relative to fixed_start to sample index
                s_idx = int(s_arrival_rel * sample_rate)
                win_center = s_idx
            else:
                # Fallback: use the peak in the reference trace
                win_center = np.argmax(np.abs(ref_norm))
            
            # win_start = max(0, win_center - pre_samples)
            # win_end   = min(npts, win_center + post_samples)
            
            x = data[:npts]
            x_window = x[win_start:win_end]
            ref_window = ref_norm[win_start:win_end]
            
            corr = np.correlate(x_window, ref_window, mode="full")
            lag3 = np.argmax(corr) - (len(x_window) - 1)
            aligned_data = np.roll(data, 0)
            # aligned_data = np.roll(data, -lag3)
            aligned_window = aligned_data[win_start:win_end]
            if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
                r = 0
            else:
                r = np.dot(aligned_window, ref_window) / \
                    (np.linalg.norm(aligned_window) * np.linalg.norm(ref_window))
            if np.linalg.norm(aligned_data) == 0 or np.linalg.norm(ref_norm) == 0:
                r_whole = 0
            else:
                r_whole = np.dot(aligned_data, ref_norm) / \
                          (np.linalg.norm(aligned_data) * np.linalg.norm(ref_norm))
            times = tr.times()[:len(aligned_data)]
            offset = - (idx + 1 + extra_offset) * offset_unit
            
            # Define station_id from the current trace as a string
            station_id = str(tr.stats.station)
            
            # Now look up station metadata using station_id
            # calculate travel time for each station 
            station_info = stations_df[stations_df['station'] == int(station_id)]
            if not station_info.empty:
                st_lat = float(station_info.iloc[0]['latitude'])
                st_lon = float(station_info.iloc[0]['longitude'])
                elev = float(station_info.iloc[0]['elevation'])  # elevation in meters
                station_label = station_id[-2:]
                # Compute epicentral distance (in degrees) for this trace
                deg_trace = locations2degrees(event_lat, event_lon, st_lat, st_lon)
                # Compute travel times for both P and S phases for this station
                trace_travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                                            distance_in_degree=deg_trace,
                                                            phase_list=["p","P","s","S"])
                p_traveltime_i = None
                s_traveltime_i = None
                for tt in trace_travel_times:
                    if tt.phase.name.upper() == "P" and p_traveltime_i is None:
                        p_traveltime_i = tt.time
                    elif tt.phase.name.upper() == "S" and s_traveltime_i is None:
                        s_traveltime_i = tt.time
                if p_traveltime_i is None:
                    p_traveltime_i = 5  # fallback for P-wave
                if s_traveltime_i is None:
                    s_traveltime_i = 10  # fallback for S-wave
                
                # Calculate arrival times for this trace
                p_arrival_i = event_time + p_traveltime_i
                s_arrival_i = event_time + s_traveltime_i
                # Compute the arrival times relative to fixed_start for plotting
                p_arrival_rel_i = p_arrival_i - fixed_start
                s_arrival_rel_i = s_arrival_i - fixed_start
            else:
                print(f"Missing station info for station {station_id}")
                station_label = station_id[-2:]
                p_arrival_rel_i = None
                s_arrival_rel_i = None

            # (Optional) Plot vertical lines for the trace-specific arrivals.
            # These lines will be drawn for each trace and may overlap if traces share similar travel times.
            if plot_all_trace_traveltimes:
                if p_arrival_rel_i is not None:
                    ax.axvline(x=p_arrival_rel_i, color="magenta", linestyle="--", lw=1)
                if s_arrival_rel_i is not None:
                    ax.axvline(x=s_arrival_rel_i, color="cyan", linestyle="--", lw=1)

            # --- Plotting each individual trace: subtract the chosen alignment time ---
            ax.plot(times - align_time, aligned_data + offset, label=station_label)
            ax.text((times - align_time)[0], aligned_data[0] + offset, f"{r:.2f}",
                    fontsize=8, color="blue", verticalalignment="top")
            ax.text((times - align_time)[-1], aligned_data[-1] + offset, f"{r_whole:.2f}",
                    fontsize=8, color="magenta", verticalalignment="top", horizontalalignment="left")
            
            # Append results for the current trace:
            # Compute station distance (in km) from epicentral distance (deg_trace)
            station_distance_km = deg_trace * 111.0
            # Compute arrival delay in seconds (difference between hypocentral time and arrival time)
            arrival_delay = (p_arrival_i - event_time) if 'p_arrival_i' in locals() and p_arrival_i is not None else ""
            result_dict3 = {
                "station": station_id,
                "distance_km": f"{station_distance_km:.4f}",
                "arrival_delay": f"{arrival_delay:.4f}" if arrival_delay != "" else "",
                "lag3": lag3            }
            results3.append(result_dict3)
            
        # --- Mark the arrival times as vertical lines in each subplot ---
        if j == 0:  # Only add labels to the first subplot
            ax.axvline(x=p_label_val, color="magenta", linestyle="--", lw=1, label="P arrival")
            ax.axvline(x=s_label_val, color="cyan", linestyle="--", lw=1, label="S arrival")
        else:
            ax.axvline(x=p_label_val, color="magenta", linestyle="--", lw=1)
            ax.axvline(x=s_label_val, color="cyan", linestyle="--", lw=1)
            
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

output_file1 = "/Users/vidale/Documents/Research/STAR/Lists/output1_" + evid + ".csv"
with open(output_file1, "w", newline="") as f:
    fieldnames = ["station", "lag1"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results1:
        writer.writerow(rec)

print(f"Written station correlation results to {output_file1}")    

output_file2 = "/Users/vidale/Documents/Research/STAR/Lists/output2_" + evid + ".csv"
with open(output_file2, "w", newline="") as f:
    fieldnames = ["station", "lag2"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results2:
        writer.writerow(rec)

print(f"Written station correlation results to {output_file2}")    

output_file3 = "/Users/vidale/Documents/Research/STAR/Lists/output3_" + evid + ".csv"
with open(output_file3, "w", newline="") as f:
    fieldnames = ["station", "distance_km", "arrival_delay", "lag3"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in results3:
        writer.writerow(rec)

print(f"Written station correlation results to {output_file3}")    
plt.show()
