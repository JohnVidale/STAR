#%% Read, reject bad traces, align by cc, save statics, stack, 
# and plot indfividual and stacked traces
# JV 3/29/2025, fixed by Ruoyan 4/1/2025, modified by Keisuke 4/25, enhanced by J 5/2025

from obspy import read, Stream, UTCDateTime
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from obspy.geodetics import locations2degrees
import csv
from itertools import chain
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")

# Select event and load parameters
evid         = "ci41034048" #"ci40789071"
channel      = "*2" # Change to "*1" for N, "*2" for E, "*Z" for Z
align_phase  = "S" # "P" for P-wave, "S" for S-wave alignment
array_select = [1, 2, 3, 4, 5] # Select the arrays to process
# array_select = [2] # Select the arrays to process
be_choosy = False  # discard traces with select = False

# Define offsets (in seconds) relative to the picked phase arrivel time
# Long window for plotting, seconds
start_time = -1 
end_time   =  2
duration = end_time - start_time
# Short window for cross-correlation, seconds
short_win_start = 0
short_win_end   = 1
lag_range = 10 # +/- Range of lags to search for in samples, time samples
#Frequency filter, Hz
min_freq   = 1
max_freq   = 5

# Specify directories
data_dir = "/Volumes/STAR2/"
work_dir = "/Users/vidale/Documents/Research/STAR/"
list_dir      = work_dir + "Lists/"
plot_dir      = work_dir + "Plots/"
results_dir   = work_dir + "Results/"
events_file    = work_dir + "Lists/List_all.csv"
stations_file = work_dir + "Lists/STAR_stations.csv"
data_file     = work_dir + "Lists/statics.csv"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the event list CSV
df = pd.read_csv(events_file)
# Load station locations and enter array centers
stations_df = pd.read_csv(stations_file)
lat_center = [  33.609800,   33.483522,   33.327171,   33.373449,   33.473353] # use element 41 for center
lon_center = [-116.454500, -116.151843, -116.366194, -116.62345, -116.646267]
# Load station statics
data_df = pd.read_csv(data_file)
# Load hypocenter
evt = df[df['id'] == evid].iloc[0]
origin_time_str = evt['time']  # e.g., "2024-11-07T08:39:06"
event_time = UTCDateTime(origin_time_str)
event_lat = float(evt['latitude'])
event_lon = float(evt['longitude'])
event_depth = float(evt['depth'])  # Event depth in km
print(f"Using event {evid} with origin time {event_time}")
print(f"Event location: lat={event_lat}, lon={event_lon}, depth={event_depth} km")

results0 = []  # Lists to accumulate station correlation results
results1 = []
results2 = []
results3 = []
results4 = []

def compute_lag(ref, d, win_start, win_end, lag_range):
    ref_window = ref[win_start:win_end]
    d_window   = d[win_start:win_end]
    corr = np.correlate(d_window, ref_window, mode="full")
    center = len(d_window) - 1
    # Restrict search to lags in [center - 10, center + 10]
    search_corr = corr[center - lag_range : center + lag_range + 1]
    best_index = np.argmax(search_corr)
    # Calculate lag relative to zero lag offset
    lag = (best_index + center - lag_range) - center
    return lag

#%% Loop over specified arrays
for i in array_select:
    print(f"Processing Array {i}")

    # Compute epicentral distance in degrees between event and the central station of the array
    center_deg = locations2degrees(event_lat, event_lon, lat_center[i-1], lon_center[i-1])

    # Compute travel times to the central station
    travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                          distance_in_degree=center_deg,
                                          phase_list=["p", "P", "s", "S"])
    print(f"    Central station: {travel_times} s    Epicentral distance: {center_deg*111:.2f} km")

    # Adopt the arrival times for earliest P and S phases
    for tt in reversed(travel_times):
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
            p_arrival_time = event_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = tt.time
            s_arrival_time = event_time + s_traveltime

    # Redefine the fixed window relative to the chosen arrival of this array.
    # (For example, start 5 seconds before and end 15 seconds after P arrival)
    if   align_phase == "P":
        abs_pick         = p_arrival_time + start_time
        arrival_time     = p_arrival_time
        arrival_time_rel = start_time - p_traveltime   # 0 is P-wave traveltime after the hypocentral time
    elif align_phase == "S":
        abs_pick         = s_arrival_time + start_time
        arrival_time     = s_arrival_time
        arrival_time_rel = start_time - s_traveltime   # 0 is S-wave traveltime after the hypocentral time
    else:
        print(f"improper alignment phase {align_phase}, not P or S")
        continue

    # Time range for plotting (for both stations and stacked traces)
    if align_phase == "P":
        t_min = (start_time + p_traveltime)
    if align_phase == "S":
        t_min = (start_time + s_traveltime)
    t_max = t_min + duration

    # Find seismogram file
    hr = abs_pick.hour
    day = abs_pick.julday
    year = abs_pick.year
    year_str = f"{year:04d}"
    day_str  = f"{day:03d}"
    hr_str   = f"{hr:02d}"
    input_file = f"{data_dir}Array{i}/{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
    tr_name = f"Array {i} {year_str} day {day_str} hour {hr_str}"
    if not os.path.exists(input_file):
        print(f"Seismogram file not found, skipping: {tr_name}")
        exit(-1)

    # Read seismogram file
    st = read(input_file)
    if len(st) > 0:
        first_station_trace = str(st[0].stats.station)
        # print("First station name from traces:", first_station_trace)

    print(f"    File read: {tr_name}")

    # Keep only traces with data in the fixed time window
    full_traces = Stream()
    for tr in st:
        # Define station_id from the current trace as a string
        station_id = int(tr.stats.station)

        # Reject stations listed as sometimes problematic
        data_info = data_df[data_df['station'] == int(station_id)]
        if data_info.empty:
            print(f"Station {station_id} not in station file!")
            exit(-1)
        select = bool(data_info.iloc[0]['select'])
        if be_choosy and select == False:
            continue

        if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + duration:
            full_traces.append(tr)
        else:
            print(f"Entire trace {tr.stats.starttime} to {tr.stats.endtime} not in time window: {abs_pick} to {abs_pick + duration}")
            print(f"Maybe crossing hour mark, maybe enhance code to handle this")
            exit(-1)
    if len(full_traces) == 0:
        print(f"No data for time window in traces in file: {input_file}")
        exit(-1)

    # Cull component, "*Z", "*1", or "*2"
    st_full_chan = full_traces.select(channel=channel)

    # Demean, taper, and filter each full trace
    for tr in st_full_chan:
        tr.detrend(type="demean")
        tr.taper(max_percentage=0.05, type="cosine")
        tr.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=4, zerophase=False)

    # Slice the plotting windows from the full traces
    st_comp = st_full_chan.slice(starttime=abs_pick, endtime=abs_pick + duration)

    num_traces = len(st_comp)
    print(f"    Array {i}: {num_traces} traces in specified {channel} channel")
    if num_traces == 0:
        print(f"No component traces in Array {i} in file {input_file}")
        exit(-1)

    # Normalize each trace after slicing the window
    for tr in st_comp:
        tr.data = tr.data / np.max(np.abs(tr.data))
    npts = min(tr.stats.npts for tr in st_comp)

    # Extract a central trace as initial correlation target
    central_tr = st_comp[40]
    sample_rate = central_tr.stats.sampling_rate
    t_axis = central_tr.times()[:npts]
    ref_init = central_tr.data[:npts]

    # Define short window for cross-correlation
    win_start = int(max(0,    sample_rate * (-start_time + short_win_start)))
    win_end   = int(min(npts, sample_rate * (-start_time + short_win_end)))
    ref_init_window = ref_init[win_start:win_end]
    ref_init_window = ref_init_window / np.max(np.abs(ref_init_window))

    # Shift by P or S phase moveout
    move_out_st = Stream()
    for tr in st_comp:
        # Compute the lag for the central trace
        station_id = tr.stats.station
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

        travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                            distance_in_degree=station_deg,
                                            phase_list=["p", "P", "s", "S", "PmP", "SmS"])

        # Extract the travel times and arrival times for P and S phases
        # Travel times are checked in reverse order so that the first arrival can be chosen.
        for tt in reversed(travel_times):
            if tt.phase.name.upper() == "P":
                p_diff = tt.time - p_traveltime #array center time minus time for this station
                p_slowness_sk = tt.ray_param / 111.19
            if tt.phase.name.upper() == "S":
                s_diff = tt.time - s_traveltime
                s_slowness_sk = tt.ray_param / 111.19

        if   align_phase == "P":
            shift = p_diff
        elif align_phase == "S":
            shift = s_diff
        lag = int(shift * sample_rate)

        # Find correlation before moveout correction
        d = tr.data[:npts]
        d_win = d[win_start:win_end]
        r = np.dot(d_win, ref_init_window) / (np.linalg.norm(d_win) * np.linalg.norm(ref_init_window))
        result_dict0 = {
            "station": station_id,
            "lag_moveout": lag,
            "corr_start": f"{r:.3f}",
        }
        results1.append(result_dict0)

        shifted_data = np.roll(d, -lag)
        tr_new = tr.copy()
        tr_new.data = shifted_data
        move_out_st.append(tr_new)

    # Align correlating with the central trace and then stack
    aligned_stack = np.zeros(npts)
    shifted_st = Stream()
    for tr in move_out_st:
        d = tr.data[:npts]

        # Correlation before shift 1
        d_win = d[win_start:win_end]
        r = np.dot(d_win, ref_init_window) / (np.linalg.norm(d_win) * np.linalg.norm(ref_init_window))

        # Compute unnormalized cross-correlation on the normalized segments
        lag1 = compute_lag(ref_init, d, win_start, win_end, lag_range)

        shifted_data = np.roll(tr.data, -lag1)
        tr_new = tr.copy()
        tr_new.data = shifted_data
        shifted_st.append(tr_new)

        # apply correlation shifts and stack
        aligned_data = np.roll(d, -lag1)
        aligned_stack += aligned_data

        # save 1st round of correlation shifts
        station_id = str(tr.stats.station) 
        result_dict1 = {
            "station": station_id,
            "lag1": lag1,
            "corr0": f"{r:.3f}",
        }
        results1.append(result_dict1)

    # Window and normalize the aligned stack
    aligned_stack_window = aligned_stack[win_start:win_end]
    aligned_stack        = aligned_stack        / np.max(np.abs(aligned_stack))
    aligned_stack_window = aligned_stack_window / np.max(np.abs(aligned_stack_window))

    # Align with the stack this time, rather than the central trace
    aligned_restack = np.zeros(npts)
    reshifted_st = Stream()
    for tr in shifted_st:
        d = tr.data[:npts]

        # Correlation before shift 2
        d_win = d[win_start:win_end]
        r = np.dot(d_win, aligned_stack_window) / (np.linalg.norm(d_win) * np.linalg.norm(aligned_stack_window))

        # Compute unnormalized cross-correlation on the normalized segments
        lag2 = compute_lag(aligned_stack, d, win_start, win_end, lag_range)

        reshifted_data = np.roll(tr.data, -lag2)
        tr_new = tr.copy()
        tr_new.data = reshifted_data
        reshifted_st.append(tr_new)

        realigned_data = np.roll(d, -lag2)
        aligned_restack += realigned_data
            
        # save 2nd round of correlation shifts
        station_id = str(tr.stats.station)
        result_dict2 = {
            "station": station_id,
            "lag2": lag2,
            "corr1": f"{r:.3f}",
        }
        results2.append(result_dict2)

    # Window and normalize the aligned restack
    aligned_restack_window = aligned_restack[win_start:win_end]
    aligned_restack        = aligned_restack        / np.max(np.abs(aligned_restack))
    aligned_restack_window = aligned_restack_window / np.max(np.abs(aligned_restack_window))

    # Align with a refined stack this 3rd time
    aligned_rerestack = np.zeros(npts)
    rereshifted_st = Stream()
    for tr in reshifted_st:
        d = tr.data[:npts]

        # Correlation before shift 3
        d_win = d[win_start:win_end]

        # Compute unnormalized cross-correlation on the normalized segments
        lag3 = compute_lag(aligned_restack, d, win_start, win_end, lag_range)
        
        r       = np.dot(d_win, aligned_restack_window) / (np.linalg.norm(d_win) * np.linalg.norm(aligned_restack_window))
        r_whole = np.dot(d,     aligned_restack)        / (np.linalg.norm(d)     * np.linalg.norm(aligned_restack))

        # save 3rd round of correlation shifts
        # I know it's not the final corr iteration, but this is an easy option
        if (r > 0.85 and r_whole > 0.5) or not be_choosy:

            rereshifted_data = np.roll(tr.data, -lag3)
            tr_new = tr.copy()
            tr_new.data = rereshifted_data
            rereshifted_st.append(tr_new)

            rerealigned_data = np.roll(d, -lag3)
            aligned_rerestack += rerealigned_data
            station_id = str(tr.stats.station)
            result_dict3 = {
                "station": station_id,
                "lag3": lag3,
                "corr2": f"{r:.3f}",
            }
            results3.append(result_dict3)

    # Window and normalize the aligned restack
    aligned_rerestack_window = aligned_rerestack[win_start:win_end]
    aligned_rerestack        = aligned_rerestack        / np.max(np.abs(aligned_rerestack))
    aligned_rerestack_window = aligned_rerestack_window / np.max(np.abs(aligned_rerestack_window))

    # --- Define subplot layout for individual traces ---
    final_num_traces = len(rereshifted_st)
    chunk_size = 10
    num_groups = math.ceil(final_num_traces / chunk_size)
    total_subplots = num_groups

    n_cols = 4
    n_rows = math.ceil(total_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    offset_unit = 1.0  # vertical spacing for individual traces

    p_label_val = p_traveltime
    s_label_val = s_traveltime

    for j, chunk_start in enumerate(range(0, final_num_traces, chunk_size)): # make each subplot
        ax = axes[j]
        chunk = rereshifted_st[chunk_start:chunk_start + chunk_size]
        extra_offset = 0
        # --- Plotting the stacked traces (in the first subplot of each chunk) ---
        if j == 0:
            stack_offset = 1.0

        # --- Plotting the individual station traces ---
        for idx, tr in enumerate(chunk):
            d = tr.data
            d_win = d[win_start:win_end]

            # Define station_id from the current trace as a string
            station_id = str(tr.stats.station)

            r       = np.dot(d_win, aligned_rerestack_window) / (np.linalg.norm(d_win) * np.linalg.norm(aligned_rerestack_window))
            r_whole = np.dot(d,     aligned_rerestack)        / (np.linalg.norm(d)     * np.linalg.norm(aligned_rerestack))

            # Now look up station metadata using station_id, not currently used
            # calculate travel time for each station
            station_info = stations_df[stations_df['station'] == int(station_id)]
            st_lat = float(station_info.iloc[0]['latitude'])
            st_lon = float(station_info.iloc[0]['longitude'])
            elev = float(station_info.iloc[0]['elevation'])  # elevation in meters
            station_label = station_id[-2:]
            # Compute epicentral distance (in degrees) for this trace
            deg_trace = locations2degrees(event_lat, event_lon, st_lat, st_lon)
            
            # Append results for the current trace:
            # Compute arrival delay in seconds (difference between hypocentral time and arrival time)
            result_dict4 = {
                "station": station_id,
                "distance_km": f"{deg_trace * 111.0:.4f}",
                "corr": f"{r:.3f}",
                "corr_long": f"{r_whole:.3f}"
            }
            results4.append(result_dict4)
            
            # --- Plotting each individual trace: subtract the chosen alignment time ---
            offset = - (idx + 1 + extra_offset) * offset_unit
            ax.plot(t_axis + start_time, d + offset, label=station_label)
            ax.text((t_axis + start_time)[0], offset + 0.7, f"{r:.2f}  {r_whole:.2f}",
                    fontsize=8, color="blue", verticalalignment="top")
        # --- Mark the arrival times as vertical lines in each subplot ---


        # Add thin black vertical lines marking the short window boundaries:
        ax.axvline(x=short_win_start, color="black", linestyle=":", lw=1)
        ax.axvline(x=short_win_end, color="black", linestyle=":", lw=1)
        # Plot the P arrival only if it lies within this range
        if align_phase == "P":
            if start_time <= 0 <= start_time + duration:
                ax.axvline(x=0, color="magenta", linestyle="-", lw=1)
            if start_time <= (s_traveltime - p_traveltime) <= start_time + duration:
                ax.axvline(x=(s_traveltime - p_traveltime), color="cyan", linestyle="-", lw=1)
        if align_phase == "S":
            if start_time <= (p_traveltime - s_traveltime) <= start_time + duration:
                ax.axvline(x=(p_traveltime - s_traveltime), color="magenta", linestyle="-", lw=1)
            if start_time <= 0 <= start_time + duration:
                ax.axvline(x=0, color="cyan", linestyle="-", lw=1)

        ax.set_title(f"Phase {align_phase} Channel {channel} Array {i} Traces {chunk_start+1} to {min(chunk_start+chunk_size, num_traces)}")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize="small")

    for idx in range(total_subplots, len(axes)):
        axes[idx].axis('off')

    # Later, when labeling the x-axis, update the label
    axes[-1].set_xlabel("Time (s) relative to P arrival")
    # Include event date and time in the overall plot title
    plt.suptitle(f"Array {i} comp. {channel[1]}  Stacks + {align_phase} aligned arrivals\nEvent: {event_time.isoformat()}", fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.90)
    
    common_ylim = axes[0].get_ylim()
    for j in range(total_subplots):
        axes[j].set_ylim(common_ylim)
    
    # Save the figure to a file
    save_path = os.path.join(plot_dir, f"Array{i}_seismograms.png")
    fig.savefig(save_path)
    # print(f"    Saved figure for Array {i} to {save_path}")
    # fig_list.append(fig)

    # After the stacks have been computed, add a separate frame for the three stacked traces:

    fig_stack, ax_stack = plt.subplots(figsize=(10, 6))
    stack_offset = 1.0  # same offset used in your main plot

    # --- Full stacked trace plot ---
    ax_stack.plot(t_axis + start_time, ref_init, color="red", lw=1, label="Central station")
    ax_stack.plot(t_axis + start_time, aligned_stack - stack_offset, color="green", lw=1, label="Aligned stack")
    ax_stack.plot(t_axis + start_time, aligned_restack - 2 * stack_offset, color="black", lw=1, label="Good stack")

    # Add thin black vertical lines at short_win_start and short_win_end on the full stacked plot
    ax_stack.axvline(x=short_win_start, color="black", linestyle=":", lw=1)
    ax_stack.axvline(x=short_win_end, color="black", linestyle=":", lw=1)
    
    # Plot the P arrival only if it lies within this range
    if align_phase == "P":
        if start_time <= 0 <= start_time + duration: # P arrival
            ax_stack.axvline( x = 0, color="magenta", linestyle="-", lw=1)
        if start_time <= (s_traveltime - p_traveltime) <= start_time + duration: # S arrival
            ax_stack.axvline( x = (s_traveltime - p_traveltime), color="cyan", linestyle="-", lw=1)

    if align_phase == "S":
        print (f"t_min: {t_min}, t_max: {t_max}, p_traveltime: {p_traveltime}, s_traveltime: {s_traveltime}")
        if start_time <= (p_traveltime - s_traveltime) <= start_time + duration: # P arrival
            ax_stack.axvline(x = (p_traveltime - s_traveltime), color="magenta", linestyle="-", lw=1)
        if start_time <= 0 <= start_time + duration: # S arrival
            ax_stack.axvline(x=0, color="cyan", linestyle="-", lw=1)

    ax_stack.set_xlabel("Time (s) relative to earthquake origin")
    ax_stack.set_ylabel("Normalized Amplitude")
    ax_stack.set_title(f"Phase {align_phase} Channel {channel} Array {i} Stacked Traces")
    ax_stack.legend(loc="upper right", fontsize="small")

    # Save the figure:
    stack_fig_path = os.path.join(plot_dir, f"Stacked_Traces_Array{i}.png")
    fig_stack.savefig(stack_fig_path)
    # print(f"Saved stacked traces figure to {stack_fig_path}")

    # Define the cross-correlation window indices (based on the reference trace)
    win_start = int(max(0,    sample_rate * (-start_time + short_win_start)))
    win_end   = int(min(npts, sample_rate * (-start_time + short_win_end)))

    # For plotting, compute the corresponding time axis for the window
    t_axis_window = t_axis[win_start:win_end] + start_time

    # --- Short cross-correlation window plot ---
    fig_short, ax_short = plt.subplots(figsize=(10, 6))
    ax_short.plot(t_axis_window, ref_init_window, color="red", lw=1, label="Central station")
    ax_short.plot(t_axis_window, aligned_stack_window, color="green", lw=1, label="Aligned stack window")
    ax_short.plot(t_axis_window, aligned_restack_window, color="black", lw=1, label="Restack window")

    # Add thin black vertical lines at short_win_start and short_win_end on the short-window plot
    ax_short.axvline(x=short_win_start, color="black", linestyle=":", lw=1)
    ax_short.axvline(x=short_win_end, color="black", linestyle=":", lw=1)
    
    ax_short.set_xlabel("Time (s) relative to P arrival")
    ax_short.set_ylabel("Normalized Amplitude")
    ax_short.set_title(f"Phase {align_phase} Channel {channel} Array {i} Stacked Traces in Short Cross-correlation Window")
    ax_short.legend(loc="upper right", fontsize="small")

    # Save the figure
    save_path_short = os.path.join(plot_dir, f"Short_Window_Stack{i}.png")
    fig_short.savefig(save_path_short)
    # print(f"Saved short window stacked traces figure to {save_path_short}")

# Merge results from results0, results1, results2, results3, and results4 using station as the key
merged_results = {}

for r in chain(results0, results1, results2, results3, results4):
    station = r["station"]
    merged_results.setdefault(station, {}).update(r)

# Compute the sum of lag1, lag2, and lag3 for each merged record
for rec in merged_results.values():
    # Convert lag values to float (or int) and default to 0 if missing
    lag1 = float(rec.get("lag1", 0))
    lag2 = float(rec.get("lag2", 0))
    lag3 = float(rec.get("lag3", 0))
    rec["sum_lags"] = lag1 + lag2 + lag3

# Define the fieldnames for the merged CSV file (including the new "sum_lags" field)
fieldnames = ["station", "sum_lags", "distance_km", "lag_moveout", "lag1", "lag2", "lag3",
              "corr_start", "corr0", "corr1", "corr2", "corr", "corr_long"]

# Write the merged results to a single CSV file.
output_file = os.path.join(results_dir, f"output_all_{evid}.csv")
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in merged_results.values():
        writer.writerow(rec)

print(f"Written merged station results to {output_file}")

plt.show()
