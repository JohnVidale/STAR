#%% Read, align, reject bad traces, stack, and plot
# JV 3/29/2025, fixed by Ruoyan 4/1/2025
# Keisuke 4/10/2025, aligns the waveforms at 0.05-0.4 Hz,

from obspy import read, Stream, UTCDateTime
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd  # Added to read the CSV file
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
import csv

def compute_lag(ref, d, win_start, win_end):
    ref_window = ref[win_start:win_end]
    d_window   = d[  win_start:win_end]
    # Compute unnormalized cross-correlation on the normalized segments
    corr = np.correlate(d_window, ref_window, mode="full")
    return np.argmax(corr) - (len(d_window) - 1)

def read_align_plot1(star_volume_path, event_file, channel, array_select, align_phase, evid):

    # Define offsets (in seconds) relative to the picked phase arrivel time
    start_time = -30   # seconds for start of analysis window
    end_time   = 30   # seconds for end   of analysis window
    duration = end_time - start_time
    #short_win_pre  =  0.1 # seconds for short cross-correlation window pre-pick
    #short_win_post =  0.1 # seconds for short cross-correlation window post-pick
    short_win_pre  =  10 # seconds for short cross-correlation window pre-pick
    short_win_post =  20 # seconds for short cross-correlation window post-pick
    search_win_pre  =  4 # seconds to search for peak near arrival time
    search_win_post =  1 # seconds to search for peak near arrival time
    rising_amp_factor = 10 # maximum amplitude divided by this factor will be used as threshold to find rising of signal
    min_freq   = 0.05 # Frequency range for bandpass filter
    max_freq   = 0.4
    plot_all_trace_traveltimes = False
    plot_with_shift = True

    # Assumes the CSV has columns "id", "time" (ISO formatted), "latitude", "longitude", and "depth"
    df = pd.read_csv(event_file)
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

    # Read station list CSV
    stations_file = "/Users/vidale/Documents/Research/STAR/Lists/STAR_stations.csv"
    stations_df = pd.read_csv(stations_file)

    results1 = []  # List to accumulate station correlation results
    results2 = []  # List to accumulate station correlation results
    results3 = []  # List to accumulate station correlation results

    # Define the directory to save figures
    save_dir1 = "/Users/vidale/Documents/Research/STAR/Plots/Plots_" + evid
    save_dir2 = "/Users/vidale/Documents/Research/STAR/Results/Results_" + evid
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    # List to store figure handles
    fig_list = []

    array_string = ""

    #%% Loop over selected arrays
    for i in array_select:
        print(f"Processing Array {i}")

        #% Compute travel time
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
        # Travel times are checked in reverse order so that the first arrival can be chosen.
        for tt in reversed(travel_times):
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
        input_file = star_volume_path + "/" + f"Array{i}/{year_str}_{day_str}/Array{i}_{year_str}_{day_str}_{hr_str}.mseed"
        tr_name = f"Array {i} {year_str} day {day_str} hour {hr_str}"
        if not os.path.exists(input_file):
            print(f"File not found, skipping: {tr_name}")
            continue
        
        #%% Read and preprocess
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
            if np.max(np.abs(tr.data)) > 0:
                tr.data = tr.data / np.max(np.abs(tr.data))
        npts = min(tr.stats.npts for tr in st_comp)

        #%% Extract the first trace to be the initial correlation target
        first_tr    = st_comp[0]
        sample_rate = st_comp[0].stats.sampling_rate
        ref = first_tr.data[:npts]
        t_axis = first_tr.times()[:npts]
        pre_samples  = int(sample_rate * short_win_pre)
        post_samples = int(sample_rate * short_win_post)
        print(f"    sample_rate = {sample_rate}")

        # Find rising of signal near arrival time
        # limit to times that exist in case correlation window is too large
        search_win_start = int(max(0,    sample_rate * (-start_time - search_win_pre )))
        search_win_end   = int(min(npts, sample_rate * (-start_time + search_win_post)))
        search_window = ref[search_win_start:search_win_end]
        peak_amp = np.max(np.abs(search_window))
        rising_index = np.argmax(np.abs(search_window) > peak_amp / rising_amp_factor)
        rising_time_shift = rising_index / sample_rate - search_win_pre
        print(f"!    rising_time_shift = {rising_time_shift}")

        # Compute the plotting time range (common for both arrivals)
        if align_phase == "P":
            t_min = start_time + p_traveltime + rising_time_shift
        if align_phase == "S":
            t_min = start_time + s_traveltime + rising_time_shift
        t_max = t_min + duration

        # limit to times that exist in case correlation window is too large
        win_start = int(max(0,    sample_rate * (-start_time - short_win_pre  + rising_time_shift)))
        win_end   = int(min(npts, sample_rate * (-start_time + short_win_post + rising_time_shift)))

        # --- 1. Compute an aligned stack correlating with the first trace ---
        aligned_stack = np.zeros(npts)
        for tr in st_comp:
            d = tr.data[:npts] # Use only the first npts samples

            # Compute unnormalized cross-correlation on the normalized segments
            lag1 = compute_lag(ref, d, win_start, win_end)

            # apply correlation shifts and stack
            aligned_data = np.roll(d, -lag1)
            aligned_stack += aligned_data

            # save 1st round of correlation shifts
            station_id = str(tr.stats.station)
            if len(station_id) == 3 and station_id.isdigit():
                    orig_station_id = station_id
                    station_id = station_id[0] + "00" + station_id[1:]
                    print(f"Corrected station name: {orig_station_id} -> {station_id}") 
            result_dict1 = {
                "station": station_id,
                "lag1": lag1            }
            results1.append(result_dict1)

        # Normalize the aligned stack
        aligned_stack = aligned_stack / np.max(np.abs(aligned_stack))

        # --- 2. Compute a "good" aligned stack omitting traces with low correlation ---
        selected_aligned_stack = np.zeros(npts)
        for tr in st_comp:
            d = tr.data[:npts]

            ref_window = aligned_stack[win_start:win_end]
            lag2 = compute_lag(aligned_stack, d, win_start, win_end)

            aligned_data = np.roll(d, -lag2)

            # Compute correlation on the aligned window
            aligned_window = aligned_data[win_start:win_end]
            if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
                r_window = 0
            else:
                r_window = np.dot(aligned_window, ref_window) / (np.linalg.norm(aligned_window)* np.linalg.norm(ref_window))
            # and whole trace correlation
            if np.linalg.norm(aligned_data)   == 0 or np.linalg.norm(aligned_stack)        == 0:
                r_whole = 0
            else:
                r_whole =  np.dot(aligned_data,   aligned_stack) / (np.linalg.norm(aligned_data)  * np.linalg.norm(aligned_stack))

            if r_window >= 0.7 and r_whole >= 0.1:
                selected_aligned_stack += aligned_data
            else:
                print(f"    Rejected trace {tr.stats.station} with window r {r_window:.2f} and whole r {r_whole:.2f}")

            # save 2nd round of correlation shifts
            station_id = str(tr.stats.station)
            if len(station_id) == 3 and station_id.isdigit():
                    orig_station_id = station_id
                    station_id = station_id[0] + "00" + station_id[1:]
                    print(f"Corrected station name: {orig_station_id} -> {station_id}")
            result_dict2 = {
                "station": station_id,
                "lag2": lag2            }
            results2.append(result_dict2)

        # Normalize the selected aligned stack
        selected_aligned_stack = selected_aligned_stack / np.max(np.abs(selected_aligned_stack))

        #%% --- Define subplot layout for individual traces ---
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

        # 1. make each subplot j
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
                ref_window = selected_aligned_stack[win_start:win_end]
                lag3 = compute_lag(selected_aligned_stack, x, win_start, win_end)

                if plot_with_shift == True:
                    aligned_data = np.roll(data, -lag3)
                else:
                    aligned_data = np.roll(data, 0)

                aligned_window = aligned_data[win_start:win_end]
                if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
                    r_window = 0
                else:
                    r_window       = np.dot(aligned_window, ref_window) / (np.linalg.norm(aligned_window) * np.linalg.norm(ref_window))
                if np.linalg.norm(aligned_data)   == 0 or np.linalg.norm(selected_aligned_stack)        == 0:
                    r_whole = 0
                else:
                    r_whole = np.dot(aligned_data, selected_aligned_stack) / (np.linalg.norm(aligned_data) * np.linalg.norm(selected_aligned_stack))
                times = tr.times()[:len(aligned_data)]
                offset = - (idx + 1 + extra_offset) * offset_unit
                
                select = False
                if r_window >= 0.6:
                    select = True

                # Define station_id from the current trace as a string
                station_id = str(tr.stats.station)
                if len(station_id) == 3 and station_id.isdigit():
                    orig_station_id = station_id
                    station_id = station_id[0] + "00" + station_id[1:]
                    print(f"Corrected station name: {orig_station_id} -> {station_id}")
                # Now look up station metadata using station_id, not currently used
                # calculate travel time for each station
                station_info = stations_df[stations_df['station'] == int(station_id)]
                st_lat = float(station_info.iloc[0]['latitude'])
                st_lon = float(station_info.iloc[0]['longitude'])
                elev = float(station_info.iloc[0]['elevation'])  # elevation in meters
                station_label = station_id[-2:]
                # Compute epicentral distance (in degrees) for this trace
                deg_trace = locations2degrees(event_lat, event_lon, st_lat, st_lon)
                arrival_delay = (p_arrival_time - event_time) if 'p_arrival_time' in locals() and p_arrival_time is not None else ""
                
                # Append results for the current trace:
                # Compute arrival delay in seconds (difference between hypocentral time and arrival time)
                result_dict3 = {
                    "station": station_id,
                    "distance_km": f"{deg_trace * 111.0:.4f}",
                    "arrival_delay": f"{arrival_delay:.4f}" if arrival_delay != "" else "",
                    "rise_shift": f"{rising_time_shift:.4f}",
                    "lag3": lag3,
                    "corr": f"{r_window:.4f}",
                    "select": select            }
                results3.append(result_dict3)

                # --- Plotting each individual trace: subtract the chosen alignment time ---
                if select == True:
                    ax.plot(t_axis + start_time + p_traveltime, aligned_data + offset, label=station_label)
                else:
                    ax.plot(t_axis + start_time + p_traveltime, aligned_data + offset, label=station_label, linestyle=':')
                ax.text((t_axis + start_time + p_traveltime)[0], offset, f"{r_window:.2f}",
                        fontsize=8, color="blue", verticalalignment="top")
                ax.text((t_axis + start_time + p_traveltime)[-1], offset, f"{r_whole:.2f}",
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
        axes[-1].set_xlabel("Time (s) relative to earthquake origin")
        # Include event date and time in the overall plot title
        plt.suptitle(f"Array {i} component {channel[1]} waveforms + aligned arrivals\nEvent: {event_time.isoformat()}", fontsize=14)
        plt.tight_layout(pad=2.0)   
        plt.subplots_adjust(top=0.90)
        
        common_ylim = axes[0].get_ylim()
        for j in range(total_subplots):
            axes[j].set_ylim(common_ylim)
        
        save_path = os.path.join(save_dir1, f"Array{i}_component_seismograms.png")
        fig.savefig(save_path)
        print(f"    Saved figure for Array {i} to {save_path}")
        fig_list.append(fig)

        array_string = array_string + str(i)

    #%% save lag to csv
    # output_file1 = "Results/output1_" + evid + "_" + array_string + ".csv"
    # with open(output_file1, "w", newline="") as f:
    #     fieldnames = ["station", "lag1"]
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for rec in results1:
    #         writer.writerow(rec)
    # output_file2 = "Results/output2_" + evid + "_" + array_string + ".csv"
    # with open(output_file2, "w", newline="") as f:
    #     fieldnames = ["station", "lag2"]
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for rec in results2:
    #         writer.writerow(rec)
    output_file3 = "/Users/vidale/Documents/Research/STAR/Results/output3_" + evid + "_" + array_string + ".csv"
    with open(output_file3, "w", newline="") as f:
        fieldnames = ["station", "distance_km", "arrival_delay", "rise_shift", "lag3", "corr", "select"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in results3:
            writer.writerow(rec)

if __name__=="__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    star_volume_path = "/Volumes/STAR2"

    # Read the event list CSV
    event_file = "/Users/vidale/Documents/Research/STAR/Lists/List_all.csv"

    channel     = "*Z" # Change to "*1" for N, "*2" for E, "*Z" for Z
    array_select = [1, 2, 3, 4, 5] # Select the arrays to process

    align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
    evid = "ci40835111" #"ci40789071"

    read_align_plot1(star_volume_path, event_file, channel, array_select, align_phase, evid)

    # plt.show()
