#%% Read and plot with shift
# Keisuke 4/11/2025

from obspy import read, Stream, UTCDateTime
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd  # Added to read the CSV file
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
import csv

def plot_with_shift(star_volume_path, event_file, channel, array_select, align_phase, evid, plot_with_shift):

    # Define offsets (in seconds) relative to the picked phase arrival time
    start_time = -5#-16   # seconds for start of analysis window
    end_time   = 5#20   # seconds for end   of analysis window
    #start_time = -0.1   # seconds for start of analysis window
    #end_time   = 0.3   # seconds for end   of analysis window
    duration = end_time - start_time
    extra_time = 30 # lengthen the window by this amount so that shifting can be applied
    search_win_pre_factor = 8 # search for peak within (P_traveltime / this_value) before arrival time
    search_win_post_factor = 12 # search for peak within (P_traveltime / this_value) after arrival time
    rising_amp_factor = 2 # maximum amplitude divided by this factor will be used as threshold to find rising of signal
    min_freq   = 4 # Frequency range for bandpass filter
    max_freq   = 40
    plot_all_trace_traveltimes = False

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
    stations_file = "Lists/STAR_stations.csv"
    stations_df = pd.read_csv(stations_file)

    # File to read lag data from
    if plot_with_shift == True:
        input_array_string = ""
        for i in array_select:
            input_array_string = input_array_string + str(i)
        data_file = "Results2/output3_" + evid + "_" + input_array_string + ".csv"
        data_df = pd.read_csv(data_file)

    # Define the directory to save figures
    if plot_with_shift == True:
        save_dir = "PlotsWithShift_" + evid
    else:
        save_dir = "PlotsOriginal_" + evid
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
                                            phase_list=["p", "P", "s", "S", "PmP", "SmS"])
        print(f"    1st station = {travel_times} s")

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
        st_window = overlapping_traces.slice(starttime = abs_pick - extra_time, endtime = abs_pick + duration + extra_time)

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

        # Find sample rate from first trace
        first_tr    = st_comp[0]
        sample_rate = st_comp[0].stats.sampling_rate

        # Find rising of signal near arrival time
        # limit to times that exist in case correlation window is too large
        ref = first_tr.data[:npts]
        search_win_pre = max(p_traveltime / search_win_pre_factor, 1.0)
        search_win_post = max(p_traveltime / search_win_post_factor, 1.0)
        search_win_start = int(max(0,    sample_rate * (-start_time + extra_time - search_win_pre )))
        search_win_end   = int(min(npts, sample_rate * (-start_time + extra_time + search_win_post)))
        search_window = ref[search_win_start:search_win_end]
        peak_amp = np.max(np.abs(search_window))
        rising_index = np.argmax(np.abs(search_window) > peak_amp / rising_amp_factor)
        rising_time_shift = rising_index / sample_rate - search_win_pre
        print(f"!    rising_time_shift = {rising_time_shift}")

        win_start = int(sample_rate * (extra_time + rising_time_shift))
        win_end   = int(sample_rate * (extra_time + rising_time_shift + duration))

        t_min = start_time + p_traveltime + rising_time_shift
        t_max = start_time + p_traveltime + rising_time_shift + duration

        selected_aligned_stack = np.zeros(win_end - win_start)
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

                # Define station_id from the current trace as a string
                station_id = str(tr.stats.station)

                ## Get lag data
                if plot_with_shift == True:
                    data_info = data_df[data_df['station'] == int(station_id)]
                    lag = int(data_info.iloc[0]['lag3'])
                    select = bool(data_info.iloc[0]['select'])
                else:
                    lag = 0
                    select = True

                aligned_data = np.roll(data, -lag)
                cut_data = aligned_data[win_start:win_end]

                times = tr.times()[:len(cut_data)]
                offset = - (idx + 1 + extra_offset) * offset_unit

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

                # --- Plotting each individual trace: subtract the chosen alignment time ---
                if select == True:
                    ax.plot(times + start_time + p_traveltime + rising_time_shift, cut_data + offset, label=station_label)
                    selected_aligned_stack += cut_data
                else:
                    ax.plot(times + start_time + p_traveltime + rising_time_shift, cut_data + offset, label=station_label, linestyle=':')
                ax.axvline(x = p_traveltime + rising_time_shift, color="black", linestyle=":", lw=1)

            # --- Mark the arrival times as vertical lines in each subplot ---
            for tt in travel_times:
                traveltime = tt.time
                # # Plot only if the arrival lies within this range
                if t_min <= traveltime <= t_max:
                    if tt.phase.name.upper() == "P":
                        ax.axvline(x=traveltime, color="magenta", linestyle="--", lw=1)
                    if tt.phase.name.upper() == "PMP":
                        ax.axvline(x=traveltime, color="blue", linestyle="--", lw=1)
                    if tt.phase.name.upper() == "S":
                        ax.axvline(x=traveltime, color="cyan", linestyle="--", lw=1)
                    if tt.phase.name.upper() == "SMS":
                        ax.axvline(x=traveltime, color="red", linestyle="--", lw=1)

            ax.set_title(f"Traces {chunk_start+1} to {min(chunk_start+chunk_size, num_traces)}")
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.legend(loc="upper right", fontsize="small")

        # Normalize the selected aligned stack
        selected_aligned_stack = selected_aligned_stack / np.max(np.abs(selected_aligned_stack))

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
        
        save_path = os.path.join(save_dir, f"Array{i}_component_seismograms.png")
        fig.savefig(save_path)
        print(f"    Saved figure for Array {i} to {save_path}")
        fig_list.append(fig)

        #%% 2. add a separate frame for the three stacked traces:
        fig_stack, ax_stack = plt.subplots(figsize=(10, 6))

        # Plot the "good" aligned stack (selected traces)
        ax_stack.plot(times + start_time + p_traveltime + rising_time_shift, selected_aligned_stack, color="black", lw=1, label="Good stack")
        ax_stack.axvline(x = p_traveltime + rising_time_shift, color="black", linestyle=":", lw=1)
        # "rising_time_shift" is not added so that the rise of signal aligns with the P arrival
        # ax_stack.plot(times + start_time + p_traveltime, selected_aligned_stack, color="black", lw=1, label="Good stack")

        # --- Mark the arrival times as vertical lines in each subplot ---
        for tt in travel_times:
            traveltime = tt.time
            # # Plot only if the arrival lies within this range
            if t_min <= traveltime <= t_max:
                if tt.phase.name.upper() == "P":
                    ax_stack.axvline(x=traveltime, color="magenta", linestyle="--", lw=1)
                if tt.phase.name.upper() == "PMP":
                    ax_stack.axvline(x=traveltime, color="blue", linestyle="--", lw=1)
                if tt.phase.name.upper() == "S":
                    ax_stack.axvline(x=traveltime, color="cyan", linestyle="--", lw=1)
                if tt.phase.name.upper() == "SMS":
                    ax_stack.axvline(x=traveltime, color="red", linestyle="--", lw=1)

        ax_stack.set_xlabel("Time (s) relative to earthquake origin")
        ax_stack.set_ylabel("Normalized Amplitude")
        ax_stack.set_title("Stacked Traces")
        ax_stack.legend(loc="upper right", fontsize="small")

        # Optionally save the figure:
        stack_fig_path = os.path.join(save_dir, f"Stacked_Traces_Array{i}.png")
        fig_stack.savefig(stack_fig_path)
        print(f"Saved stacked traces figure to {stack_fig_path}")

        output_file = os.path.join(save_dir, f"Stack_Array{i}.csv")
        with open(output_file, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(['time_event', 'time_arrival', 'time_rising', 'value'])
            for ai, bi, ci, di in zip(times + start_time + p_traveltime + rising_time_shift,
                                times + start_time + rising_time_shift,
                                times + start_time,
                                selected_aligned_stack):
                wr.writerow([ai, bi, ci, di])

if __name__=="__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    star_volume_path = "/Volumes/STAR"
    #star_volume_path = "/Volumes/Hao/AoA"

    # Read the event list CSV
    event_file = "Lists/List_wide.csv"
    # event_file = "Lists/myList.csv"

    channel     = "*Z" # Change to "*1" for N, "*2" for E, "*Z" for Z
    array_select = [1, 2, 3, 4, 5] # Select the arrays to process
    #array_select = [1, 2, 3, 4] # Select the arrays to process
    #array_select = [5] # Select the arrays to process

    align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
    evid = "ci40789071"
    # evid = "my202431610"

    plot_with_shift(star_volume_path, event_file, channel, array_select, align_phase, evid, False)

    # plt.show()
