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
plot_all_traces = True

channel     = "*Z" # "*1" for N, "*2" for E, "*Z" for Z
# array_select = [1]
array_select = [1, 2, 3, 4, 5] # Select the arrays to process

# Define offsets (in seconds) relative to the picked phase arrival time
align_phase = "P" # "P" for P-wave, "S" for S-wave alignment
start_time = -2   # seconds relative to pick, for start of analysis window
end_time   =  20   # seconds relative to pick, for end   of analysis window

min_freq   = 1 # Frequency range for bandpass filter
max_freq   = 10

# Hypocenter, the CSV has columns "id", "time" (ISO formatted), "latitude", "longitude", and "depth"
evid = "ci40825231" #"ci40789071"
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
save_dir = "PlotsWithShift"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List to store figure handles
fig_list = []

for idx, i in enumerate(array_select):
    print(f"Processing Array {i}")
    # --- Compute epicentral distance
    first_station_id = i * 10000 + 1
    first_station_meta = stations_df[stations_df['station'] == first_station_id].iloc[0]
    station_lat = float(first_station_meta['latitude'])
    station_lon = float(first_station_meta['longitude'])
    deg = locations2degrees(event_lat, event_lon, station_lat, station_lon)
    print(f"    Epicentral distance: {deg*111:.2f} km")

    # --- Compute travel times using the first station
    travel_times = model.get_travel_times(source_depth_in_km=event_depth,
                                          distance_in_degree=deg,
                                          phase_list=["p", "P", "s", "S"])
    print(f"    1st station travel times: {travel_times}")
    for tt in travel_times:
        if tt.phase.name.upper() == "P":
            p_traveltime = tt.time
            p_arrival_time = event_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = tt.time
            s_arrival_time = event_time + s_traveltime

    # --- Define the fixed window relative to the chosen arrival
    if align_phase == "P":
        abs_pick = p_arrival_time + start_time
    elif align_phase == "S":
        abs_pick = s_arrival_time + start_time
    else:
        print("Did not find chosen arrival")
        continue

    # --- Read the data file for this array
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
    # Keep only traces with data in the fixed time window:
    overlapping_traces = Stream()
    for tr in st:
        if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + (end_time - start_time):
            overlapping_traces.append(tr)
    if len(overlapping_traces) == 0:
        print(f"No data for time window in file: {input_file}")
        continue
    # Slice to fixed time window:
    st_window = overlapping_traces.slice(starttime=abs_pick, 
                                           endtime=abs_pick + (end_time - start_time))

    # --- Now process the chosen component (channel stored in the variable "channel")
    st_comp = st_window.select(channel=channel)
    num_traces = len(st_comp)
    print(f"    Array {i}: {num_traces} traces in {channel} channel")
    if num_traces == 0:
        continue
    for tr in st_comp:
        tr.detrend(type="demean")
        tr.taper(max_percentage=0.05, type="cosine")
        tr.filter('bandpass', freqmin=min_freq, freqmax=max_freq, corners=4, zerophase=False)
        tr.data = tr.data / np.max(np.abs(tr.data))
    sample_rate = st_comp[0].stats.sampling_rate
    nstack = int(sample_rate * (end_time - start_time))
    # Compute unshifted and shifted stacks for this array:
    stack_unshifted = np.zeros(nstack)
    stack_shifted   = np.zeros(nstack)
    count = 0
    for tr in st_comp:
        unshifted = tr.data[:nstack]
        # Retrieve lag from station statics (data_df) using station ID
        data_info = data_df[data_df['station'] == int(tr.stats.station)]
        if data_info.empty:
            continue
        if not bool(data_info.iloc[0]['select']):
            continue
        lag = int(data_info.iloc[0]['lag3'])
        if lag > 10 or lag < -10:
            lag = 0
        shifted = np.roll(unshifted, -lag)
        stack_unshifted += unshifted
        stack_shifted   += shifted
        count += 1
    if count > 0:
        stack_unshifted /= count
        stack_shifted   /= count
    else:
        continue
    t_stack = np.linspace(start_time, end_time, nstack)

    # --- Plot each array's stacks in its own figure
    fig = plt.figure(figsize=(10, 5))
    plt.plot(t_stack, stack_unshifted, label="Unshifted", color="blue")
    plt.plot(t_stack, stack_shifted, label="Shifted", color="red")
    plt.xlabel("Time (s) relative to pick")
    plt.ylabel("Normalized Amplitude")
    plt.title(f"Array {i}: Stacks")
    plt.legend(loc="upper right", fontsize="small")
    fig_list.append(fig)
plt.show()