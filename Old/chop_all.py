from obspy import read, Stream, UTCDateTime
import os

# Define the fixed start and end time
before_time = 1800
after_time  = 1800
fixed_start = UTCDateTime("2024-11-07T08:39:06") - before_time
fixed_end   = UTCDateTime("2024-11-07T08:39:06") + after_time
hr = 8
# day = 39
year = 2025
day = 312

for i in range(4, 5):
    combined_stream = Stream()
    for j in range(1, 2):
    # for j in range(1, 82):
        j_str    = f"{j   :02d}"  # Two-digit   format for j
        day_str  = f"{day :03d}"  # Three-digit format for day
        year_str = f"{year:04d}"  # Four-digit  format for year
        input_file1 = f"/Volumes/AoA_Data/AofA_MSEED_DATA_1_2_3/ARRAY_{i}_MSEED/7V.{i}00{j_str}.00.DP2.D.{year_str}.{day_str}.mseed"
        input_file2 = f"/Volumes/AoA_Data_2/AofA_MSEED_DATA_3C_4_5/ARRAY_{i}_MSEED/7V.{i}00{j_str}.00.DP2.D.{year_str}.{day_str}.mseed"
        tr_name = f"7V.{i}00{j_str}.00.DP2.D.{year_str}.{day_str}.mseed"

        if   os.path.exists(input_file1):
            input_file = input_file1
        elif os.path.exists(input_file2):
            input_file = input_file2
        else:
            print(f"File not found, skipping: {tr_name}")
            continue

        st = read(input_file2)
        print(f"File processing: {tr_name}")

        # try:
        # except Exception as e:
        #     print(f"Error reading {input_file}: {e}")
        #     continue

        # Only include traces that overlap the fixed time window.
        overlapping_traces = Stream()
        for tr in st:
            if tr.stats.endtime >= fixed_start and tr.stats.starttime <= fixed_end:
                overlapping_traces.append(tr)
        if len(overlapping_traces) == 0:
            print(f"No overlapping traces for fixed window in file: {input_file}")
            continue

        # Now slice these overlapping traces to the fixed time window
        st_window = overlapping_traces.slice(starttime=fixed_start, endtime=fixed_end)

        # Preprocess
        st_window.detrend("demean")
        # st_window.taper(max_percentage=0.05, type="hann")

        for tr in st_window:
            if tr.stats.sampling_rate == 500:
                tr.decimate(factor=5, no_filter=True)
            else:
                print(f"Unexpected sampling rate in trace {tr.id}: {tr.stats.sampling_rate}")

        combined_stream += st_window
        # print(f"Processed: {input_file}")

    # Write the combined output
    output_file = f"/Users/vidale/Documents/Research/STAR/Big_event/combined{i}.mseed"
    combined_stream.write(output_file, format="MSEED")
    print(f"Combined stream written to {output_file}")

    # Count and report number of unique stations
    unique_stations = set(tr.stats.station for tr in combined_stream)
    print(f"Number of unique stations in combined file: {len(unique_stations)}")
    del combined_stream