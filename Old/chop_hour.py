from obspy import read
from obspy import UTCDateTime

for i in range(1, 6):
    input_file = f"/Users/vidale/Documents/Research/STAR/Cayman_event/7V.{i}0001.00.DP2.D.2025.039.mseed"
    output_file = f"/Users/vidale/Documents/Research/STAR/Cayman_event/halfhour{i}.mseed"

    # Read the MiniSEED file
    st = read(input_file)

    # Determine the last hour timestamp
    end_time = st[-1].stats.endtime  # Get the end time of the last trace
    start_time_last_30min = end_time - 1800  # 30 minutes before the end

    # Select only the data from the last hour
    st_last_30min = st.slice(starttime=start_time_last_30min, endtime=end_time)

    # Preprocess the selected data:
    # 1. Remove the mean (demean)
    st_last_30min.detrend("demean")
    
    # 2. Apply a taper (Hann taper with 5% of the trace)
    st_last_30min.taper(max_percentage=0.05, type="hann")
    
    # 3. Decimate to 100 sps (input is 500 sps, so decimation factor is 5)
    for tr in st_last_30min:
        if tr.stats.sampling_rate == 500:
            tr.decimate(factor=5, no_filter=True)
        else:
            print(f"Unexpected sampling rate in trace {tr.id}: {tr.stats.sampling_rate}")

    # Write the processed stream to the output file
    st_last_30min.write(output_file, format="MSEED")
    print(f"Processed last hour written to {output_file}")