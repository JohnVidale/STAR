from obspy import read
import matplotlib.pyplot as plt
import numpy as np

# Use only the combined1 file
filename = "/Users/vidale/Documents/Research/STAR/Big_event/combined1.mseed"
st_file = read(filename)

# Process only the Z component
component = 'Z'
start_time = 0
dur = 3600  # seconds

plt.figure(figsize=(10, 6))

# Select Z component traces
st_comp = st_file.select(channel="*" + component)
if len(st_comp) == 0:
    print("No traces found for component", component)
else:
    # Process and plot each trace individually with an offset of 1 unit
    for i, tr in enumerate(st_comp):
        # Apply bandpass filter between 2 Hz and 20 Hz
        tr.filter('bandpass', freqmin=2, freqmax=20)
        
        # Trim the trace to the specified duration
        tr.trim(starttime=tr.stats.starttime + start_time,
                endtime=tr.stats.starttime + start_time + dur)
        
        # Normalize the trace by its RMS amplitude
        data = tr.data
        rms = np.sqrt(np.mean(data**2))
        if rms != 0:
            data = data / rms
        
        # Get the time axis for the trace
        times = tr.times()[:len(data)]
        
        # Compute vertical offset (each trace offset by 1 unit)
        offset = i * 1.0
        
        # Plot the normalized trace with offset
        plt.plot(times, data + offset, label=f"Trace {i+1}")

plt.title("Normalized Z Component Seismograms with 1 Unit Offset (Combined1)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude + Offset")
plt.legend()
plt.tight_layout()
plt.show()