from obspy import read
import matplotlib.pyplot as plt
from obspy import Stream
import numpy as np
import matplotlib.ticker as ticker

# Use only the combined1 file
filename = "/Users/vidale/Documents/Research/STAR/Big_event/combined1.mseed"
st_file = read(filename)

# Process all components: Z, 1, and 2
components = ['Z', '1', '2']
start_time = 0
dur = 3600  # seconds

plt.figure(figsize=(10, 6))

for comp in components:
    st_comp = st_file.select(channel="*" + comp)
    if len(st_comp) == 0:
        continue

    # Omit trace 15: remove the trace at index 14 if it exists
    if len(st_comp) > 14:
        st_comp.pop(14)

    # Apply bandpass filter between 2 Hz and 20 Hz
    st_comp.filter('bandpass', freqmin=2, freqmax=20)

    # Trim each trace to the specified duration
    for tr in st_comp:
        tr.trim(starttime=tr.stats.starttime + start_time,
                endtime=tr.stats.starttime + start_time + dur)

    # Normalize each trace by its RMS amplitude before stacking
    npts = min(tr.stats.npts for tr in st_comp)
    normalized_traces = []
    for tr in st_comp:
        data = tr.data[:npts]
        rms = np.sqrt(np.mean(data ** 2))
        if rms != 0:
            data = data / rms
        normalized_traces.append(data)

    # Stack normalized traces by summing them
    data_stack = np.sum(normalized_traces, axis=0)

    # Use the time axis from one of the trimmed traces
    times = st_comp[0].times()[:len(data_stack)]

    plt.plot(times, data_stack, label=f"Component {comp}")

plt.title("Stacked Seismograms for Components Z, 1, and 2 (Combined1)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.tight_layout()
plt.show()