from obspy import read
import matplotlib.pyplot as plt
from obspy import Stream
import numpy as np
import matplotlib.ticker as ticker
from obspy.signal.filter import envelope

# Use only the combined1 file
filename = "/Users/vidale/Documents/Research/STAR/Big_event/combined1.mseed"
st_file = read(filename)
dur = 3600

# Process all components: Z, 1, and 2
components = ['Z', '1', '2']

# Create a single figure for the combined1 file
fig = plt.figure(figsize=(10, 6))

for comp in components:
    st_comp = st_file.select(channel="*" + comp)
    if len(st_comp) == 0:
        continue
    
    # Omit trace 15: remove trace at index 14 if it exists
    if len(st_comp) > 14:
        st_comp.pop(14)

    # Apply a bandpass filter between 1 Hz and 10 Hz
    st_comp.filter('bandpass', freqmin=1, freqmax=10)
    
    # Trim each trace to the specified duration
    for tr in st_comp:
        tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.starttime + dur)
    
    # Normalize each trace by its RMS amplitude before stacking
    npts = min(tr.stats.npts for tr in st_comp)
    normalized_traces = []
    for tr in st_comp:
        data = tr.data[:npts]
        rms = np.sqrt(np.mean(data**2))
        if rms != 0:
            data = data / rms
        normalized_traces.append(data)
    
    # Stack all normalized traces by summing them
    data_stack = np.sum(normalized_traces, axis=0)
    
    # Compute the envelope of the stacked data
    env_data = envelope(data_stack)
    
    # Use the time axis from one of the trimmed traces
    times = st_comp[0].times()[:len(env_data)]
    
    plt.plot(times, env_data, label=f"Component {comp}")

plt.title("Envelopes of Stacked Seismograms (Combined1, omitting Trace 15)")
plt.xlabel("Time (s)")
plt.ylabel("Envelope Amplitude")
plt.ylim(-0.5, 2.5)
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.legend()
plt.tight_layout()
plt.show()