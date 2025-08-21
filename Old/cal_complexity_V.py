# from Hao Zhang, 7/29/2025

import glob

import pandas as pd
import numpy as np
from scipy import stats
from obspy import UTCDateTime
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter

from util import get_rr_parallel

event = "Wenchuan"  # Change this to the desired event name

# parameteres
params = {
    "Ambon":      {"scale": np.arange(1.5, 5.6, 0.5), "num_c":  1200, "pct": 20, "nmin": 10},
    "Baja":       {"scale": np.arange(5,  12.1, 1  ), "num_c":  7000, "pct": 20, "nmin": 10},
    "Darfield":   {"scale": np.arange(2,   6.1, 0.5), "num_c":  3000, "pct": 20, "nmin": 10},
    "Dingri":     {"scale": np.arange(6,  13.1, 1  ), "num_c": 10000, "pct": 20, "nmin": 10}, # take a long time to calculate
    "Idaho":      {"scale": np.arange(4,   8.1, 0.5), "num_c":  5000, "pct": 20, "nmin": 10},
    "Jiuzhai":    {"scale": np.arange(1.6, 4.1, 0.3), "num_c":  6000, "pct": 20, "nmin": 10},
    "Maduo":      {"scale": np.arange(5,  12.1, 1  ), "num_c":  1200, "pct": 20, "nmin": 10},
    "Napa":       {"scale": np.arange(2,   4.7, 0.4), "num_c":  1300, "pct": 20, "nmin": 10},
    "Nevada":     {"scale": np.arange(2,   4.1, 0.5), "num_c": 11000, "pct": 20, "nmin": 10},
    "Petrinja":   {"scale": np.arange(1,   4.1, 0.5), "num_c":  7200, "pct": 20, "nmin": 10},
    "Ridgecrest": {"scale": np.arange(5,  11.1, 1  ), "num_c":  8100, "pct": 20, "nmin": 10},
    "Sarez":      {"scale": np.arange(5,  14.1, 1  ), "num_c":  2000, "pct": 20, "nmin": 10},
    "Sivrice":    {"scale": np.arange(3,  10.1, 1  ), "num_c":  6500, "pct": 20, "nmin": 10},
    "Turkiye":    {"scale": np.arange(5,  10.1, 1  ), "num_c":  8000, "pct": 20, "nmin": 10},
    "Wenchuan":   {"scale": np.arange(6,  14.1, 1  ), "num_c":  2000, "pct": 20, "nmin": 10},
}

# Original values for the parameters
# params = {
#     "Ambon": {"scale": np.arange(1.5, 5.6, 0.5), "num_c": 1200, "pct": 20, "nmin": 10},
#     "Baja_California": {"scale": np.arange(5, 12.1, 1), "num_c": 7000, "pct": 20, "nmin": 10},
#     "Darfield": {"scale": np.arange(2, 6.1, 0.5), "num_c": 3000, "pct": 10, "nmin": 10},
#     "Dingri": {"scale": np.arange(6, 13.1, 1), "num_c": 10000, "pct": 20, "nmin": 20}, # take a long time to calculate
#     "Idaho": {"scale": np.arange(4, 8.1, 0.5), "num_c": 5000, "pct": 10, "nmin": 10},
#     "Jiuzhai": {"scale": np.arange(1.6, 4.1, 0.3), "num_c": 6000, "pct": 10, "nmin": 10},
#     "Maduo": {"scale": np.arange(5, 12.1, 1), "num_c": 1200, "pct": 10, "nmin": 10},
#     "Meinong": {"scale": np.arange(2, 5.6, 0.5), "num_c": 500, "pct": 10, "nmin": 10},
#     "Napa": {"scale": np.arange(2, 4.7, 0.4), "num_c": 1300, "pct": 10, "nmin": 10},
#     "Nevada": {"scale": np.arange(2, 4.1, 0.5), "num_c": 11000, "pct": 20, "nmin": 20},
#     "Petrinja": {"scale": np.arange(1, 4.1, 0.5), "num_c": 7200, "pct": 10, "nmin": 10},
#     "Ridgecrest": {"scale": np.arange(5, 11.1, 1), "num_c": 8000, "pct": 10, "nmin": 20},
#     "Sarez": {"scale": np.arange(5, 14.1, 1), "num_c": 2000, "pct": 10, "nmin": 10},
#     "Sivrice": {"scale": np.arange(3, 10.1, 1), "num_c": 6500, "pct": 20, "nmin": 20},
#     "Turkiye": {"scale": np.arange(5, 10.1, 1), "num_c": 8000, "pct": 20, "nmin": 20},
#     "Wenchuan": {"scale": np.arange(6, 14.1, 1), "num_c": 2000, "pct": 10, "nmin": 10},
# }

catalog = glob.glob(f"/Users/vidale/Documents/Research/Hao_fault_complexity/Catalogs/{event}*.ctlg*")[0]

df = pd.read_csv(catalog, header=None, names=["time", "lat", "lon", "depth", "mag"])

ev_ot = [UTCDateTime(t) for t in df["time"]]
ev_lat = np.array(df[  "lat"].values)
ev_lon = np.array(df[  "lon"].values)
ev_dep = np.array(df["depth"].values)
ev_mag = np.array(df[  "mag"].values)

# reduce the number of events to accelerate the calculation
if event == "Dingri":
    # ev_ot = ev_ot[ev_mag > 1.0]
    ev_lat = ev_lat[ev_mag > 1.0]
    ev_lon = ev_lon[ev_mag > 1.0]
    ev_dep = ev_dep[ev_mag > 1.0]
    ev_mag = ev_mag[ev_mag > 1.0]

lon_c, lat_c = ev_lon.mean(), ev_lat.mean()

ev_x = 111.19 * (ev_lon - lon_c) * np.cos(np.deg2rad(lat_c))
ev_y = 111.19 * (ev_lat - lat_c)

distances = np.sqrt(ev_x**2 + ev_y**2)
avg_distance = distances.mean()
# print(f"Average distance of ev points from the origin: {avg_distance:.2f} km")

scales = params[event]["scale"]
num_c  = params[event]["num_c"]
pct    = params[event]["pct"]
nmin   = params[event]["nmin"]

rr, rr_std = get_rr_parallel(ev_x, ev_y, ev_dep, scales, num_c=num_c, pct=pct, nmin=nmin, n_jobs=8)

# saved results for Dingri since it takes a long time to calculate
# scales = np.arange(6, 13.1, 1)
# rr = np.array([1.10, 1.24, 1.35, 1.44, 1.52, 1.58, 1.63, 1.67])
# rr_std = np.array([0.23, 0.22, 0.21, 0.21, 0.21, 0.23, 0.25, 0.27])

# do linear regression
log_scales = np.log10(scales)
log_rr = np.log10(rr)

slop, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_rr)

n = len(log_scales)
alpha = 0.05 # 95% confidence level
t_val = stats.t.ppf(1 - alpha / 2, df=n-2)
ci = t_val * std_err
b_ci_lower = slop - ci
b_ci_upper = slop + ci

print(f"{event}  ave_dist = {avg_distance:.4f} b = {slop:.4f} ({b_ci_lower:.4f}, {b_ci_upper:.4f})")

# plot the results
x_interp = np.linspace(scales[0] * 0.9, scales[-1] * 1.1, 100)
y_interp = 10 ** intercept * x_interp ** slop

fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(111)

ax.plot(x_interp, y_interp, color='#DA2925', linewidth=2, label=f'Slope = {slop:.4f} ({b_ci_lower:.4f}, {b_ci_upper:.4f})')
ax.errorbar(scales, rr, yerr=rr_std, fmt='o', capsize=3, elinewidth=1, markersize=4, color="#364E98")

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xticks(scales)
ax.set_xticklabels([f"{s:.1f}" for s in scales])

yticks = np.arange(np.floor((rr[0]-rr_std[0])*10)/10, rr[-1]+rr_std[-1], 0.1)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.1f}" for y in yticks])

ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:g}'))

ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:g}'))

ax.set_xlabel("Scale (km)", fontsize=12)
ax.set_ylabel("Average Thickness (km)", fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.legend(fontsize=10, loc='upper left')

plt.savefig(f"/Users/vidale/Documents/Research/Hao_fault_complexity/Figs/Outputs/{event}.pdf", bbox_inches='tight', dpi=200)
# plt.show()

