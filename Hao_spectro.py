'''
Created by Hao Zhang (hzhang63@usc.edu)
'''

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from obspy.core import *
import numpy as np

from scipy import signal

from obspy.clients.fdsn import Client

class ImageGUI:
    def __init__(self, root, st, spec, save_path):
        self.root = root
        self.root.title("Tremor GUI")
        self.save_path = save_path

        self.n = len(st)
        self.st = st
        self.spec = spec
        self.sampling_rate = 40

        self.fig, self.axes = plt.subplots(self.n, 1, figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.x_min0, self.x_max0 = 0, 3600
        self.y_min, self.y_max = 1, 30

        self.lines = [None] * self.n
        self.rects = [None] * self.n
        self.wf_lines = [None] * self.n
        self.backgrounds = [None] * self.n
        self.xlim_history = [[[self.x_min0, self.x_max0]] for _ in range(self.n)]
        self.picks = []
        self.p_lines = [[]] * self.n

        self.press_event = None
        self.move_event = None
        
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        # self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.root.bind("u", self.on_key_press)
        self.root.bind("p", self.on_key_press)
        self.root.bind("s", self.on_key_press)
        self.root.bind("q", lambda e: self.root.quit())

        self.draw_plots()

    def draw_plots(self):
        """
        This function draws the plots. Change this function if you want to change the appearance of the plots (spectrogram, etc.)
        """
        for i, ax in enumerate(self.axes):

            ax.pcolormesh(self.spec[i][1], self.spec[i][0], np.log10(self.spec[i][2]), shading='gouraud', cmap="jet", vmin=-22, vmax=-15)

            if i != self.n-1:
                ax.set_xticks([])
            else:
                x_ticks = np.arange(self.st[i].times()[0], self.st[i].times()[-1], 300)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(x_//60) for x_ in x_ticks], fontsize=6)
                ax.set_xlabel("Time (minutes)", fontsize=6)
            ax.set_yticks([5, 10, 15, 20])
            ax.set_yticklabels([5, 10, 15, 20], fontsize=6)
            ax.set_xlim(self.xlim_history[0][-1])
            ax.set_ylim(self.y_min, self.y_max)
            # plt.colorbar(c, ax=ax)
            ax.set_ylabel(f"{self.st[i].stats.network}.{self.st[i].stats.station}", fontsize=6)

        for i, ax in enumerate(self.axes):
            self.wf_lines[i] = Line2D(self.st[i].times(), self.st[i].data / np.max(np.abs(self.st[i].data)) * 4 + 25, color='black', lw=0.5)
            ax.add_line(self.wf_lines[i])
        self.canvas.draw()
        for i, ax in enumerate(self.axes):
            self.backgrounds[i] = self.canvas.copy_from_bbox(ax.bbox)
        
    def on_mouse_move(self, event):
        """
        Track the mouse movement and draw a vertical line on the plot
        """
        if event.inaxes is None:
            return

        for i, ax in enumerate(self.axes):
            self.canvas.restore_region(self.backgrounds[i])

        x = event.xdata

        if x is not None:
            for i, ax in enumerate(self.axes):
                if self.lines[i] is not None:
                    self.lines[i].remove()
                    self.lines[i] = None
                self.lines[i], = ax.plot([x, x], ax.get_ylim(), color='blue', lw=0.5)
                ax.draw_artist(self.lines[i])
        if self.press_event is not None:
            x0, y0 = self.press_event.xdata, self.press_event.ydata
            x1, y1 = event.xdata, event.ydata
            if None not in [x0, y0, x1, y1]:
                for i, ax in enumerate(self.axes):
                    if self.rects[i] is None:
                        self.rects[i] = ax.add_patch(plt.Rectangle((x0, self.y_min), x1 - x0, self.y_max - self.y_min, 
                                                    color='pink', alpha=0.5, lw=0))
                    else:
                        self.rects[i].set_bounds(x0, self.y_min, x1 - x0, self.y_max - self.y_min)
                    ax.draw_artist(self.rects[i])
            
        self.move_event = event
        for i, ax in enumerate(self.axes):
            self.canvas.blit(ax.bbox)
    
    def on_mouse_press(self, event):
        """
        Record the mouse press event
        """
        if event.inaxes is None:
            return
        self.press_event = event
    
    def on_mouse_release(self, event):
        """
        Record the mouse release event and zoom in
        """
        if self.press_event is None or event.inaxes is None:
            return

        x0, _ = self.press_event.xdata, self.press_event.ydata
        x1, _ = event.xdata, event.ydata
        if None in [x0, x1] or x0 == x1:
            self.press_event = None
            return
        
        for i, ax in enumerate(self.axes):

            if self.wf_lines[i] is not None:
                self.wf_lines[i].remove()
                self.wf_lines[i] = None

            if self.rects[i] is not None:
                self.rects[i].remove()
                self.rects[i] = None
            
            self.xlim_history[i].append([min(x0, x1), max(x0, x1)])
            x_min_temp, x_max_temp = self.xlim_history[i][-1]
            n_start, n_end = int(x_min_temp * self.st[i].stats.sampling_rate), int(x_max_temp * self.st[i].stats.sampling_rate)
            norm_amp = np.max(np.abs(self.st[i].data[n_start:n_end]))
            self.wf_lines[i] = Line2D(self.st[i].times(), self.st[i].data / norm_amp * 4 + 25, color='black', lw=0.5)
            ax.add_line(self.wf_lines[i])
            ax.set_xlim(x_min_temp, x_max_temp)

        self.canvas.draw()
        for i, ax in enumerate(self.axes):
            self.backgrounds[i] = self.canvas.copy_from_bbox(ax.bbox)
        
        self.press_event = None
    
    def on_key_press(self, event):
        """
        Press 'u' to undo the last zoom in;
        Press 'p' to pick the current time;
        Press 's' to save the picks
        """
        if event.keysym == "u":
            if len(self.xlim_history[0]) <= 1:
                return

            for i, ax in enumerate(self.axes):

                if self.wf_lines[i] is not None:
                    self.wf_lines[i].remove()
                    self.wf_lines[i] = None

                if self.lines[i] is not None:
                    self.lines[i].remove()
                    self.lines[i] = None
                
                self.xlim_history[i].pop()
                x_min_temp, x_max_temp = self.xlim_history[i][-1]
                n_start, n_end = int(x_min_temp * self.st[i].stats.sampling_rate), int(x_max_temp * self.st[i].stats.sampling_rate)
                norm_amp = np.max(np.abs(self.st[i].data[n_start:n_end]))
                self.wf_lines[i] = Line2D(self.st[i].times(), self.st[i].data / norm_amp * 4 + 25, color='black', lw=0.5)
                ax.add_line(self.wf_lines[i])
                ax.set_xlim(x_min_temp, x_max_temp)

            self.canvas.draw()
            for i, ax in enumerate(self.axes):
                self.backgrounds[i] = self.canvas.copy_from_bbox(ax.bbox)

        elif event.keysym == "p":
            x = self.move_event.xdata if self.move_event is not None else None
            if x is not None:
                self.picks.append(x)
                for i, ax in enumerate(self.axes):
                    self.p_lines[i].append(ax.axvline(x, color='red', lw=0.5))
                    self.lines[i].remove()
                    self.lines[i] = None

            self.canvas.draw()
            for i, ax in enumerate(self.axes):
                    self.backgrounds[i] = self.canvas.copy_from_bbox(ax.bbox)

        elif event.keysym == "s":
            if len(self.picks) == 0:
                return
            else:
                with open(self.save_path, "w", encoding="utf-8") as f:
                    for pick in self.picks:
                        f.write(f"{self.st[0].stats.starttime + pick}\n")
                print(f"Saved to {self.save_path}")
                

if __name__ == "__main__":

    # Client = Client("SCEDC")
    net_stas = [["AZ", "PFO"], ["AZ", "FRD"], ["CI", "BOR"], ["CI", "RCR"], ["CI", "TOR"]]

    t0 = UTCDateTime("2025-02-08T23:23:14.341Z")

    # st = Stream()

    # for net, sta in net_stas:
    #     st_ = Client.get_waveforms(net, sta, "*", "BHZ", t0, t0 + 3600)
    #     st.append(st_[0])
    # st = st.filter("bandpass", freqmin=1, freqmax=10)
    # st = st.slice(t0+60, t0 + 3500)
    st = read("data.mseed")
    st = st.select(component="Z")

    # time domain waveform
    st_lf = st.copy()
    st_lf = st_lf.filter("bandpass", freqmin=0.1, freqmax=2)
    st_lf.taper(max_percentage=0.02)

    # data to calculate spectrogram
    st = st.filter("bandpass", freqmin=1, freqmax=16)
    st.taper(max_percentage=0.02)
    data_spec = []
    for tr in st:
        f, t, Sxx = signal.spectrogram(tr.data, fs=tr.stats.sampling_rate, nperseg=128, noverlap=16)
        data_spec.append([f, t, Sxx])
    
    root = tk.Tk()
    app = ImageGUI(root, st_lf, data_spec, 'test_output.txt')
    root.mainloop()