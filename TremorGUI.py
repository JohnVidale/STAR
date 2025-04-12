import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from obspy.core import read
import numpy as np

class ImageGUI:
    def __init__(self, root, st, save_path):
        self.root = root
        self.root.title("Tremor GUI")
        self.save_path = save_path

        self.n = len(st)
        self.st = st

        self.fig, self.axes = plt.subplots(self.n, 1, figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.lines = [None] * self.n
        self.rects = [None] * self.n
        self.press_event = None
        self.move_event = None
        self.xlim_history = []
        self.picks = []
        self.p_lines = [[]] * self.n

        self.y_min, self.y_max = -2.5, 2.5

        self.draw_plots()

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.root.bind("u", self.on_key_press)
        self.root.bind("p", self.on_key_press)
        self.root.bind("s", self.on_key_press)
        self.root.bind("q", lambda e: self.root.quit())

    def draw_plots(self):
        """
        This function draws the plots. Change this function if you want to change the appearance of the plots (spectrogram, etc.)
        """
        for i, ax in enumerate(self.axes):
            
            ax.plot(self.st[i].times(), self.st[i].data, color='black', lw=0.5)
            if i != self.n-1:
                ax.set_xticks([])
            else:
                x_ticks = np.arange(self.st[i].times()[0], self.st[i].times()[-1], 20)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([str(x_) for x_ in x_ticks], fontsize=8)
                # ax.set_xticklabels([(self.st[0].stats.starttime + x).strftime("%Y-%m-%d %H:%M:%S") for x in x_ticks], fontsize=8, rotation=45)
                ax.set_xlabel("Time [s]", fontsize=8)
            ax.set_yticks([])
            ax.set_ylim(self.y_min, self.y_max)
            ax.set_ylabel(f"{self.st[i].stats.network}.{self.st[i].stats.station}", fontsize=8)
        self.xlim_history.append(ax.get_xlim())
        self.canvas.draw()

    def on_mouse_move(self, event):
        """
        Track the mouse movement and draw a vertical line on the plot
        """
        if event.inaxes is None:
            return
        
        x = event.xdata
        if x is not None:
            for i, ax in enumerate(self.axes):
                if self.lines[i] is not None:
                    self.lines[i].remove()
                self.lines[i] = ax.axvline(x, color='blue', lw=0.5)
        self.canvas.draw()
        self.move_event = event

    def on_mouse_drag(self, event):
        """
        Drag to zoom in
        """
        if event.inaxes is None:
            return
        
        if self.press_event is not None:
            x0, y0 = self.press_event.xdata, self.press_event.ydata
            x1, y1 = event.xdata, event.ydata
            if None not in [x0, y0, x1, y1]:
                for i, ax in enumerate(self.axes):
                    if self.rects[i] is not None:
                        self.rects[i].remove()
                    self.rects[i] = ax.add_patch(plt.Rectangle((x0, self.y_min), x1 - x0, self.y_max - self.y_min, 
                                                    color='pink', alpha=0.3, lw=0))
        self.canvas.draw()
    
    def on_mouse_press(self, event):
        if event.inaxes is None:
            return
        
        self.press_event = event
    
    def on_mouse_release(self, event):
        if self.press_event is None or event.inaxes is None:
            return
        
        x0, _ = self.press_event.xdata, self.press_event.ydata
        x1, _ = event.xdata, event.ydata
        if None in [x0, x1]:
            return
        
        for i, ax in enumerate(self.axes):
            if self.rects[i] is not None:
                self.rects[i].remove()
                self.rects[i] = None
            ax.set_xlim(min(x0, x1), max(x0, x1))
        self.xlim_history.append(ax.get_xlim())
        
        self.canvas.draw()
        self.press_event = None
    
    def on_key_press(self, event):
        """
        Press 'u' to undo the last zoom in;
        Press 'p' to pick the current time;
        Press 's' to save the picks
        """
        if event.keysym == "u":
            if len(self.xlim_history) <= 1:
                return
            else:
                self.xlim_history.pop()
                x_min_temp, x_max_temp = self.xlim_history[-1]
                for i, ax in enumerate(self.axes):
                    ax.set_xlim(x_min_temp, x_max_temp)
                self.canvas.draw()
        elif event.keysym == "p":
            x = self.move_event.xdata if self.move_event is not None else None
            if x is not None:
                self.picks.append(x)
                for i, ax in enumerate(self.axes):
                    self.p_lines[i].append(ax.axvline(x, color='red', lw=0.5))
            self.canvas.draw()
        elif event.keysym == "s":
            if len(self.picks) == 0:
                return
            else:
                with open(self.save_path, "w", encoding="utf-8") as f:
                    for pick in self.picks:
                        f.write(f"{self.st[0].stats.starttime + pick}\n")
                print(f"Saved to {self.save_path}")
                

if __name__ == "__main__":

    st = read("./test_data/7V*.mseed")
    st = st.select(component="Z")


    root = tk.Tk()
    app = ImageGUI(root, st, 'test_output.txt')
    root.mainloop()
