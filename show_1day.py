#%% Show a day of 10-minute thumbnail spectrograms.  
# Clicking on a thumbnail brings up a full-scale spectrogram
# John Vidale 5/4/2025, from code written by Hao Zhang

import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import os
import datetime

def show_full_image(image_path):
    base = os.path.basename(image_path)
    parts = base.split('_')
    try:
        year = int(parts[0])
        day = int(parts[1])
        date_dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
        date_str = date_dt.strftime("%b %d (%A)")  # e.g. "Nov 12 (Tue)"
    except Exception as e:
        date_str = ""
    
    top = Toplevel()
    # Include the formatted date string in the title.
    if date_str:
        top.title(f"{date_str} - {image_path}")
    else:
        top.title(image_path)
    
    img = Image.open(image_path)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(top, image=photo)
    label.image = photo
    label.pack()

month = 2
day = 10

date = datetime.datetime(2024, month, day) # should also work for start of 2025
julian_day = date.timetuple().tm_yday

if julian_day > 100:
    year = 2024
else:
    year = 2025

julian_day = date.timetuple().tm_yday
print(f"julian date {julian_day}")

start = [year, julian_day]
end = [2025, 50]

# build a list of years and days to process
if start[0] == end[0]:
    dates = [[start[0], i] for i in range(start[1], end[1] + 1)]
else:
    dates = [[start[0], i] for i in range(start[1], 367)] + [[end[0], i] for i in range(1, end[1] + 1)]

thumbnail_size = (200, 160)

def show_day(idx):
    year, day = dates[idx]
    image_dir = f'/Volumes/STAR2/10min/{year}_{day:03d}' # directory with full images
    image_dir_tight = f'/Volumes/STAR2/10min_thumbnail/{year}_{day:03d}' # directory with thumbnail images
    image_files = [f"{year}_{day:03d}_{i:02d}_{j}.png" for i in range(24) for j in range(6)]
    
    root = tk.Tk()
    date_dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
    root.title(f"{year}_{day:03d} - {date_dt.strftime('%b %d (%A)')}")
    root.geometry("3750x1500")
    
    canvas = tk.Canvas(root, borderwidth=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    fig_count = 0
    for filename in image_files: # create a list of thumbnail images
        image_path = os.path.join(image_dir, filename)
        image_path_tight = os.path.join(image_dir_tight, filename)
        try:
            img = Image.open(image_path_tight)
            img_resized = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
    
            # Create a frame to hold the thumbnail and its label
            thumb_frame = tk.Frame(scroll_frame)
    
            btn = tk.Button(thumb_frame, image=photo, command=lambda p=image_path: show_full_image(p), bd=0, highlightthickness=0)
            btn.image = photo
            btn.pack()
    
            # Calculate starting hour and minute based on thumbnail order.
            hour = fig_count // 6
            minute = (fig_count % 6) * 10
            lbl = tk.Label(thumb_frame, text=f"{hour:02d}:{minute:02d}")
            lbl.pack()
    
            thumb_frame.grid(row=fig_count // 18, column=fig_count % 18, padx=0, pady=0)
            fig_count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
    
    # Navigation buttons placed at the top-right of the window
    nav_frame = tk.Frame(root)
    nav_frame.pack(side="top", anchor="ne", pady=10, padx=10)
    
    if idx > 0:
        prev_btn = tk.Button(nav_frame, text="Previous Day", command=lambda: (root.destroy(), show_day(idx-1)))
        prev_btn.pack(side="top", fill="x", padx=5)
    
    if idx < len(dates)-1:
        next_btn = tk.Button(nav_frame, text="Next Day", command=lambda: (root.destroy(), show_day(idx+1)))
        next_btn.pack(side="top", fill="x", padx=5)
    
    root.mainloop()

# Start at the first day in the dates list.
show_day(0)