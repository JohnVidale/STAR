import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import os
import datetime


def show_full_image(image_path):
    top = Toplevel()
    top.title(image_path)
    img = Image.open(image_path)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(top, image=photo)
    label.image = photo
    label.pack()

# start = [2024, 309]
start = [2024, 316]
end = [2025, 50]

# build a list of years and days to process
if start[0] == end[0]:
    dates = [[start[0], i] for i in range(start[1], end[1] + 1)]
else:
    dates = [[start[0], i] for i in range(start[1], 367)] + [[end[0], i] for i in range(1, end[1] + 1)]


thumbnail_size = (200, 160)

for year, day in dates: # progress through days
        
    image_dir = f'/Volumes/STAR2/10min/{year}_{day:03d}' # directory with full images
    image_dir_tight = f'/Volumes/STAR2/10min_thumbnail/{year}_{day:03d}' # directory with thumbnail images

    image_files = [f"{year}_{day:03d}_{i:02d}_{j}.png" for i in range(24) for j in range(6)] # fill 6 10-minute by 24 hours grid

    root = tk.Tk()
    date_dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1)
    root.title(f"{year}_{day:03d} - {date_dt.strftime('%b %d (%A)')}")
    root.geometry("1250x1500")

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

            thumb_frame.grid(row=fig_count // 6, column=fig_count % 6, padx=0, pady=0)
            fig_count += 1
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    root.mainloop()