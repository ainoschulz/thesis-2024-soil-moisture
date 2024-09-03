"""
Code for visualising despeckled Sentinel-1 data.

Author: AS
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import netCDF4 as nc4
import warnings
warnings.filterwarnings("ignore")
from warnings import filterwarnings
warnings.filterwarnings("ignore")

# location of datasets
path_start = # Set your own path
water_path = # Set your own path

#make water mask
file2read = nc4.Dataset(water_path, 'r')
water_mask = file2read.variables['water_mask'][:] * 1

#get all files of despeckled data
data_files = [file for file in os.listdir(path_start) if file.endswith('.npy')]

#grouping files by date
files_by_date = {}
for file in data_files:
    file_parts = file.split('_')
    date = file_parts[2]
    if date not in files_by_date:
        files_by_date[date] = []
    files_by_date[date].append(file)



# Plot one file at a time
for date, files in files_by_date.items():
    print(f"Started handling {date} now...")

    # Iterate over each file for the current date
    for i, file in enumerate(files, 1):
        plt.figure(figsize=(10, 7))
        plt.suptitle(f"Despeckled Sentinel-1 image of Kilpisjärvi\n ({date})\n")

        # Load file
        file_path = os.path.join(path_start, file)
        data = np.load(file_path)

        # mask data for water
        masked_data = data * water_mask

        # Extract polarization
        file_parts = file.split('_')
        pol = file_parts[-1][:2]

        # Title for plot
        title = f"{file_parts[3]} {pol}"

        # Plot
        zero_color = 'white'
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, 255))
        custom_colors = [zero_color] + viridis_colors.tolist()
        cmap = mcolors.ListedColormap(custom_colors)
        plt.imshow(masked_data, cmap=cmap, extent=[0, masked_data.shape[1], 0, masked_data.shape[0]], vmin=0, vmax=1)
        plt.colorbar(label="Intensity", location="right", pad=0.2, shrink=0.3)
        plt.title(title, loc='center', x=0.65)

        # Set limits of plot
        if pol == 'VV':
            plt.clim([0, 0.2])
        else:
            plt.clim([0, 0.1])

        # Remove axis
        plt.axis("off")

        # Save plot
        f_name = os.path.join(path_start, 'plots', f'despeckled_{date}_{pol}.png')
        plt.savefig(f_name)

        # Close plot
        plt.close()

    print(f"{date} is now handled, moving on the next one...")

