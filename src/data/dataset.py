''' This pythong program is designed to be a data loader for the ML model'''

# Importing packages #
import os
import numpy as np
import torch 
from torch.utils.data import Dataset 
import xarray as xr 


# Radar Variables within the TorNET Dataset that we want to extract
RADAR_vars = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]
#DBZ: Rainrate/reflectivity product
#VEL: Velocity product
#KDP: Dual Polarized radar product (measures phases of vertical and Horizontal pulses)
#RHOHV: Correlation Coefficent (measures the similarity of particles in the radar scan)
#ZDR: Differential Reflectivity
#WIDTH: Spectrum Width

#Defining the Class

class TorNETDatabase(Dataset):
    def __init__ (self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.nc')
        ]
        self.files.sort()
        print(f'found{len(self.files)} samples in {data_dir}')

    def __len__(self):
        return len(self.files)
    
    def __getitem__ (self, idx):
        filepath = self.files[idx]
        ds = xr.open_dataset(filepath)

        # Getting the last frame 
        channels = []
        for var in RADAR_vars:
            for sweep in range(2):
                data = ds[var].values[-1, :, : , sweep]
                #Replace the NaN's with 0s
                data = np.nan_to_num(data, nan = 0.0)
                channels.append(data)

        # Add a range folding mask (average the last frame for all sweeps)
        mask = ds['range_folded_mask'].values[-1,:,:,sweep].astype(np.float32)
        channels.append(mask)

        x = np.stack(channels, axis = 0).astype(np.float32)
        label = float(ds['frame_labels'].values[-1])

        ds.close()

        x = torch.tensor(x)
        y = torch.tensor(label)

        if self.transform:
            x = self.transform(x)

        return x,y

