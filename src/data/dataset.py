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
    def __init__ (self, data_dirs, transform = None):

        self.transform = transform 
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.files = []
        for data_dir in data_dirs:

            self.files += [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith('.pt')
            ]
        self.files.sort()
        print(f'found{len(self.files)} samples across {len(data_dir)} years')

    def __len__(self):
        return len(self.files)
    
    def __getitem__ (self, idx):
        data = torch.load(self.files[idx], weights_only = True)

        x = data['x']
        y = data['y']

        if self.transform:
            x = self.transform(x)

        return x,y

