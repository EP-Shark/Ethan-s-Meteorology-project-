import os
import sys
sys.path.append('C:/Users/edubp/Ethan-s-Meteorology-project-')

import torch
import xarray as xr
import numpy as np
from tqdm import tqdm

# Config
DATA_YEARS = [2018, 2019, 2020, 2021, 2022]
RAW_DIR = 'C:/Users/edubp/Ethan-s-Meteorology-project-/data/raw'
PT_DIR = 'C:/Users/edubp/Ethan-s-Meteorology-project-/data/processed'

RADAR_VARS = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH']

def preprocess_file(filepath):

    ''' Opens a .nc file and converts it into a tensor (.pt) file
    Retunrs x (13,120,240) and y (0.0 or 1.0) 
    '''

    ds= xr.open_dataset(filepath)

    #Build channels
    channels = []
    for var in RADAR_VARS:
        for sweep in range(2):
            data = ds[var].values[-1,:,:,sweep]
            data = np.nan_to_num(data, nan = 0.0)
            channels.append(data)

    #Range folding mask
    mask = ds['range_folded_mask'].values[-1, :, :, 0].astype(np.float32)
    channels.append(mask)
    
    # Stack into tensor
    x = torch.tensor(np.stack(channels, axis=0).astype(np.float32))
    
    # Get label and category from filename
    filename = os.path.basename(filepath)
    category = filename.split('_')[0]  # TOR, NUL, or WRN
    y = torch.tensor(1.0 if category == 'TOR' else 0.0)
    
    ds.close()
    
    return x, y, category

def preprocess_split(split = 'train'):
    '''Preprocess all files given a split (train or test)
    '''
    print(f'\nPreprocessing {split} split...')

    for year in DATA_YEARS:
        src_dir = os.path.join(RAW_DIR, split, str(year))
        dst_dir = os.path.join(PT_DIR, split, str(year))

        #skip if source does not exist

        if not os.path.exists(src_dir):
            print(f"Skipping {year} - not found")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir) if f.endswith('.nc')]
        print(f'\nYear {year}: {len(files)} files')

        #Track stats
        tor_count = 0
        nul_count = 0 
        wrn_count = 0
        skipped = 0

        for filename in tqdm(files, desc=f"{year}"):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename.replace('.nc', '.pt'))

            # Skip if already processed
            if os.path.exists(dst_path):
                skipped += 1
                continue

            try:
                x, y, category = preprocess_file(src_path)
                
                # Save as .pt with metadata
                torch.save({
                    'x': x,
                    'y': y,
                    'category': category,
                    'filename': filename
                }, dst_path)
                
                # Track counts
                if category == 'TOR':
                    tor_count += 1
                elif category == 'WRN':
                    wrn_count += 1
                else:
                    nul_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                skipped += 1
                continue

        print(f"Year {year} done:")
        print(f"  TOR: {tor_count}")
        print(f"  WRN: {wrn_count}")
        print(f"  NUL: {nul_count}")
        print(f"  Skipped: {skipped}")

if __name__ == "__main__":
    preprocess_split('train')
    preprocess_split('test')

    print("\nPreprocessing completed!")
    print(f'Processed files saved to {PT_DIR}')