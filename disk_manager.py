import os
import pickle
import bisect

import torch
from torch.utils.data import Dataset
import numpy as np

from configs import Config

class DiskReplayBuffer:
    def __init__(self, data_dir, max_size, action_space_size):
        self.data_dir = data_dir
        self.max_size = max_size
        self.cursor = 0

        # Define data shapes for memmap
        # Note: 'spatial' uses uint8 for bit-packed data
        self.specs = {
            'phases':            {'shape': (max_size,), 'dtype': np.int32},

            'scalar':            {'shape': (max_size, Config.SCALAR_FEATURE_SIZE), 'dtype': np.float32},
            'ship_entities':     {'shape': (max_size, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), 'dtype': np.float32},
            'ship_coords':       {'shape': (max_size, Config.MAX_SHIPS, 3), 'dtype': np.float32},
            'ship_def_tokens':   {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), 'dtype': np.float32},
            'spatial':           {'shape': (max_size, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], (Config.BOARD_RESOLUTION[1]+7)//8), 'dtype': np.uint8},
            'relations':         {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_SHIPS, 20), 'dtype': np.float32},
            'active_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},
            'target_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},

            'target_policies':   {'shape': (max_size, action_space_size), 'dtype': np.float32},
            'target_values':     {'shape': (max_size, 1), 'dtype': np.float32},
            'target_win_probs':  {'shape': (max_size, 2), 'dtype': np.float32},
            'target_ship_hulls': {'shape': (max_size, Config.MAX_SHIPS), 'dtype': np.float32},
            'target_game_length':{'shape': (max_size, 6), 'dtype': np.float32},
        }

        self.files = {}
        os.makedirs(self.data_dir, exist_ok=True)
        for name, spec in self.specs.items():
            path = os.path.join(self.data_dir, f'{name}.npy')
            # Open in read/write mode
            self.files[name] = np.memmap(path, dtype=spec['dtype'], mode='w+', shape=spec['shape'])

    def add_batch(self, batch_data):
        """
        Efficiently writes a batch of data to disk, handling circular buffer wrapping.
        """
        # Get batch size from one of the arrays
        batch_size = batch_data['scalar'].shape[0]
        if batch_size == 0: return

        # Determine indices
        start = self.cursor
        end = start + batch_size

        if end > self.max_size:
            raise ValueError(f"Batch write exceeds buffer limit! Cursor: {start}, Batch: {batch_size}, Max: {self.max_size}. Increase WORKER_BUFFER_SIZE.")
        
        # Write data
        for key, arr in batch_data.items():
            if key in self.files:
                self.files[key][start:end] = arr
        
        # Update cursor 
        self.cursor = end

    def trim_buffer(self):
        """
        Finalizes the batch by truncating all files to the exact size of data written.
        Must be called before uploading.
        """
        final_size = self.cursor
        if final_size == 0:
            print("[Buffer] Warning: No data to trim.")
            return

        print(f"[Buffer] Trimming files from {self.max_size} to {final_size} samples...")

        # 1. Flush and close all memmaps to release file handles
        for name, mm in self.files.items():
            mm.flush()
            del mm 
        self.files.clear()

        # 2. Truncate files on disk
        for name, spec in self.specs.items():
            path = os.path.join(self.data_dir, f'{name}.npy')
            if os.path.exists(path):
                # Calculate new size in bytes
                # shape[1:] represents the size of a single sample
                element_shape = spec['shape'][1:] 
                num_elements = np.prod(element_shape) if element_shape else 1
                dtype_size = np.dtype(spec['dtype']).itemsize
                
                target_bytes = int(final_size * num_elements * dtype_size)
                
                with open(path, 'r+b') as f:
                    f.truncate(target_bytes)
                    
    def flush(self):
        for mm in self.files.values():
            mm.flush()

class ArmadaDiskDataset(Dataset):
    def __init__(self, data_root, num_workers, max_size_per_worker, action_space_size):
        self.data_root = data_root
        self.max_size = max_size_per_worker
        
        # Duplicate specs to read
        self.specs = {
            'phases':            {'shape': (max_size_per_worker,), 'dtype': np.int32},

            'scalar':            {'shape': (max_size_per_worker, Config.SCALAR_FEATURE_SIZE), 'dtype': np.float32},
            'ship_entities':     {'shape': (max_size_per_worker, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), 'dtype': np.float32},
            'ship_coords':       {'shape': (max_size_per_worker, Config.MAX_SHIPS, 3), 'dtype': np.float32},
            'ship_def_tokens':   {'shape': (max_size_per_worker, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), 'dtype': np.float32},
            'spatial':           {'shape': (max_size_per_worker, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], (Config.BOARD_RESOLUTION[1]+7)//8), 'dtype': np.uint8},
            'relations':         {'shape': (max_size_per_worker, Config.MAX_SHIPS, Config.MAX_SHIPS, 20), 'dtype': np.float32},
            'active_ship_id':    {'shape': (max_size_per_worker,), 'dtype': np.uint8},
            'target_ship_id':    {'shape': (max_size_per_worker,), 'dtype': np.uint8},

            'target_policies':   {'shape': (max_size_per_worker, action_space_size), 'dtype': np.float32},
            'target_values':     {'shape': (max_size_per_worker, 1), 'dtype': np.float32},
            'target_win_probs':  {'shape': (max_size_per_worker, 2), 'dtype': np.float32},
            'target_ship_hulls': {'shape': (max_size_per_worker, Config.MAX_SHIPS), 'dtype': np.float32},
            'target_game_length':{'shape': (max_size_per_worker, 6), 'dtype': np.float32},
        }
        
        # Multi-worker aggregation logic
        self.worker_data = []  # Stores dicts: {'memmaps': {...}, 'size': int}
        self.cumulative_sizes = [0]
        self.total_size = 0

        for i in range(1,num_workers+1):
            worker_subdir = os.path.join(data_root, f"worker_{i:02d}",data_root)
            meta_path = os.path.join(worker_subdir, "metadata.pkl")
            
            # 1. Determine size for this worker
            current_size = 0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                        current_size = meta.get('current_size', 0)
                except Exception:
                    pass # Treat corrupted/locked file as empty

            # 2. If worker has data, open memmaps
            if current_size > 0:
                memmaps = {}
                try:
                    for name, spec in self.specs.items():
                        path = os.path.join(worker_subdir, f'{name}.npy')
                        # Open read-only
                        memmaps[name] = np.memmap(path, dtype=spec['dtype'], mode='r', shape=spec['shape'])
                    
                    self.worker_data.append({'memmaps': memmaps, 'size': current_size})
                    self.total_size += current_size
                    self.cumulative_sizes.append(self.total_size)
                except Exception as e:
                    print(f"Warning: Failed to load data for worker {i}: {e}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 1. Find which worker owns this index
        # cumulative_sizes example: [0, 100, 250]
        # if idx = 150, bisect_right returns 2. worker_idx = 2-1 = 1.
        worker_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        
        # 2. Calculate local index
        local_idx = idx - self.cumulative_sizes[worker_idx]
        
        worker = self.worker_data[worker_idx]
        memmaps = worker['memmaps']

        # 3. Fetch data
        sample = {
            'phases': int(memmaps['phases'][local_idx]),

            'scalar': torch.tensor(memmaps['scalar'][local_idx]),
            'ship_entities': torch.tensor(memmaps['ship_entities'][local_idx]),
            'ship_coords': torch.tensor(memmaps['ship_coords'][local_idx]),
            'ship_def_tokens': torch.tensor(memmaps['ship_def_tokens'][local_idx]),
            
            'spatial': torch.tensor(memmaps['spatial'][local_idx]).long(), # Convert spatial to Long
            'relations': torch.tensor(memmaps['relations'][local_idx]),
            'active_ship_id': int(memmaps['active_ship_id'][local_idx]),
            'target_ship_id': int(memmaps['target_ship_id'][local_idx]),

            'target_policies': torch.tensor(memmaps['target_policies'][local_idx]),
            'target_values': torch.tensor(memmaps['target_values'][local_idx]),
            'target_ship_hulls': torch.tensor(memmaps['target_ship_hulls'][local_idx]),
            'target_game_length': torch.tensor(memmaps['target_game_length'][local_idx]),
            'target_win_probs': torch.tensor(memmaps['target_win_probs'][local_idx]),

        }
        return sample