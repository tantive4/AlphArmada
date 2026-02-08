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
        self.current_size = 0
        
        # Load metadata if exists to restore cursor position (persistency)
        self.meta_path = os.path.join(data_dir, "metadata.pkl")
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.cursor = meta.get('cursor', 0)
                    self.current_size = meta.get('current_size', 0)
            except Exception:
                print("Warning: Could not load metadata.pkl, starting buffer from 0.")

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
            if not os.path.exists(path):
                # Create file with 'w+' (allocates space)
                mm = np.memmap(path, dtype=spec['dtype'], mode='w+', shape=spec['shape'])
                del mm
            # Open in read/write mode
            self.files[name] = np.memmap(path, dtype=spec['dtype'], mode='r+', shape=spec['shape'])

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

        if end <= self.max_size:
            # Case 1: Contiguous write (No wrapping)
            for key, arr in batch_data.items():
                if key in self.files:
                    self.files[key][start:end] = arr
        else:
            # Case 2: Wrapped write
            first_part_len = self.max_size - start
            second_part_len = batch_size - first_part_len
            
            for key, arr in batch_data.items():
                if key in self.files:
                    # Write to end of buffer
                    self.files[key][start:self.max_size] = arr[:first_part_len]
                    # Write remainder to beginning
                    self.files[key][0:second_part_len] = arr[first_part_len:]
        
        # Update cursor and size
        self.cursor = (self.cursor + batch_size) % self.max_size
        self.current_size = min(self.current_size + batch_size, self.max_size)

        # Save metadata logic can be deferred or done periodically
        # doing it here for safety
        with open(self.meta_path, 'wb') as f:
            pickle.dump({'cursor': self.cursor, 'current_size': self.current_size}, f)

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
            worker_subdir = os.path.join(data_root, f"worker_{i:02d}")
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
        # NOTE: .clone() is important because memmap is read-only and torch might want a writeable copy,
        # plus it moves data from disk-mapped-page to RAM.
        sample = {
            'phases': int(memmaps['phases'][local_idx]),

            'scalar': torch.from_numpy(memmaps['scalar'][local_idx]).clone(),
            'ship_entities': torch.from_numpy(memmaps['ship_entities'][local_idx]).clone(),
            'ship_coords': torch.from_numpy(memmaps['ship_coords'][local_idx]).clone(),
            'ship_def_tokens': torch.from_numpy(memmaps['ship_def_tokens'][local_idx]).clone(),
            # Convert spatial to Long or Byte. Long is often safer for downstream gathering/masking.
            'spatial': torch.from_numpy(memmaps['spatial'][local_idx]).clone().long(),
            'relations': torch.from_numpy(memmaps['relations'][local_idx]).clone(),
            'active_ship_id': int(memmaps['active_ship_id'][local_idx]),
            'target_ship_id': int(memmaps['target_ship_id'][local_idx]),

            'target_policies': torch.from_numpy(memmaps['target_policies'][local_idx]).clone(),
            'target_values': torch.from_numpy(memmaps['target_values'][local_idx]).clone(),
            'target_ship_hulls': torch.from_numpy(memmaps['target_ship_hulls'][local_idx]).clone(),
            'target_game_length': torch.from_numpy(memmaps['target_game_length'][local_idx]).clone(),
        }
        return sample