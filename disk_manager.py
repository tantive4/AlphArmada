import os
import pickle

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
            'ship_coords':       {'shape': (max_size, Config.MAX_SHIPS, 2), 'dtype': np.float32},
            'ship_def_tokens':   {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), 'dtype': np.float32},
            'spatial':           {'shape': (max_size, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], (Config.BOARD_RESOLUTION[1]+7)//8), 'dtype': np.uint8},
            'relations':         {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_SHIPS, 16), 'dtype': np.uint8},
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
    def __init__(self, data_dir, max_size, current_size, action_space_size):
        self.data_dir = data_dir
        self.size = current_size
        
        # Duplicate specs to read
        self.specs = {
            'phases':            {'shape': (max_size,), 'dtype': np.int32},

            'scalar':            {'shape': (max_size, Config.SCALAR_FEATURE_SIZE), 'dtype': np.float32},
            'ship_entities':     {'shape': (max_size, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), 'dtype': np.float32},
            'ship_coords':       {'shape': (max_size, Config.MAX_SHIPS, 2), 'dtype': np.float32},
            'ship_def_tokens':   {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), 'dtype': np.float32},
            'spatial':           {'shape': (max_size, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], (Config.BOARD_RESOLUTION[1]+7)//8), 'dtype': np.uint8},
            'relations':         {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_SHIPS, 16), 'dtype': np.uint8},
            'active_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},
            'target_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},

            'target_policies':   {'shape': (max_size, action_space_size), 'dtype': np.float32},
            'target_values':     {'shape': (max_size, 1), 'dtype': np.float32},
            'target_win_probs':  {'shape': (max_size, 2), 'dtype': np.float32},
            'target_ship_hulls': {'shape': (max_size, Config.MAX_SHIPS), 'dtype': np.float32},
            'target_game_length':{'shape': (max_size, 6), 'dtype': np.float32},
        }
        
        self.memmaps = {}
        for name, spec in self.specs.items():
            path = os.path.join(self.data_dir, f'{name}.npy')
            # Open in read-only mode for safety during training
            self.memmaps[name] = np.memmap(path, dtype=spec['dtype'], mode='r', shape=spec['shape'])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # NOTE: .clone() is important because memmap is read-only and torch might want a writeable copy,
        # plus it moves data from disk-mapped-page to RAM.
        sample = {
            'phases': int(self.memmaps['phases'][idx]),

            'scalar': torch.from_numpy(self.memmaps['scalar'][idx]).clone(),
            'ship_entities': torch.from_numpy(self.memmaps['ship_entities'][idx]).clone(),
            'ship_coords': torch.from_numpy(self.memmaps['ship_coords'][idx]).clone(),
            'ship_def_tokens': torch.from_numpy(self.memmaps['ship_def_tokens'][idx]).clone(),
            # Convert spatial to Long or Byte. Long is often safer for downstream gathering/masking.
            'spatial': torch.from_numpy(self.memmaps['spatial'][idx]).clone().long(),
            'relations': torch.from_numpy(self.memmaps['relations'][idx]).clone(),
            'active_ship_id': int(self.memmaps['active_ship_id'][idx]),
            'target_ship_id': int(self.memmaps['target_ship_id'][idx]),

            'target_policies': torch.from_numpy(self.memmaps['target_policies'][idx]).clone(),
            'target_values': torch.from_numpy(self.memmaps['target_values'][idx]).clone(),
            'target_ship_hulls': torch.from_numpy(self.memmaps['target_ship_hulls'][idx]).clone(),
            'target_game_length': torch.from_numpy(self.memmaps['target_game_length'][idx]).clone(),
        }
        return sample