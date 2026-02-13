import os
import pickle
import bisect

import torch
from torch.utils.data import Dataset
import numpy as np

from configs import Config

class DiskReplayBuffer:
    def __init__(self, data_dir, max_size):
        self.data_dir = data_dir
        self.max_size = max_size
        self.cursor = 0
        self.specs = self.get_specs(max_size)

        
            
        self.files = {}
        os.makedirs(self.data_dir, exist_ok=True)
        for name, spec in self.specs.items():
            path = os.path.join(self.data_dir, f'{name}.npy')
            # Open in read/write mode
            self.files[name] = np.memmap(path, dtype=spec['dtype'], mode='w+', shape=spec['shape'])

    @staticmethod
    def get_specs(max_size):
        # Define data shapes for memmap
        # Note: 'spatial' uses uint8 for bit-packed data
        return {
            'phases':            {'shape': (max_size,), 'dtype': np.int32},
            'scalar':            {'shape': (max_size, Config.SCALAR_FEATURE_SIZE), 'dtype': np.float32},
            'ship_entities':     {'shape': (max_size, Config.MAX_SHIPS, Config.SHIP_ENTITY_FEATURE_SIZE), 'dtype': np.float32},
            'ship_coords':       {'shape': (max_size, Config.MAX_SHIPS, 3), 'dtype': np.float32},
            'ship_def_tokens':   {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_DEFENSE_TOKENS, Config.DEF_TOKEN_FEATURE_SIZE), 'dtype': np.float32},
            'spatial':           {'shape': (max_size, Config.MAX_SHIPS, 10, Config.BOARD_RESOLUTION[0], (Config.BOARD_RESOLUTION[1]+7)//8), 'dtype': np.uint8},
            'relations':         {'shape': (max_size, Config.MAX_SHIPS, Config.MAX_SHIPS, 20), 'dtype': np.float32},
            'active_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},
            'target_ship_id':    {'shape': (max_size,), 'dtype': np.uint8},
            'target_policies':   {'shape': (max_size, Config.MAX_ACTION_SPACE), 'dtype': np.float32},
            'target_values':     {'shape': (max_size, 1), 'dtype': np.float32},
            'target_win_probs':  {'shape': (max_size, 2), 'dtype': np.float32},
            'target_ship_hulls': {'shape': (max_size, Config.MAX_SHIPS), 'dtype': np.float32},
            'target_game_length':{'shape': (max_size, 6), 'dtype': np.float32},
        }
    
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

        # 3. Save Metadata (since all batches have different size)
        meta_path = os.path.join(self.data_dir, "metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump({'current_size': final_size}, f)
        print(f"[Buffer] Trimmed and saved metadata for size {final_size}")

    def flush(self):
        for mm in self.files.values():
            mm.flush()


def aggregate_staging_buffers(staging_root, output_dir):
    """
    Merges all buffers in staging_root subdirectories into a single buffer in output_dir.
    """
    # 1. Identify all valid staging subdirectories
    staging_dirs = sorted([
        os.path.join(staging_root, d) for d in os.listdir(staging_root) 
        if os.path.isdir(os.path.join(staging_root, d))
    ])
    
    if not staging_dirs:
        print("[Aggregator] No staging buffers found.")
        return

    # 2. Calculate total size
    total_size = 0
    buffer_sizes = []
    
    print(f"[Aggregator] Found {len(staging_dirs)} buffers. Calculating total size...")
    
    for d in staging_dirs:
        meta_path = os.path.join(d, "metadata.pkl")
        size = 0
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                size = pickle.load(f).get('current_size', 0)
        else:
            # Fallback: infer from scalar.npy size
            scalar_path = os.path.join(d, "scalar.npy")
            if os.path.exists(scalar_path):
                file_size = os.path.getsize(scalar_path)
                itemsize = Config.SCALAR_FEATURE_SIZE * 4 # float32 = 4 bytes
                size = file_size // itemsize
        
        buffer_sizes.append(size)
        total_size += size

    print(f"[Aggregator] Total aggregated size: {total_size} samples.")

    # 3. Create Output Buffer (Memmapped)
    os.makedirs(output_dir, exist_ok=True)
    specs = DiskReplayBuffer.get_specs(total_size)
    output_files = {}
    
    for name, spec in specs.items():
        path = os.path.join(output_dir, f'{name}.npy')
        output_files[name] = np.memmap(path, dtype=spec['dtype'], mode='w+', shape=spec['shape'])

    # 4. Copy Data
    cursor = 0
    for idx, d in enumerate(staging_dirs):
        current_size = buffer_sizes[idx]
        if current_size == 0: continue

        for name, spec in specs.items():
            src_path = os.path.join(d, f'{name}.npy')
            if os.path.exists(src_path):
                # Open read-only
                src_mm = np.memmap(src_path, dtype=spec['dtype'], mode='r', shape=(current_size, *spec['shape'][1:]))
                output_files[name][cursor : cursor + current_size] = src_mm
                del src_mm # Close handle

        cursor += current_size

    # 5. Flush and Metadata
    for mm in output_files.values():
        mm.flush()
        del mm
    
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump({'current_size': total_size}, f)

    print(f"[Aggregator] Aggregation complete at {output_dir}")

class ArmadaDiskDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        
        # 1. Find all valid timestamp subdirectories
        self.worker_dirs = sorted([
            os.path.join(data_root, d) for d in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, d))
        ])
        
        # Multi-worker aggregation logic
        self.worker_data = []  # Stores dicts: {'memmaps': {...}, 'size': int}
        self.cumulative_sizes = [0]
        self.total_size = 0

        print(f"[Dataset] Loading {len(self.worker_dirs)} buffer folders...")

        for w_dir in self.worker_dirs:
            meta_path = os.path.join(w_dir, "metadata.pkl")
            current_size = 0
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as f:
                        current_size = pickle.load(f).get('current_size', 0)
                except:
                    pass
            
            if current_size > 0:
                memmaps = {}
                specs = DiskReplayBuffer.get_specs(current_size)
                
                try:
                    for name, spec in specs.items():
                        path = os.path.join(w_dir, f'{name}.npy')
                        memmaps[name] = np.memmap(path, dtype=spec['dtype'], mode='r', shape=spec['shape'])
                    
                    self.worker_data.append({'memmaps': memmaps, 'size': current_size})
                    self.total_size += current_size
                    self.cumulative_sizes.append(self.total_size)
                except Exception as e:
                    print(f"Warning: Failed to load {w_dir}: {e}")

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