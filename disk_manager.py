import os
import pickle
import random
import time

import torch
from torch.utils.data import IterableDataset, get_worker_info
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




    # 3. Generate Random Permutation (The Magic Step)
    # This ensures that reading the final file sequentially = random sampling
    permutation = np.random.permutation(total_size)

    # 4. Process Key-by-Key to save RAM
    # We load all data for ONE key (e.g., 'spatial') into RAM, shuffle it, and write it.
    os.makedirs(output_dir, exist_ok=True)
    specs = DiskReplayBuffer.get_specs(total_size)
    
    for name, spec in specs.items():
        # A. Allocate memory for the full merged array
        # Note: We use a standard numpy array for fast in-memory shuffling, then write to file
        merged_data = np.empty(spec['shape'], dtype=spec['dtype'])
        
        # B. Load data from all workers into merged_data
        cursor = 0
        for idx, d in enumerate(staging_dirs):
            current_size = buffer_sizes[idx]
            if current_size == 0: continue
            
            src_path = os.path.join(d, f'{name}.npy')
            if os.path.exists(src_path):
                # Read from disk
                src_mm = np.memmap(src_path, dtype=spec['dtype'], mode='r', shape=(current_size, *spec['shape'][1:]))
                merged_data[cursor : cursor + current_size] = src_mm
                del src_mm # Close handle
            else:
                print(f"[DOWNLOADER] WARNING missing file {src_path}")
            
            cursor += current_size

        # C. Apply Shuffle
        shuffled_data = merged_data[permutation]
        
        # D. Write to Final Output (Memmap is good for writing large chunks)
        out_path = os.path.join(output_dir, f'{name}.npy')
        output_mm = np.memmap(out_path, dtype=spec['dtype'], mode='w+', shape=spec['shape'])
        output_mm[:] = shuffled_data[:]
        output_mm.flush()
        
        # E. Cleanup Memory
        del output_mm
        del merged_data
        del shuffled_data
        import gc; gc.collect()

    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump({'current_size': total_size}, f)

    print(f"[Aggregator] Aggregation complete at {output_dir}")
  
class ArmadaChunkDataset(IterableDataset):
    def __init__(self, data_root, seq_len=128):
        self.data_root = data_root
        self.seq_len = seq_len
        # We need the element shapes (ignoring the batch size dimension)
        full_specs = DiskReplayBuffer.get_specs(1)
        self.feature_specs = {k: {'dtype': v['dtype'], 'shape': v['shape'][1:]} for k, v in full_specs.items()}
        
        self.buffer_dirs = []
        self.refresh_buffer_list()

    def refresh_buffer_list(self):
        """Scans the data root for valid chunk directories."""
        if not os.path.exists(self.data_root):
            self.buffer_dirs = []
            return

        # Look for folders containing metadata.pkl
        candidates = sorted([
            os.path.join(self.data_root, d) for d in os.listdir(self.data_root) 
            if os.path.isdir(os.path.join(self.data_root, d))
        ])
        
        self.buffer_dirs = [d for d in candidates if os.path.exists(os.path.join(d, "metadata.pkl"))]
        # print(f"[Loader] Found {len(self.buffer_dirs)} valid chunks.")

    def _load_chunk(self, chunk_dir):
        """Loads one sequence (128 samples) from the chunk."""
        try:
            with open(os.path.join(chunk_dir, "metadata.pkl"), 'rb') as f:
                meta = pickle.load(f)
                total_size = meta['current_size']
        except:
            return None

        if total_size <= self.seq_len:
            return None
            
        start_idx = random.randint(0, total_size - self.seq_len)
        data_slice = {}

        for name, spec in self.feature_specs.items():
            file_path = os.path.join(chunk_dir, f"{name}.npy")
            if not os.path.exists(file_path): continue
            
            # Open memmap, copy, close
            full_shape = (total_size, *spec['shape'])
            mm = np.memmap(file_path, dtype=spec['dtype'], mode='r', shape=full_shape)
            
            # .copy() is CRITICAL here to detach from file handle for multiprocessing safety
            arr = mm[start_idx : start_idx + self.seq_len].copy()
            
            # Convert to Tensor immediately
            tensor = torch.from_numpy(arr)
            
            # Type handling
            if name in ['phases', 'active_ship_id', 'target_ship_id']:
                tensor = tensor.long()
            
            data_slice[name] = tensor
            del mm
            
        return data_slice

    def __iter__(self):
        """
        Infinite generator running on each worker.
        """
        worker_info = get_worker_info()
        if worker_info is not None:
            # We are in a worker process
            # Seed = Base Seed + Worker ID
            seed = (int(time.time()) + worker_info.id) % 2**32
            random.seed(seed)
            np.random.seed(seed)
        else:
            # We are in the main process (single-thread debugging)
            pass

        # If no data, yield nothing and finish (avoids infinite crash loops)
        if not self.buffer_dirs:
            return

        while True:
            # Pick random chunk
            chunk_path = random.choice(self.buffer_dirs)
            
            # Load data
            batch = self._load_chunk(chunk_path)
            
            if batch is not None:
                yield batch
            else:
                # If chunk was invalid/too small, remove it from local list to avoid retrying immediately
                if chunk_path in self.buffer_dirs:
                    self.buffer_dirs.remove(chunk_path)