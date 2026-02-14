import argparse
import shutil
import time

import torch.optim as optim

from storage_manager import *
from disk_manager import aggregate_staging_buffers
from big_deep import load_recent_model, load_model
from alpharmada import AlphArmadaWorker, AlphArmadaTrainer
from configs import Config

def work(worker_id: int) -> None:
    """
    1. Download latest model from Vessl
    2. Run self-play to generate replay buffer data
    3. Upload replay buffer to Vessl
    """
    download_model()

    model = load_model()
    worker = AlphArmadaWorker(model, worker_id)
    try:
        worker.self_play()
    except Exception as e:
        print(f"[WORKER] Unknown Error occured!!!! {e}")

    upload_replay_result(worker_id)

def train() -> None:
    """
    1. Manage Sliding Window (keep max 10 folders in 'replay_buffers')
    2. Create dataset from remaining folders
    3. Train
    """

    # --- 1. Sliding Window Management ---
    buffer_root = Config.REPLAY_BUFFER_DIR
    if not os.path.exists(buffer_root):
        os.makedirs(buffer_root)
        download_model(save_best=False)
    all_buffers = sorted([
        os.path.join(buffer_root, d) for d in os.listdir(buffer_root) 
        if os.path.isdir(os.path.join(buffer_root, d))
    ])
    
    num_chunk = len(all_buffers)
    
    MIN_WINDOW = 4
    MAX_WINDOW = 40

    if num_chunk < MIN_WINDOW: 
        print("[TRAINER] Not enough data")
        time.sleep(60)
        return
    elif num_chunk > MAX_WINDOW:
        to_delete = all_buffers[:-MAX_WINDOW] # Keep the last 10
        for p in to_delete:
            print(f"[SlidingWindow] Deleting old buffer: {p}")
            shutil.rmtree(p)
    chunk_ratio = num_chunk / MAX_WINDOW

    # --- 2. Train ---
    model, current_iter = load_recent_model()

    optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE * chunk_ratio, 
            weight_decay=Config.L2_LAMBDA
            )
    
    trainer = AlphArmadaTrainer(model, optimizer)
    trainer.train_model(new_checkpoint=current_iter + 1)

    upload_model()

def download_all(num_worker) -> None:
    """
    Downloader Logic:
    1. Loop 1..20 workers
    2. Check for NEW timestamp
    3. Download to staging/replay{01..32}
    4. If staging full, aggregate to replay_buffers/{timestamp} and reset.
    """
    
    worker_timestamps = {} # {worker_id: last_seen_timestamp_string}
    staging_dir = "staging"
    output_dir = Config.REPLAY_BUFFER_DIR
    
    staging_idx = 1
    MAX_STAGING = 8
    
    if os.path.exists(staging_dir):
        # Clean start to ensure index alignment
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    print("[Downloader] Started monitoring workers...")

    while True:
        data_downloaded_this_loop = False
        
        for i in range(1, num_worker + 1): 
            time.sleep(2)
            try:
                # Check remote timestamp
                latest_ts = get_worker_timestamp(i)
                
                if latest_ts and latest_ts != worker_timestamps.get(i):
                    # Found new data!
                    target_path = os.path.join(staging_dir, f"replay{staging_idx:02d}")
                    
                    print(f"[Downloader] New data from Worker {i}: {latest_ts} -> Slot {staging_idx}")
                    download_replay_result(i, local_path=target_path)

                    # Update state
                    worker_timestamps[i] = latest_ts
                    staging_idx += 1
                    data_downloaded_this_loop = True

                    # Check aggregation trigger
                    if staging_idx > MAX_STAGING:
                        print("[Downloader] Staging full (32/32). Aggregating...")
                        
                        agg_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        final_output_path = os.path.join(output_dir, agg_timestamp)

                        aggregate_staging_buffers(staging_dir, final_output_path)
                        
                        # Reset Staging
                        
                        shutil.rmtree(staging_dir)
                        os.makedirs(staging_dir)
                        print("[Downloader] Cleared staging area\n")
                        staging_idx = 1
                        
            except Exception as e:
                print(f"[Downloader] Error processing worker {i}: {e}")
        
        if not data_downloaded_this_loop:
            time.sleep(60) # Sleep if no new data to avoid spamming Vessl API

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="worker", help="Mode: worker / trainer / downloader")
    parser.add_argument("--no_loop", dest="loop", action="store_false", default=True, help="type main_run.py --no_loop for single run setup")
    parser.add_argument("--worker_id", type=int, required=False, help="Machine ID for multi-machine setup")
    parser.add_argument("--num_worker", type=int, required=False, help="Total number of workers in multi-machine setup")
    args = parser.parse_args()
        
    
    if args.mode == "worker":
        while True:
            work(args.worker_id)
            if args.no_loop : break

    elif args.mode == "trainer":
        while True:
            train()

    elif args.mode == "downloader":
        download_all(args.num_worker)

if __name__ == "__main__":
    main()