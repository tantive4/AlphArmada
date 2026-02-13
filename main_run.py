import argparse

import torch.optim as optim

from storage_manager import *
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
    worker.self_play()

    upload_replay_result(worker_id)

def train(num_worker) -> None:
    """
    1. Download latest replay buffer from Vessl
    2. Train the model on the replay buffer
    3. Upload the new model to Vessl
    """

    model, current_iter = load_recent_model()

    optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.L2_LAMBDA
            )
    
    trainer = AlphArmadaTrainer(model, optimizer, num_worker)
    trainer.train_model(new_checkpoint=current_iter + 1)

    upload_model()

def download_all(num_worker) -> None:
    """
    Continuously downloads data from all workers (1-20) in a loop.
    Designed to run in the background while the trainer runs.
    """
    
    for i in range(1, num_worker + 1): 
        try:
            target_dir = os.path.join(Config.REPLAY_BUFFER_DIR, f"worker_{i:02d}")

            download_replay_result(i, local_path=target_dir)
            
        except Exception as e:
            print(f"[DOWNLOAD] Error downloading worker {i}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="worker", help="Mode: worker / trainer / downloader")
    parser.add_argument("--worker_id", type=int, required=False, help="Machine ID for multi-machine setup")
    parser.add_argument("--num_worker", type=int, required=False, help="Total number of workers in multi-machine setup")
    args = parser.parse_args()
        
    
    if args.mode == "worker":
        download_replay_result(args.worker_id)
        while True:
            work(args.worker_id)

    elif args.mode == "trainer": 
        download_all(args.num_worker)
        while True:
            train(args.num_worker)

    elif args.mode == "downloader":
        download_all(args.num_worker)

if __name__ == "__main__":
    main()