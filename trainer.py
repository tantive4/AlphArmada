import argparse
import os
import shutil
import time

import torch.optim as optim

from learning.model.big_deep import load_recent_model
from learning.params.configs import Config
from orchestration.alpharmada import AlphArmadaTrainer
from orchestration.storage_manager import upload_model


def train() -> None:
    """
    1. Manage the replay-buffer sliding window.
    2. Create a dataset from the remaining folders.
    3. Train and upload a new checkpoint.
    """
    buffer_root = Config.REPLAY_BUFFER_DIR
    if not os.path.exists(buffer_root):
        os.makedirs(buffer_root)
    all_buffers = sorted([
        os.path.join(buffer_root, d) for d in os.listdir(buffer_root)
        if os.path.isdir(os.path.join(buffer_root, d))
    ])

    num_chunk = len(all_buffers)

    min_window = 4
    max_window = 40

    if num_chunk < min_window:
        time.sleep(60)
        return
    elif num_chunk > max_window:
        to_delete = all_buffers[:-max_window]
        for p in to_delete:
            print(f"[TRAINER] Deleting old buffer: {p}")
            shutil.rmtree(p)
    chunk_ratio = num_chunk / max_window

    model, current_iter = load_recent_model()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE * chunk_ratio,
        weight_decay=Config.L2_LAMBDA,
    )

    trainer = AlphArmadaTrainer(model, optimizer)
    trainer.train_model(new_checkpoint=current_iter + 1)

    upload_model()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AlphArmada trainer.")
    parser.add_argument("--no_loop", dest="loop", action="store_false", default=True, help="Run once and exit")
    args = parser.parse_args()

    print(f"[TRAINER] Running on {Config.DEVICE}")
    while True:
        train()
        if not args.loop:
            break


if __name__ == "__main__":
    main()
