import argparse
import datetime
import os
import shutil
import time

from learning.params.configs import Config
from learning.replay.disk_manager import aggregate_staging_buffers
from orchestration.downloader import load_state, save_state
from orchestration.storage_manager import download_replay_result, get_worker_timestamp


def download_all(num_worker: int) -> None:
    """
    1. Monitor workers for new replay timestamps.
    2. Download new replay data into staging slots.
    3. Aggregate full staging batches into the replay-buffer directory.
    """
    worker_timestamps = load_state()
    staging_dir = "staging"
    output_dir = Config.REPLAY_BUFFER_DIR

    staging_idx = 1
    max_staging = 8

    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    print("[DOWNLOADER] Started monitoring workers...")

    while True:
        data_downloaded_this_loop = False

        for i in range(1, num_worker + 1):
            time.sleep(2)
            try:
                latest_ts = get_worker_timestamp(i)

                if latest_ts and latest_ts != worker_timestamps.get(i):
                    target_path = os.path.join(staging_dir, f"replay{staging_idx:02d}")

                    print(f"[DOWNLOADER] New data from Worker {i}: {latest_ts} -> Slot {staging_idx}")
                    download_replay_result(i, local_path=target_path)

                    worker_timestamps[i] = latest_ts
                    save_state(worker_timestamps)
                    staging_idx += 1
                    data_downloaded_this_loop = True

                    if staging_idx > max_staging:
                        print("[DOWNLOADER] Staging full. Aggregating...")

                        agg_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        final_output_path = os.path.join(output_dir, agg_timestamp)

                        aggregate_staging_buffers(staging_dir, final_output_path)

                        shutil.rmtree(staging_dir)
                        os.makedirs(staging_dir)
                        print("[DOWNLOADER] Cleared staging area\n")
                        staging_idx = 1

            except Exception as e:
                print(f"[DOWNLOADER] Error processing worker {i}: {e}")

        if not data_downloaded_this_loop:
            time.sleep(60)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AlphArmada replay downloader.")
    parser.add_argument("--num_worker", type=_positive_int, required=True, help="Total number of workers")
    args = parser.parse_args()

    print(f"[DOWNLOADER] Running on {Config.DEVICE}")
    download_all(args.num_worker)


if __name__ == "__main__":
    main()
