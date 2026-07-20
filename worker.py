import argparse
import time

import vessl

from learning.model.big_deep import load_model
from learning.params.configs import Config
from orchestration.alpharmada import AlphArmadaWorker
from orchestration.storage_manager import download_recent_model, upload_replay_result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def work(
    worker_id: int,
    deep_mcts_iteration: int = Config.MCTS_ITERATION,
    standard_mcts_iteration: int = Config.MCTS_ITERATION_FAST,
    batched_game_size: int = Config.PARALLEL_PLAY,
) -> None:
    """
    1. Download latest model from Vessl.
    2. Run self-play to generate replay buffer data.
    3. Upload replay buffer to Vessl.
    """
    download_recent_model()

    model = load_model()
    worker = AlphArmadaWorker(
        model,
        worker_id,
        deep_mcts_iteration=deep_mcts_iteration,
        standard_mcts_iteration=standard_mcts_iteration,
        batched_game_size=batched_game_size,
    )
    try:
        worker.self_play()
        upload_replay_result(worker_id)
    except Exception as e:
        print(f"[WORKER] Unknown Error occurred!!!! {e}")
        upload_replay_result(worker_id, upload_replay=False)

        vessl.log(payload={"error": worker_id})
        time.sleep(1)
        vessl.log(payload={"error": 0})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AlphArmada self-play worker.")
    parser.add_argument("--worker_id", type=int, required=True, help="Machine ID for multi-machine setup")
    parser.add_argument("--no_loop", dest="loop", action="store_false", default=True, help="Run once and exit")
    parser.add_argument(
        "--deep_mcts_iteration",
        "--deep-mcts-iteration",
        "--mcts_iteration",
        "--mcts-iteration",
        dest="deep_mcts_iteration",
        type=_positive_int,
        default=Config.MCTS_ITERATION,
        help=f"Deep-search MCTS iterations. Default: {Config.MCTS_ITERATION}",
    )
    parser.add_argument(
        "--standard_mcts_iteration",
        "--standard-mcts-iteration",
        "--mcts_iteration_fast",
        "--mcts-iteration-fast",
        dest="standard_mcts_iteration",
        type=_positive_int,
        default=Config.MCTS_ITERATION_FAST,
        help=f"Standard-search MCTS iterations. Default: {Config.MCTS_ITERATION_FAST}",
    )
    parser.add_argument(
        "--batched_game_size",
        "--batched-game-size",
        "--batch_game_size",
        "--batch-game-size",
        dest="batched_game_size",
        type=_positive_int,
        default=Config.PARALLEL_PLAY,
        help=f"Parallel self-play games per worker batch. Default: {Config.PARALLEL_PLAY}",
    )
    args = parser.parse_args()

    print(f"[WORKER] Running on {Config.DEVICE}")
    while True:
        work(
            args.worker_id,
            deep_mcts_iteration=args.deep_mcts_iteration,
            standard_mcts_iteration=args.standard_mcts_iteration,
            batched_game_size=args.batched_game_size,
        )
        if not args.loop:
            break


if __name__ == "__main__":
    main()
