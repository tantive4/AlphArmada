import os
import datetime
import posixpath
import re
import vessl
import vessl.storage
from learning.params.configs import Config

def _format_time(start_time, end_time):
    elapsed_time = end_time - start_time
    miniutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    return f"{int(miniutes):02d}:{int(seconds):02d}"

def _latest_model_path(model_files) -> str:
    model_paths = [str(f.path) for f in model_files]
    versioned_paths = [
        path for path in model_paths
        if re.search(r"model_iter_(\d+)\.pth$", os.path.basename(path))
    ]
    if versioned_paths:
        return max(
            versioned_paths,
            key=lambda path: int(re.search(r"model_iter_(\d+)\.pth$", os.path.basename(path)).group(1))
        )
    if not model_paths:
        raise FileNotFoundError("No model files found in BigDeep model repository.")
    return sorted(model_paths)[-1]

def get_worker_timestamp(worker_id: int) -> str:
    """
    Checks the remote volume for the latest timestamp file in 'timestamp/' folder.
    Returns the filename (e.g., 'timestamp_20260102_120000.txt') or None if not found.
    """
    try:
        # List files in the 'timestamp' directory of the volume
        files = vessl.storage.list_volume_files(
            storage_name="vessl-storage",
            volume_name="alpharmada-worker-common",
            path=posixpath.join(f"output_{worker_id:02d}", "timestamp")
        )
        # Filter for timestamp files and sort to get the latest
        timestamp_files = [f.path for f in files if "timestamp_" in f.path]
        if not timestamp_files:
            return ""
        return sorted(timestamp_files)[-1]  # Return the latest one
    except Exception as e:
        print(f"[CHECK] Error checking worker-{worker_id:02d}: {e}")
        return ""

def download_replay_result(worker_id : int, local_path: str="output") -> None:
    """for downloader"""
    volume_name = f"alpharmada-worker-{worker_id:02d}"

    start_time = datetime.datetime.now()
    vessl.storage.download_volume_file(
        source_storage_name="vessl-storage",
        source_volume_name=volume_name,
        dest_path=local_path
    )
    end_time = datetime.datetime.now()
    time = _format_time(start_time, end_time)

    print(f"[DOWNLOAD] worker-{worker_id:02d} data ({time})\n")


def upload_replay_result(worker_id : int, upload_replay : bool=True) -> None:
    """for worker"""

    start_time = datetime.datetime.now()

    # Upload Replay Buffer
    volume_name = f"alpharmada-worker-{worker_id:02d}"
    path = Config.REPLAY_BUFFER_DIR
    
    if upload_replay:
        vessl.storage.upload_volume_file(
            source_path=path,
            dest_storage_name="vessl-storage",
            dest_volume_name=volume_name
        )

    # Upload Sub Outputs
    vessl.storage.upload_volume_file(
        source_path="output",
        dest_storage_name="vessl-storage",
        dest_volume_name="alpharmada-worker-common",
        dest_path=f"output_{worker_id:02d}"
    )

    if upload_replay:
        # Upload Timestamp Flag (The "Commit" Signal)
        # Format: timestamp_YYYYMMDD_HHMMSS.txt
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        flag_filename = f"timestamp_{timestamp}.txt"

        try:
            with open(flag_filename, 'w') as f:
                pass

            vessl.storage.upload_volume_file(
                source_path=flag_filename,
                dest_storage_name="vessl-storage",
                dest_volume_name="alpharmada-worker-common",
                dest_path=posixpath.join(f"output_{worker_id:02d}", "timestamp")
            )
        finally:
            if os.path.exists(flag_filename):
                os.remove(flag_filename)
    
    end_time = datetime.datetime.now()
    time = _format_time(start_time, end_time)
    print(f"[UPLOAD] worker-{worker_id:02d} replay ({time})\n")

def download_recent_model(local_path:str = Config.CHECKPOINT_DIR, save_best:bool = True) -> None:
    """for worker"""
    os.makedirs(local_path, exist_ok=True)
    model_list = vessl.list_model_volume_files(
        repository_name="BigDeep",
        model_number=1,
        path=""
    )
    model_path = _latest_model_path(model_list)

    dest_name = "model_best.pth" if save_best else os.path.basename(model_path)
    dest_path = os.path.join(local_path, dest_name)
    vessl.download_model_volume_file(
        repository_name="BigDeep",
        model_number=1,
        source_path=model_path,
        dest_path=dest_path
    )
    print(f"[DOWNLOAD] {model_path}")

def download_model_version(version:int, local_path:str = Config.CHECKPOINT_DIR) -> None:
    """for evaluator"""
    os.makedirs(local_path, exist_ok=True)
    model_path = f"model_iter_{version:03d}.pth"
    dest_path = os.path.join(local_path, model_path)
    vessl.download_model_volume_file(
        repository_name="BigDeep",
        model_number=1,
        source_path=model_path,
        dest_path=dest_path
    )
    print(f"[DOWNLOAD] {model_path}")

def upload_model(local_path:str = Config.CHECKPOINT_DIR) -> None:
    """for trainer"""
    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(local_path) if f.startswith('model_iter_') and f.endswith('.pth')]
    
    # Find the checkpoint with the highest iteration number
    latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(local_path, latest_checkpoint_file)

    vessl.upload_model_volume_file(
        repository_name="BigDeep",
        model_number=1,
        source_path=checkpoint_path,
        dest_path=latest_checkpoint_file
    )
    print(f"[UPLOAD] {latest_checkpoint_file}\n") 
