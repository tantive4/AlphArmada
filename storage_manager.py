import os
import datetime
import vessl
import vessl.storage
from configs import Config

def _format_time(start_time, end_time):
    elapsed_time = end_time - start_time
    miniutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    return f"{int(miniutes):02d}:{int(seconds):02d}"


def download_replay_result(worker_id : int, local_path: str="output") -> None:
    """for downloader"""
    volume_name = f"alpharmada-volume-worker-{worker_id:02d}"

    start_time = datetime.datetime.now()
    vessl.storage.download_volume_file(
        source_storage_name="vessl-storage",
        source_volume_name=volume_name,
        dest_path=local_path
    )
    end_time = datetime.datetime.now()
    time = _format_time(start_time, end_time)

    print(f"[DOWNLOAD] worker-{worker_id:02d} data ({time})")


def upload_replay_result(worker_id : int) -> None:
    """for worker"""

    start_time = datetime.datetime.now()

    # Upload Replay Buffer
    volume_name = f"alpharmada-volume-worker-{worker_id:02d}"
    path = Config.REPLAY_BUFFER_DIR
    
    vessl.storage.upload_volume_file(
        source_path=path,
        dest_storage_name="vessl-storage",
        dest_volume_name=volume_name
    )

    # Upload Sub Outputs
    vessl.storage.upload_volume_file(
        source_path="output",
        dest_storage_name="vessl-storage",
        dest_volume_name="alpharmada-volume-worker-common",
        dest_path=f"output_{worker_id:02d}"
    )

    # Upload Timestamp Flag (The "Commit" Signal)
    # Format: timestamp_YYYYMMDD_HHMMSS.txt
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flag_filename = f"timestamp_{timestamp}.txt"
    
    # Create empty file
    with open(flag_filename, 'w') as f:
        pass 
        
    vessl.storage.upload_volume_file(
        source_path=flag_filename,
        dest_storage_name="vessl-storage",
        dest_volume_name=volume_name,
        dest_path=os.path.join("timestemp", flag_filename)
    )
    
    end_time = datetime.datetime.now()
    time = _format_time(start_time, end_time)
    print(f"[UPLOAD] worker-{worker_id:02d} replay ({time})")

def download_model(local_path:str = Config.CHECKPOINT_DIR) -> None:
    """for worker"""
    model_list = vessl.list_model_volume_files(
        repository_name="BigDeep",
        model_number=1,
        path=""
    )
    model_path = str(model_list[-1].path)
    vessl.download_model_volume_file(
        repository_name="BigDeep",
        model_number=1,
        source_path=model_path,
        dest_path=os.path.join(local_path, "model_best.pth")
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
    print(f"[UPLOAD] {latest_checkpoint_file}")