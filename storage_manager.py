import os
import vessl
import vessl.storage
from configs import Config


def upload_replay_result(worker_id : int, path: str="output") -> None:
    volume_name = f"alpharmada-volume-worker-{worker_id:02d}"

    # Renew the output/game_visuals directory
    if worker_id == 1:
        vessl.storage.delete_volume_file(
            storage_name="vessl-storage",
            volume_name=volume_name,
            path="output/game_visuals",
            recursive=True
        )

    vessl.storage.upload_volume_file(
        source_path=path,
        dest_storage_name="vessl-storage",
        dest_volume_name=volume_name,
    )

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