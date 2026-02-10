import os
import vessl
import vessl.storage
from vessl.util.exception import VesslApiException
from configs import Config

# def create_worker_volume(worker_id : int) -> None:
#     volume_name = f"alpharmada-volume-worker-{worker_id:02d}"
#     try:
#         vessl.storage.create_volume(
#             name=volume_name,
#             storage_name="vessl-storage",
#         )
#         print(f"Created volume {volume_name}")
#     except VesslApiException as e:
#         # 409 Conflict: Duplicate entity
#         if e.status == 409 :
#             pass
#         else:
#             raise e


def upload_replay_result(worker_id : int, path: str=".") -> None:
    volume_name = f"alpharmada-volume-worker-{worker_id:02d}"

    vessl.storage.upload_volume_file(
        source_path=path,
        dest_storage_name="vessl-storage",
        dest_volume_name=volume_name,
    )

def download_model(path:str = Config.CHECKPOINT_DIR) -> None:
    vessl.storage.download_volume_file(
        source_storage_name="vessl-storage",
        source_volume_name="alpharmada-volume-model",
        dest_path=path,
    )

def upload_model(path:str = Config.CHECKPOINT_DIR) -> None:
    vessl.storage.delete_volume_file(
        storage_name="vessl-storage",
        volume_name="alpharmada-volume-model",
        path="",
        recursive=True,
    )


    # Find all checkpoint files
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.startswith('model_iter_') and f.endswith('.pth')]
    
    # Find the checkpoint with the highest iteration number
    latest_checkpoint_file = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, latest_checkpoint_file)

    vessl.storage.upload_volume_file(
        source_path=checkpoint_path,
        dest_storage_name="vessl-storage",
        dest_volume_name="alpharmada-volume-model",
        dest_path="",
    )
    vessl.storage.upload_volume_file(
        source_path=checkpoint_path,
        dest_storage_name="vessl-storage",
        dest_volume_name="alpharmada-volume-trainer",
        dest_path="model",
    )
