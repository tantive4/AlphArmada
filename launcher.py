import vessl
import vessl.storage
import io
import time

# Define the base configuration template using f-string placeholders
# We replace '01' with '{worker_id}' in the name, description, and command
yaml_template = """name: alpharmada-worker-{worker_id}
description: AlphArmada Self Play Worker {worker_id}
tags:
  - worker
import:
  /root/alpharmada/: git://github.com/tantive4/alpharmada.git
resources:
  cluster: snu-eng-gtx1080
  preset: gpu-1
image: quay.io/vessl-ai/torch:2.1.0-cuda12.2-r3
run:
  - command: |-
      pip install --upgrade pip --quiet
      pip install torch --index-url https://download.pytorch.org/whl/cu126 --quiet
      pip install numpy shapely pillow cython numba scikit-image tqdm vessl --quiet > /dev/null 2>&1
      pip install --upgrade vessl --quiet

      python cython_compile/setup.py build_ext --inplace > /dev/null


      # Run the worker
      python -u main_run.py --mode worker --worker_id {worker_id}
    workdir: /root/alpharmada
"""

def launch_run():
    # Loop through worker IDs 01 to 20
    run_id = []
    for i in range(1, 21):
        # Format ID with leading zero (01, 02, ... 20)
        worker_id = f"{i:02d}"
        
        # Generate the specific YAML content for this worker
        current_yaml_body = yaml_template.format(worker_id=worker_id)
        current_file_name = f"alpharmada-worker-{worker_id}.yaml"

        print(f"Starting run for Worker {worker_id}...")

        # Create the run
        # We wrap the string in StringIO because the signature expects a TextIO object for 'yaml_file'
        with io.StringIO(current_yaml_body) as f:
            run = vessl.create_run(
                  yaml_file=f,
                  yaml_body=current_yaml_body,
                  yaml_file_name=current_file_name,
                  project_name="alpharmada"  # Overriding the default project
            )
            run_id.append(run.id)
            
        time.sleep(60)

    print("All runs initiated successfully.")
    for i in range(1, 21):
        print(f"Run {i} created with ID:", run_id[i-1])

def delete_volume_files():
    for i in range(1, 21):
        worker_id = f"{i:02d}"
        vessl.storage.delete_volume_file(
            storage_name="vessl-storage",
            volume_name=f"alpharmada-volume-worker-{worker_id}",
            path="",
            recursive=True
        )
        time.sleep(1)

if __name__ == "__main__":
    launch_run()
    # delete_volume_files()