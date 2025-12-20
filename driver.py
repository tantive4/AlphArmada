import subprocess
import sys
from configs import Config

def run_driver():
    # Loop through all iterations defined in your config
    for i in range(Config.ITERATIONS):
        print(f"\n[DRIVER] Launching subprocess for Iteration {i}...")
        
        # This is equivalent to typing: python self_play.py --iter 0
        # check=True will raise an error if self_play.py crashes
        try:
            subprocess.run(
                [sys.executable, "self_play.py", "--iter", str(i)], 
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[DRIVER] Error: Iteration {i} crashed!")
            break
            
        print(f"[DRIVER] Iteration {i} finished. Metal Cache Wiped (Process Terminated).")

if __name__ == "__main__":
    run_driver()