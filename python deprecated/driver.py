import subprocess
import sys
from configs import Config

def run_driver():
    print(f"[SETUP DEVICE] {Config.DEVICE}")
    # Loop through all iterations defined in your config
    for i in range(Config.ITERATIONS):
        print(f"\n[DRIVER] Launching Iteration {i+1}/{Config.ITERATIONS }")
        
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
            

if __name__ == "__main__":
    run_driver()