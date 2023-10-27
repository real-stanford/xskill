import os
import subprocess
import time
import numpy as np


def main():
    num_process = 10
    interval = int(610 / num_process)
    index = np.arange(0, 610, interval)
    for embodiment in ['robot', 'human']:
        # Execute each process in parallel.
        procs = []
        for start_eps in index:
            end_eps = start_eps + interval
            end_eps = np.clip(end_eps, a_min=0, a_max=603)
            procs.append(
                subprocess.Popen([  # pylint: disable=consider-using-with
                    "python3",
                    "create_kitchen_dataset.py",
                    f"start_eps={start_eps}",
                    f"end_eps={end_eps}",
                    f"embodiment={embodiment}",
                ]))

        # Wait for each process to terminate.
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
