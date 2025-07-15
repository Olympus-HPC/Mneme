import os
import subprocess
from tqdm import tqdm

failed_dirs = []
base_dir = os.getcwd()
log_file = "make_failed_dirs.log"

# Clear previous log file
with open(log_file, "w") as f:
    pass

for entry in tqdm(os.listdir(base_dir)):
    dir_path = os.path.join(base_dir, entry)
    if os.path.isdir(dir_path):
        try:
            result = subprocess.run(
                ["make"],
                cwd=dir_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                failed_dirs.append(entry)
                with open(log_file, "a") as f:
                    f.write(f"{entry}\n")
        except Exception as e:
            failed_dirs.append(entry)
            with open(log_file, "a") as f:
                f.write(f"{entry} (Exception: {e})\n")

print(f"Number of folders where make failed: {len(failed_dirs)}")
print(f"See {log_file} for details.")
