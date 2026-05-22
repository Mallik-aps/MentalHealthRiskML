import subprocess
import sys

steps = [
    [sys.executable, "src/synthetic_data_generator.py", "--config", "config/config.yaml"],
    [sys.executable, "src/train_cv.py", "--config", "config/config.yaml"],
    [sys.executable, "src/interpretability.py", "--config", "config/config.yaml"],
    [sys.executable, "src/fairness_analysis.py", "--config", "config/config.yaml"],
]

for cmd in steps:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
print("All steps completed.")
