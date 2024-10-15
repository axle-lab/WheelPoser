import os
import subprocess

def run_python_scripts_in_directory(directory):
    # List all files in the given directory
    files = os.listdir(directory)

    # Filter out only Python (.py) files
    python_files = [f for f in files if f.endswith('.py')]

    # Sort files to ensure they are executed in order
    python_files.sort()
    # print(python_files)

    # Loop through each Python file and execute it
    for file in python_files:
        if 'Finetune' in file:
            filepath = os.path.join(directory, file)
            print(f"Running {filepath}...")
            subprocess.run(['python', filepath])

# # # Example usage

directory = 'scripts/2. Experiments/2.2 TransPose_WheelPoser/2.2.0 Training/2.2.0.1 Leaf2Full'  # Replace with the path to your directory
run_python_scripts_in_directory(directory)

directory = 'scripts/2. Experiments/2.2 TransPose_WheelPoser/2.2.0 Training/2.2.0.2 Full2Pose'  # Replace with the path to your directory
run_python_scripts_in_directory(directory)

print("Training done!")