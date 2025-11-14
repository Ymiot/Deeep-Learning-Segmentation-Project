import os
import glob

split_dir = "splits"
split_files = glob.glob(os.path.join(split_dir, "*.txt"))

for split_file in split_files:
    print(f"\nChecking file: {split_file}")
    with open(split_file, "r") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")
    missing = 0
    for i, line in enumerate(lines[:10]):  # affiche les 10 premières lignes
        paths = line.strip().split(",")
        exists_list = [os.path.exists(p) for p in paths]
        print(f"{i+1}. {paths} -> exists: {exists_list}")
        if not all(exists_list):
            missing += 1

    if missing > 0:
        print(f"Warning: {missing} line(s) have missing files in {split_file}")
    else:
        print(f"All checked files exist for first 10 lines of {split_file}")
