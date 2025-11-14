import os
import glob

SPLIT_DIR = "splits"

def check_split_file(file_path, max_lines=10):
    print(f"\nChecking file: {file_path}")
    if not os.path.exists(file_path):
        print("  -> File does not exist!")
        return
    
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines: {len(lines)}")
    
    for i, line in enumerate(lines[:max_lines]):
        paths = [p.strip() for p in line.split(",")]
        exists = [os.path.exists(p) for p in paths]
        print(f"{i+1}. {paths} -> exists: {exists}")
    
    print(f"All checked files exist for first {min(max_lines, len(lines))} lines of {os.path.basename(file_path)}")

def main():
    split_files = glob.glob(os.path.join(SPLIT_DIR, "*.txt"))
    for f in split_files:
        check_split_file(f)

if __name__ == "__main__":
    main()
