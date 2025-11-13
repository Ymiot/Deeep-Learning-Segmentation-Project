import argparse, os, glob, random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["phc","retina"], required=True)
    p.add_argument("--root", required=True, help="Root folder of dataset containing 'images' subfolder")
    p.add_argument("--output", required=True, help="Output directory for split index files")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    img_dir = os.path.join(args.root, "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    indices = list(range(len(files)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(args.train * n)
    n_val = int(args.val * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    os.makedirs(args.output, exist_ok=True)
    prefix = args.dataset
    for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(os.path.join(args.output, f"{prefix}_{name}.txt"), "w") as f:
            for i in idxs:
                f.write(str(i)+"\n")
    print(f"Splits written for {args.dataset}: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

if __name__ == "__main__":
    main()
