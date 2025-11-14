import argparse, os, glob, random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["retina","phc"], required=True,
                   help="retina = DRIVE; phc = phc_data layout with train/test.")
    p.add_argument("--root", required=True, help="Root directory of dataset.")
    p.add_argument("--output", required=True, help="Output directory for split files.")
    p.add_argument("--train_frac", type=float, default=0.8,
                   help="Fraction of TRAIN folder used for training (rest for validation).")
    p.add_argument("--test_frac", type=float, default=0.2,
                   help="Fraction of TEST folder randomly sampled for test split.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def write_list(paths, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Wrote {len(paths)} -> {out_path}")

def split_list(file_list, frac_train, seed):
    random.seed(seed)
    idx = list(range(len(file_list)))
    random.shuffle(idx)
    n_train = int(frac_train * len(file_list))
    train = [file_list[i] for i in idx[:n_train]]
    val = [file_list[i] for i in idx[n_train:]]
    return train, val

def sample_fraction(file_list, frac, seed):
    random.seed(seed)
    n = int(frac * len(file_list))
    if n < 1:
        n = 1
    return random.sample(file_list, n)

def main():
    args = parse_args()

    if args.dataset == "retina":
        # DRIVE layout
        train_img_dir = os.path.join(args.root, "training", "images")
        test_img_dir = os.path.join(args.root, "test", "images")
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
        test_imgs_all = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
        if not train_imgs or not test_imgs_all:
            raise FileNotFoundError("Could not find DRIVE images. Check paths.")
        train_paths, val_paths = split_list(train_imgs, args.train_frac, args.seed)
        test_paths = sample_fraction(test_imgs_all, args.test_frac, args.seed)

        write_list(train_paths, os.path.join(args.output, "retina_train.txt"))
        write_list(val_paths, os.path.join(args.output, "retina_val.txt"))
        write_list(test_paths, os.path.join(args.output, "retina_test.txt"))

    else:
        # phc_data layout: /train/images/*.jpg, /test/images/*.jpg
        train_img_dir = os.path.join(args.root, "train", "images")
        test_img_dir = os.path.join(args.root, "test", "images")
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
        test_imgs_all = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
        if not train_imgs or not test_imgs_all:
            raise FileNotFoundError("Could not find PhC images. Check paths.")
        train_paths, val_paths = split_list(train_imgs, args.train_frac, args.seed)
        test_paths = sample_fraction(test_imgs_all, args.test_frac, args.seed)

        write_list(train_paths, os.path.join(args.output, "phc_train.txt"))
        write_list(val_paths, os.path.join(args.output, "phc_val.txt"))
        write_list(test_paths, os.path.join(args.output, "phc_test.txt"))

if __name__ == "__main__":
    main()
