import argparse, os, glob, random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["phc","retina"], required=True,
                   help="retina = DRIVE layout (/training,/test). phc = generic images/masks layout.")
    p.add_argument("--root", required=True, help="Dataset root directory.")
    p.add_argument("--output", required=True, help="Output directory for split files.")
    p.add_argument("--train", type=float, default=0.7, help="Train fraction (for datasets where we split).")
    p.add_argument("--val", type=float, default=0.15, help="Val fraction (for datasets where we split).")
    # For generic PhC-like datasets
    p.add_argument("--images_dir", type=str, default="images", help="Relative images dir under root (PHC).")
    p.add_argument("--masks_dir", type=str, default="masks", help="Relative masks dir under root (PHC).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def write_list(paths, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Wrote {len(paths)} lines to {out_path}")

def main():
    args = parse_args()
    random.seed(args.seed)

    if args.dataset == "retina":
        # Expect official DRIVE structure:
        #   root/training/images/*.tif
        #   root/test/images/*.tif
        train_img_dir = os.path.join(args.root, "training", "images")
        test_img_dir = os.path.join(args.root, "test", "images")

        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
        test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))

        if len(train_imgs) == 0 or len(test_imgs) == 0:
            raise FileNotFoundError(
                f"No images found. Expected DRIVE at {args.root} with training/images and test/images."
            )

        # Split training into train/val; keep official test as test.
        indices = list(range(len(train_imgs)))
        random.shuffle(indices)
        n = len(indices)
        n_train = int(args.train * n)
        n_val = int(args.val * n)

        train_paths = [train_imgs[i] for i in indices[:n_train]]
        val_paths = [train_imgs[i] for i in indices[n_train:n_train+n_val]]
        test_paths = test_imgs  # official test list, no shuffle

        write_list(train_paths, os.path.join(args.output, "retina_train.txt"))
        write_list(val_paths, os.path.join(args.output, "retina_val.txt"))
        write_list(test_paths, os.path.join(args.output, "retina_test.txt"))

    else:
        # Generic PhC-style: root/images, root/masks
        img_dir = os.path.join(args.root, args.images_dir)
        files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No images found under {img_dir}. Adjust --images_dir or root."
            )

        indices = list(range(len(files)))
        random.shuffle(indices)
        n = len(indices)
        n_train = int(args.train * n)
        n_val = int(args.val * n)

        train_paths = [files[i] for i in indices[:n_train]]
        val_paths = [files[i] for i in indices[n_train:n_train+n_val]]
        test_paths = [files[i] for i in indices[n_train+n_val:]]

        write_list(train_paths, os.path.join(args.output, "phc_train.txt"))
        write_list(val_paths, os.path.join(args.output, "phc_val.txt"))
        write_list(test_paths, os.path.join(args.output, "phc_test.txt"))

if __name__ == "__main__":
    main()
