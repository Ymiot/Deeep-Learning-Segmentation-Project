import argparse
import os
import glob
import random

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

def write_triplets(triplets, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for img_path, mask_path, label_path in triplets:
            # label_path may be empty string if not available (e.g. DRIVE test)
            f.write(",".join([img_path, mask_path, label_path]) + "\n")
    print(f"Wrote {len(triplets)} triplets -> {out_path}")

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

def make_phc_triplet(img_path, root, subset):
    # phc layout assumed: <root>/<subset>/images/*.jpg, <root>/<subset>/mask/*.png, <root>/<subset>/labels/*.png
    base = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(root, subset, "mask", base + ".png")
    label_path = os.path.join(root, subset, "labels", base + ".png")
    return img_path, mask_path, label_path

def make_drive_triplet(img_path, root, train=True):
    # DRIVE naming examples:
    # images: 21_training.tif
    # masks: 21_training_mask.gif
    # labels (training): 21_manual1.gif  (sometimes '21_manual1.gif' or '21_manual.gif' depending on dataset)
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]  # e.g. 21_training
    # mask has suffix "_mask.gif" in training; in test images names are like 01_test.tif
    # We'll try two mask patterns to be robust
    if train:
        mask_candidate = os.path.join(root, "training", "mask", stem + "_mask.gif")
        # label: try both possible conventions
        # 21_training -> 21_manual1.gif  OR stem.replace("_training","") + "_manual1.gif"
        base_no_suffix = stem.replace("_training", "").replace("_test", "")
        label_candidate1 = os.path.join(root, "training", "1st_manual", f"{base_no_suffix}_manual1.gif")
        label_candidate2 = os.path.join(root, "training", "1st_manual", f"{base_no_suffix}_manual.gif")
        label_path = label_candidate1 if os.path.exists(label_candidate1) else label_candidate2
    else:
        # For test, labels usually aren't provided in DRIVE; set label empty
        mask_candidate = os.path.join(root, "test", "mask", stem + "_mask.gif")
        label_path = ""
    mask_path = mask_candidate
    return img_path, mask_path, label_path

def collect_triplets(img_paths, dataset, root, subset_is_train):
    triplets = []
    missing = []
    for img in img_paths:
        if dataset == "phc":
            img_p, mask_p, label_p = make_phc_triplet(img, root, subset_is_train)
        else:
            img_p, mask_p, label_p = make_drive_triplet(img, root, train=(subset_is_train=="training"))
        ok = True
        if not os.path.exists(img_p):
            ok = False
            missing.append(("image", img_p))
        if not os.path.exists(mask_p):
            ok = False
            missing.append(("mask", mask_p))
        # label may be empty string (e.g. DRIVE test), else check
        if label_p and not os.path.exists(label_p):
            ok = False
            missing.append(("label", label_p))
        if ok:
            triplets.append((img_p, mask_p, label_p))
    return triplets, missing

def main():
    args = parse_args()

    if args.dataset == "retina":
        train_img_dir = os.path.join(args.root, "training", "images")
        test_img_dir = os.path.join(args.root, "test", "images")
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
        test_imgs_all = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
        if not train_imgs or not test_imgs_all:
            raise FileNotFoundError("Could not find DRIVE images. Check paths.")
        train_imgs_sel, val_imgs_sel = split_list(train_imgs, args.train_frac, args.seed)
        test_imgs_sel = sample_fraction(test_imgs_all, args.test_frac, args.seed)

        train_triplets, missing_train = collect_triplets(train_imgs_sel, "retina", args.root, "training")
        val_triplets, missing_val = collect_triplets(val_imgs_sel, "retina", args.root, "training")
        test_triplets, missing_test = collect_triplets(test_imgs_sel, "retina", args.root, "test")

        if missing_train or missing_val or missing_test:
            print("Warning: some expected files were missing. See list below:")
            for m in missing_train + missing_val + missing_test:
                print(m)

        write_triplets(train_triplets, os.path.join(args.output, "retina_train.txt"))
        write_triplets(val_triplets, os.path.join(args.output, "retina_val.txt"))
        write_triplets(test_triplets, os.path.join(args.output, "retina_test.txt"))

    else:
        # phc_data layout: /train/images/*.jpg, /test/images/*.jpg ; masks and labels inside mask/ and labels/
        train_img_dir = os.path.join(args.root, "train", "images")
        test_img_dir = os.path.join(args.root, "test", "images")
        train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
        test_imgs_all = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
        if not train_imgs or not test_imgs_all:
            raise FileNotFoundError("Could not find PhC images. Check paths.")
        train_imgs_sel, val_imgs_sel = split_list(train_imgs, args.train_frac, args.seed)
        test_imgs_sel = sample_fraction(test_imgs_all, args.test_frac, args.seed)

        train_triplets, missing_train = collect_triplets(train_imgs_sel, "phc", args.root, "train")
        val_triplets, missing_val = collect_triplets(val_imgs_sel, "phc", args.root, "train")
        test_triplets, missing_test = collect_triplets(test_imgs_sel, "phc", args.root, "test")

        if missing_train or missing_val or missing_test:
            print("Warning: some expected files were missing. See list below:")
            for m in missing_train + missing_val + missing_test:
                print(m)

        write_triplets(train_triplets, os.path.join(args.output, "phc_train.txt"))
        write_triplets(val_triplets, os.path.join(args.output, "phc_val.txt"))

