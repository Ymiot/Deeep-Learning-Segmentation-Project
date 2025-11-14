from lib.datasets.retina_dataset import RetinaDataset
from lib.datasets.phc_dataset import PhCDataset

def test_dataset(split_file, dataset_type="phc"):
    with open(split_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    if dataset_type == "phc":
        ds = PhCDataset(lines)
    else:
        ds = RetinaDataset(lines)

    print(f"{dataset_type} dataset length: {len(ds)}")
    for i in range(min(5, len(ds))):
        sample = ds[i]
        print(f"{i}: ", sample if dataset_type=="phc" else {k: v.size() for k,v in sample.items()})

# Exemple
test_dataset("splits/phc_train.txt", "phc")
test_dataset("splits/retina_train.txt", "retina")
