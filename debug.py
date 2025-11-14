from lib.datasets.phc_dataset import PhCDataset
from torchvision import transforms

# Transformation simple
transform = transforms.ToTensor()

# Instancier le dataset avec le split train
dataset = PhCDataset("splits/phc_train.txt", transform=transform)

# Prendre le premier élément
item = dataset[0]

print("Item:", item)
print("Type:", type(item))
if isinstance(item, (tuple, list)):
    print("Nombre d'éléments dans le tuple/list:", len(item))
    print("Image tensor shape:", item[0].shape)
    print("Label tensor shape:", item[1].shape)
else:
    print("Returned item is not a tuple/list")
