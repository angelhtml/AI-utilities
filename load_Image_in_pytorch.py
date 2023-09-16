import torch
import glob
from torchvision.io import read_image
import torchvision.transforms as transforms


image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3]),
    ]
)


class IntelDataset(torch.utils.data.Dataset):  # inheritin from Dataset class
    def __init__(self, path, transforms):
        self.pictures = glob.glob(f"{path}/*")
        self.transforms = transforms

    def __len__(self):
        return len(self.folders)  # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        image = read_image(self.pictures[idx])
        image = self.transforms(image)
        label = image

        return image, label
