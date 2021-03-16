import torch.utils.data.Dataset
from PIL import Image

class CamVidDataset(Dataset):
    def __init__(self, image_path, label_path, transform):
        self.image_paths = self.getPaths(image_path)
        self.label_paths = self.getPaths(label_path)
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.label_paths[index])
        x, y = transform(image), transform(mask)
        return x,y
    
    def __len__(self):
        return len(self.image_paths)
