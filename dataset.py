# dataset.py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class NumpyDataset(Dataset):
    def __init__(self, images_path, labels_path, image_size=(224,224)):
        self.images = np.load(images_path, allow_pickle=False)
        self.labels = np.load(labels_path, allow_pickle=False).astype("float32").reshape(-1)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),                # [0,1], CxHxW
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]                 # H,W,3 uint8
        img_t = self.transform(img)
        label = self.labels[idx]
        return img_t, label
