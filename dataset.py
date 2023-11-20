from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, img_path, mask_path, cfg, transform=None):
        self.image = img_path
        self.mask = mask_path
        n_samples = len(self.image)
        self.cfg = cfg
        # 데이터 미리 섞어줌
        np.random.seed(self.cfg["seed"])
        idxs = np.random.permutation(range(n_samples))

        self.image = np.array(self.image)[idxs]
        self.mask = np.array(self.mask)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.image)  # 데이터셋 길이

    def __getitem__(self, i):
        image = np.array(Image.open(self.image[i]))
        mask = np.array(Image.open(self.mask[i]))
        data = self.transform(image=image, mask=mask)
        image = data["image"]
        mask = data["mask"]
        return image, mask
