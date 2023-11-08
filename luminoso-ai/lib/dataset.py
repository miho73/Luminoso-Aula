import copy
import os

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class RiddikulusDataset(Dataset):
    def __init__(self, path, device, transform):
        self.device = device
        self.transform = transform

        files_path = path + '/files'
        self.classes = os.listdir(files_path)

        self.ds = [
            [label, files_path + '/' + label + '/' + code] for label in self.classes for code in
            os.listdir(files_path + '/' + label)
        ]

        self.class_to_code = {label: code for code, label in enumerate(self.classes)}
        self.code_to_class = {code: label for code, label in enumerate(self.classes)}

        self.classes = [
            self.class_to_code[label] for label in self.classes
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = cv2.imread(self.ds[idx][1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        category = self.class_to_code[self.ds[idx][0]]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = torch.tensor(image).type(torch.FloatTensor).to(self.device)
        category = torch.tensor(category)
        label = category.type(torch.LongTensor).to(self.device)

        return image, label

    def visualize(self, idx=0, samples=10, cols=5):
        dataset = copy.deepcopy(self)
        dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(samples):
            image, lbl = dataset[idx+i]
            ax.ravel()[i].set_title(lbl.cpu().item(), fontsize=18, color='#ba3a15')
            ax.ravel()[i].imshow(image.cpu().numpy())
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()
