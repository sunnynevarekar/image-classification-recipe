import os
from PIL import Image
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

class ImageClassificationDataset(Dataset):
    def __init__(self, filepath, image_ids, labels, transforms=None):
        self.filepath = filepath
        self.image_ids = image_ids
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.filepath, self.image_ids[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        if self.labels:
            label = self.labels[idx]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.image_ids)


def load_annotations_from_folder(filepath):
    image_ids = []
    labels = []
    #directory names are labels 
    dir_names = sorted(os.listdir(filepath))
    for i, dir_name in enumerate(dir_names):
        ids = sorted(os.listdir(os.path.join(filepath, dir_name)))
        ids = [os.path.join(dir_name, id) for id in ids]
        image_ids.extend(ids)
        labels.extend([i]*len(ids))    

    return image_ids, labels, dir_names

def split_data(images, labels, **kwargs):
    return train_test_split(images, labels, **kwargs)
