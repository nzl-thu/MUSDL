import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pickle
from utils import *
from opts import *


def load_imgs(vname):
    transform = transforms.Compose([transforms.CenterCrop(H),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    path = os.path.join(frames_dir, vname)
    img_names = sorted(os.listdir(path))
    imgs = torch.zeros(num_frames, C, H, W)
    partition = np.linspace(0, len(img_names) - 1, num=num_frames, dtype=np.int)

    for i in range(num_frames):
        img_path = os.path.join(path, img_names[partition[i]])
        img = Image.open(img_path)
        img = transform(img)
        imgs[i] = img
    return imgs  # tchw


class VideoDataset(Dataset):
    def __init__(self, fold, mode, cls):
        super().__init__()
        self.cls = cls
        self.mode = mode

        with open(os.path.join(info_dir, 'label.pkl'), 'rb') as f:
            self.label_dict = pickle.load(f)
        self.load_fold(fold)

    def __len__(self):
        return len(self.name_list)

    def load_fold(self, fold):
        folds = [0, 1, 2, 3]
        if self.mode == 'train':
            folds.pop(fold)
        else:
            folds = [fold]

        with open(os.path.join(info_dir, 'splits.pkl'), 'rb') as f:
            cv_file = pickle.load(f)  # info of cross validation

        self.name_list = []
        all_list = cv_file[self.cls]
        for fold in folds:
            for vid in all_list[fold]:
                self.name_list.append(vid + '_capture1')  # only loads left view

    def __getitem__(self, item):
        imgs = load_imgs(self.name_list[item])
        name = self.name_list[item][:-9]  # *_capture1
        labels = torch.tensor(self.label_dict[name])
        return imgs, labels

