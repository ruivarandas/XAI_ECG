from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split, LeavePGroupsOut, LeaveOneGroupOut
import attr
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import json
from os import sep
from random import random


@attr.s(auto_attribs=True)
class DirManagement:
    project_dir: str
    labels_dict: dict
    heartbeat: str
    config: dict
    
    @property
    def labels_list(self):
        return list(self.labels_dict.keys())
    
    @property
    def data_dir(self):
        return Path(self.project_dir) / f"figures_{self.heartbeat}"
    
    @property
    def raw_data_dir(self):
        return Path(self.project_dir) / "raw_figures_no_grid"
    
    @property
    def all_filenames(self):
        all_filenames = []
        for folder in self.raw_data_dir.iterdir():
            for image in folder.glob("*.png"):
                all_filenames.append(image)
        return all_filenames

    @property
    def all_labels(self):
        all_labels = {}
        for folder in self.raw_data_dir.iterdir():
            for txt in folder.glob("*.txt"):
                all_labels[folder.name] = np.loadtxt(Path(folder) / txt.name, dtype=np.object)[1:, 1]
        return all_labels

    @property
    def groups(self):
        groups = []
        for folder in self.raw_data_dir.iterdir():
            for txt in folder.glob("*.txt"):
                groups.append([int(folder.parts[-1])] * len(np.loadtxt(Path(folder) / txt.name, dtype=np.object)[1:, 1]))
        return np.concatenate(groups)

    @property
    def all_labels_list(self):
        labels = []
        for key in self.all_labels:
            labels.append(self.all_labels[key])
        return np.concatenate(labels)

    @staticmethod
    def split_groups(filenames, labels, groups, size):
        lpgo = LeavePGroupsOut(n_groups=size)
        flag = False
        for i, (train, test) in enumerate(lpgo.split(filenames, labels, groups=groups)):
            if random() > 0.95:
                flag = True
                train_filenames, train_labels, train_groups = np.array(filenames)[train], np.array(labels)[train], np.array(groups)[train]
                test_filenames, test_labels, test_groups = np.array(filenames)[test], np.array(labels)[test],np.array(groups)[test]
                break
        if not flag:
            train_filenames, train_labels, train_groups = np.array(filenames)[train], np.array(labels)[train], np.array(groups)[train]
            test_filenames, test_labels, test_groups = np.array(filenames)[test], np.array(labels)[test], np.array(groups)[test]

        return train_filenames, test_filenames, train_groups, train_labels

    def create_datasets_LeaveOneGroupOut(self, test_size, val_size):
        # Get 10% for the test set
        n_test_groups = int(len(set(self.groups))*test_size)
        train, test_filenames, train_groups, train_labels = self.split_groups(self.all_filenames, self.all_labels_list, self.groups, n_test_groups)
        n_val_groups = int(len(set(train_groups))*(val_size/(1-test_size)))
        train_filenames, val_filenames, _, _ = self.split_groups(train, train_labels, train_groups, n_val_groups)
        return train_filenames, val_filenames, test_filenames

    def create_datasets_paper_division(self):
        train_filenames, val_filenames, test_filenames = [], [], []
        for folder in self.raw_data_dir.iterdir():
            try:
                if int(folder.stem) in self.config["test_patients"]:
                    test_filenames.append(list(folder.glob("*.png")))
                elif int(folder.stem) in self.config["train_patients"]:
                    train_filenames.append(list(folder.glob("*.png")))
                elif int(folder.stem) in self.config["val_patients"]:
                    val_filenames.append(list(folder.glob("*.png")))
            except ValueError:
                pass
        return np.concatenate(train_filenames), np.concatenate(val_filenames), np.concatenate(test_filenames)

    def _create_new_dirs(self):
        """
        create new organized directories
        """
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        Path.mkdir(self.data_dir)
        datasets = ["train", "val", "test"]
        for dataset in datasets:
            dataset_dir = self.data_dir / dataset
            Path.mkdir(dataset_dir)
            for label in self.labels_list:
                Path.mkdir(dataset_dir / label)
    
    def write_data(self, train_filenames, val_filenames, test_filenames):
        """

        copy the images from raw dir to the new directory
        """
        self._create_new_dirs()
        all_labels = self.all_labels
        for dataset in [("train", train_filenames), ("val", val_filenames), ("test", test_filenames)]:
            print(dataset[0])
            for i, filename in enumerate(dataset[1]):
                print(f"{i+1}/{len(dataset[1])}", end='\r')
                signal, segment = str(filename).split(sep)[-2:]
                segment = segment.split('_')[0]
                # print(all_labels[signal][int(segment)], self.labels_dict["normal"])
                for label in self.labels_dict:
                    if all_labels[signal][int(segment)] in self.labels_dict[label]:
                        shutil.copy(filename, self.data_dir / dataset[0] / label / f"{filename.stem}_{signal}.png")
                        
                        
@attr.s(auto_attribs=True)
class DataPreparation:
    data_dir: Path
    device: str = attr.ib(default=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), init=False)

    @staticmethod
    def data_transformations():
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((200, 1500)),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop((200, 1500)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.CenterCrop((200, 1500)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        return data_transforms

    @staticmethod
    def normalize_figure(images):
        for i, image in enumerate(images):
            images[i] = (image - torch.min(image))/(torch.max(image) - torch.min(image))

        return images
    
    def create_dataloaders(self, batch_size, shuffle, num_workers, sets=('train', 'val', 'test')):
        data_transforms = self.data_transformations()
        image_datasets = {x: datasets.ImageFolder((self.data_dir / x).as_posix(), data_transforms[x]) for x in sets}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) for x in sets}
        dataset_sizes = {x: len(image_datasets[x]) for x in sets}
        class_names = image_datasets['train'].classes
        return dataloaders, dataset_sizes, class_names
    
    @staticmethod
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

if '__main__' == __name__:
    with open("config.json") as json_file:
            data = json.load(json_file)
            json_file.close()
    dirmanag = DirManagement('.', {"abnormal": ["A", "a", "J", "S", "V", "E", "F"],
                                             "normal": ["N", "L", "R", "e", "j"]}, "mid", data)
    train, val, test = dirmanag.create_datasets_paper_division()
    dirmanag.write_data(train, val, test)
