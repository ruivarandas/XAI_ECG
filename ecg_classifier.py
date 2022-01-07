import attr
from pathlib import Path
from data_management import DirManagement, DataPreparation
from train import train_and_eval, train_and_eval_logo
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import torchvision
from torchvision import models
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@attr.s(auto_attribs=True)
class ECGClassifier:
    configurations: dict
    config_path: str
    device: str = attr.ib(default='cuda:1', init=False)
    model: torchvision.models = attr.ib(default=None, init=False)
    optimizer: torch.optim = attr.ib(default=None, init=False)
    exp_lr_scheduler: torch.optim.lr_scheduler = attr.ib(default=None, init=False)
    dataloaders: dict = attr.ib(default=None, init=False)
    datasets_sizes: dict = attr.ib(default=None, init=False)
    class_names: list = attr.ib(default=None, init=False)

    @property
    def labels(self):
        if self.configurations['multiclass']:
            return 'labels_multi'
        else:
            return 'labels_bin'
    
    def labels_file(self, beat):
        raw_data_dir = Path(self.configurations["data_dir"]) / "raw_figures_no_grid"
        labels = Path(self.configurations["data_dir"]) / f"labels_{beat}"
        print(f"Labels path: {labels}")
        for filename in labels.iterdir():
            shutil.copy(filename, raw_data_dir / filename.stem)
            
    def _prepare_data(self):
        heartbeat = self.configurations['heartbeat']
        self.labels_file(heartbeat)
        if not self.configurations["dirs_already_prepared"]:
            if self.configurations["multiclass"]:
                dir_prep = DirManagement(Path(self.configurations["data_dir"]), self.configurations["labels_multi"], heartbeat, self.configurations)
            else:
                dir_prep = DirManagement(Path(self.configurations["data_dir"]), self.configurations["labels_bin"], heartbeat, self.configurations)
            train, val, test = dir_prep.create_datasets_paper_division()
            dir_prep.write_data(train, val, test)

            data_prep = DataPreparation(dir_prep.data_dir)
        else:
            data_prep = DataPreparation(Path(self.configurations["data_dir"]) / f"figures_{heartbeat}")

        sets = ['train', 'val', 'test']

        self.device = data_prep.device
        self.dataloaders, self.datasets_sizes, self.class_names = data_prep.create_dataloaders(
            self.configurations["batch_size"],
            self.configurations["shuffle_data"],
            self.configurations["number_workers"], sets)

    def _define_model(self):
        model = models.resnet50(pretrained=self.configurations["pretrained"])             
        n_feat = model.fc.in_features
        class_names = list(self.configurations[self.labels].keys())
        model.fc = nn.Linear(n_feat, len(class_names))
        self.model = model.to(self.device)

    def get_class_balance(self):
        class_balance = {}
        for label in self.configurations[self.labels]:
            class_balance[label] = 0
            
        data_dir = Path(self.configurations["data_dir"]) / "raw_figures"
        for folder in data_dir.iterdir():
            for signal in folder.glob("*.txt"):
                labels = np.loadtxt(signal.as_posix(), dtype=np.object)[1:, 1]
                for label in labels:
                    for label_aux in self.configurations[self.labels]:
                        if label in self.configurations[self.labels][label_aux]:
                            class_balance[label_aux] += 1
        total = 0
        weights_list = []
        for label in sorted(list(class_balance.keys())):
            total += class_balance[label]
            weights_list.append(class_balance[label])
        
        return total/np.array(weights_list)

    def _loss(self):
        if self.configurations["weighted_loss"]:
            weights = self.get_class_balance()
            weights = torch.FloatTensor(weights).to(self.device)
            loss = nn.CrossEntropyLoss(weight=weights)
        else:
            loss = nn.CrossEntropyLoss()
        return loss

    def _define_learning(self):
        """
        Add differential learning rate
        :return:
        """
        if self.configurations["diff_learn"]:
            parameters = [
                {'params': self.model.layer1.parameters(), 'lr': 1e-6},
                {'params': self.model.layer2.parameters(), 'lr': 1e-6},
                {'params': self.model.layer3.parameters(), 'lr': 1e-5},
                {'params': self.model.layer4.parameters(), 'lr': 1e-4},
                {'params': self.model.fc.parameters(), 'lr': 1e-4}
            ]
        else:
            parameters = self.model.parameters()
        
        self.optimizer = optim.Adam(parameters, weight_decay=self.configurations["weight_decay"],
                                    lr=self.configurations["learning_rate"])

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=self.configurations["decay_step"],
                                                    gamma=self.configurations["lr_scheduler_gamma"])

    def _save_model(self, model, metrics, epoch):
        now = datetime.now()
        path = Path.cwd() / f"models/label_{self.configurations['heartbeat']}"
        name = f"{self.configurations['model_name']}_{now.strftime('d_%d_t_%H_%M')}"
        trained_model_filepath = path / f"{name}.pth"
        model_config_filepath = path / f"{name}.json"
        torch.save(model, trained_model_filepath.as_posix())

        with open(self.config_path) as json_file:
            data = json.load(json_file)

        data["last_trained_model"] = trained_model_filepath.as_posix()
        data["epochs"] = epoch

        with open(self.config_path, 'w') as outfile:
            json.dump(data, outfile)
        data["metrics"] = metrics
        with open(model_config_filepath, 'w') as new_file:
            json.dump(data, new_file)

    def train_and_eval(self):
        self._prepare_data()
        print("Folders created and data prepared")
        self._define_model()
        self._define_learning()
        loss = self._loss()
        model, metrics, epoch = train_and_eval(self.model, loss, self.optimizer, self.exp_lr_scheduler, self.device,
                                    self.dataloaders, self.datasets_sizes, self.configurations["epochs"],
                                    self.configurations["early_stop"], self.configurations["multiclass"])
        self._save_model(self.model, metrics, epoch)
        for metric in metrics:
            if metric.split(" ")[0] != "best":
                self.plot(metrics[metric], metric, f"{metric}_per_epoch")

    def plot(self, plottable, ylabel='', name=''):
        now = datetime.now()
        plt.clf()
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.plot(plottable)
        plt.savefig(f'plots/label_{self.configurations["heartbeat"]}/{name}_{now.strftime("d_%d_t_%H_%M")}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    configure_seed(42)
    with open("config.json") as json_file:
            data = json.load(json_file)
            json_file.close()
    model_init = ECGClassifier(data, "config.json")
    model_init.train_and_eval()
