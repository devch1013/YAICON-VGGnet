from torchvision import datasets
import torch
from torchvision import datasets
import torchvision.transforms as transforms


class MyDataLoader:
    def __init__(
        self,
        dataset_name="cifar100",
        transform=None,
        dataset_root: str = "/home/ubuntu/datasets/",
    ):
        self.dataset_root = dataset_root
        dataset_name = dataset_name.upper()
        if dataset_name == "CIFAR100":
            self.train_dataset, self.test_dataset = self._get_dataset(
                datasets.CIFAR100, dataset_name=dataset_name, transform=transform
            )
        elif dataset_name == "CIFAR10":
            self.train_dataset, self.test_dataset = self._get_dataset(
                datasets.CIFAR10, dataset_name=dataset_name, transform=transform
            )

        else:
            AssertionError("dataset_name is not supported")

    def _get_dataset(self, dataset_class, dataset_name, transform=None):
        train_dataset = dataset_class(
            root=self.dataset_root + dataset_name,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = dataset_class(
            root=self.dataset_root + dataset_name,
            train=False,
            download=True,
            transform=transform,
        )

        return train_dataset, test_dataset

    def _get_shuffle_loader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def get_train_loader(self, batch_size):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    def get_test_loader(self, batch_size):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

    def get_dataloader(self, batch_size):
        return self.get_train_loader(batch_size), self.get_test_loader(batch_size)

    def get_dataloader_with_validation(self, batch_size):
        testloader = self.get_test_loader(batch_size)
        train_dataset, validation_data = torch.utils.data.random_split(
            self.train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1)
        )
        trainloader = self._get_shuffle_loader(train_dataset, batch_size)
        validationloader = self._get_shuffle_loader(validation_data, batch_size)
        return trainloader, validationloader, testloader
