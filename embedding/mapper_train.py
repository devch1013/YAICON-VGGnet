import torch

from torch.utils.data import random_split


from embedding.trainer.train.train_class_mapper_reprojection import Trainer
from embedding.trainer.models.mapping_model import BindNetwork
from embedding.trainer.utils.embedding_dataset_all import EmbeddingDataset
model_name = "mapping_model"
root_dir = "/home/work/YAI-Summer/YAICON/VGGnet"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_class = Trainer(base_dir=root_dir, config_dir=f"embedding/trainer/models/{model_name}.yaml")
    # model = TestModel()
    # model = model.to(device)
    train_class.set_model(
        BindNetwork,
    )
    print("dataset load start")
    dataset = EmbeddingDataset()
    print("dataset_load finished!!")
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, validate_dataset = random_split(dataset, [0.9, 0.1], generator = generator1)

    train_class.set_train_dataloader(dataset=train_dataset)
    train_class.set_validation_dataloader(dataset=validate_dataset)
    train_class.enable_ckpt(f"models/ckpt/{model_name}")
    train_class.enable_tensorboard()
    train_class.train()