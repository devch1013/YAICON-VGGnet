import torch
import yaml
import sys
from tqdm.auto import tqdm
import os
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import datetime
from pathlib import Path
from embedding.trainer.utils.utils import *
import torch.nn.functional as F
from ImageBind.imagebind.models.imagebind_model import ModalityType


# from models.CRF.crf_model import crf


class Trainer:
    def __init__(self, base_dir: str, imagebind, t5, config_dir: str = "models/config.yaml"):
        self.save_ckpt = False
        self.base_dir = base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_config(config_dir)
        self.imagebind_model = imagebind
        self.t5_model = t5
        self.model = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.validation = False
        self.writer = None

    def _load_config(self, config_dir: str):
        with open(config_dir) as f:
            cfg = yaml.safe_load(f)
        print("Model Name: ", cfg["model-name"])
        return cfg

    def enable_ckpt(self, save_dir: str):
        assert save_dir is not None and self.base_dir is not None
        self.save_ckpt = True
        Path(os.path.join(save_dir)).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self.best_ckpt = []

    def enable_tensorboard(self, log_dir: str = None, save_image_log=False):
        self.save_image_log = save_image_log
        # print(self.save_image_log)
        current_time = datetime.datetime.now() + datetime.timedelta(hours=9)
        current_time = current_time.strftime("%m-%d-%H:%M")
        Path(os.path.join(self.base_dir, f"log")).mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            log_dir = f"log/{self.cfg['model-name']}__epochs-{self.cfg['train']['epoch']}_batch_size-{self.cfg['train']['batch-size']}__{current_time}"
        tensorboard_dir = os.path.join(self.base_dir, log_dir)
        self.writer = SummaryWriter(tensorboard_dir)
        print("TensorBoard logging enabled")

    def set_model(self, model_class, state_dict=None):
        """ """
        self.model = model_class(**self.cfg["model"]).to(self.device)
        if state_dict != None:
            self.model.load_state_dict(torch.load(state_dict))
        self._set_train_func()

    def set_pretrained_model(self, model):
        self.model = model.to(self.device)
        self._set_train_func()

    def _set_train_func(self):
        self.optimizer = get_optimizer(self.model, self.cfg["train"]["optimizer"])
        self.criterion = get_criterion(self.cfg["train"]["criterion"])
        self.scheduler = get_scheduler(self.optimizer, self.cfg["train"]["lr_scheduler"])

    def set_train_dataloader(
        self,
        dataset,
    ):
        print("data loader setting complete")
        self.train_dataloader = DataLoader(
            dataset, batch_size=self.cfg["train"]["batch-size"], shuffle=True, num_workers=16
        )

    def set_validation_dataloader(
        self,
        dataset,
    ):
        self.val_dataloader = DataLoader(
            dataset, batch_size=self.cfg["validation"]["batch-size"], shuffle=True, num_workers=16
        )
        self.validation = True

    def train(
        self,
        epoch: int = None,
    ):
        
        assert self.model is not None
        assert self.train_dataloader is not None

        print("batch size: ", self.cfg["train"]["batch-size"])

        if epoch == None:
            epoch = self.cfg["train"]["epoch"]

        for i in range(epoch):
            self._train(i)
            # if self.validation:
            #     self._validate(i)

            # test(model, device, test_loader, criterion)

            # if batch_idx % test_interval == test_interval - 1 or batch_idx == len(train_loader) - 1:
            #     test(model, device, test_loader, criterion, args)

        if self.writer is not None:
            self.writer.close()

    def _train(self, current_epoch):
        
        self.model.train()
        total_loss = 0
        batch_idx = 0
        print("train for epoch ", current_epoch)
        for data in tqdm(self.train_dataloader):
            # print('*', end="")
            with torch.no_grad():
                imagebind_embedding = self.imagebind_model(data, "text")[ModalityType.TEXT].to(torch.float32)
                t5_embedding = self.t5_model.encode_prompt(data, device="cuda")[0].to(torch.float32)

            self.optimizer.zero_grad()

            output = self.model(imagebind_embedding)

            loss, losses = self._get_loss(output=output, target=t5_embedding)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if self.writer is not None:
                self.writer.add_scalar(
                    "Train Loss",
                    loss.item(),
                    batch_idx + current_epoch * (len(self.train_dataloader)),
                )
                if type(losses) == dict:
                    for key, value in losses.items():
                        self.writer.add_scalar(
                            "Train " + key,
                            value,
                            batch_idx + current_epoch * (len(self.train_dataloader)),
                        )
                self.writer.add_scalar(
                    "learning rate", self.optimizer.param_groups[0]["lr"], current_epoch
                )
            batch_idx += 1
            # break

        print("Train loss=", total_loss / len(self.train_dataloader))
        if self.validation:
            total_val_loss = 0
            self.model.eval()
            print("validation of epoch ", current_epoch)
            with torch.no_grad():
                for data in tqdm(self.val_dataloader):
                    # print("data:", data)
                    
                    imagebind_embedding = self.imagebind_model(data, "text")[ModalityType.TEXT].to(torch.float32)
                    t5_embedding = self.t5_model.encode_prompt(data, device="cuda")[0].to(torch.float32)
                    outputs = self.model(imagebind_embedding)

                    # output = output.squeeze(dim=1)
                    val_loss, val_losses = self._get_loss(output=outputs, target=t5_embedding)
                    total_val_loss += val_loss.item()
                    output = outputs
                    # break

        total_val_loss = total_val_loss / len(self.val_dataloader)
        print("Validation loss=", total_val_loss)
        if self.scheduler != None:
            self.scheduler.step(total_val_loss) 
        if self.writer is not None:
            self.writer.add_scalar(
                "Train Loss per Epoch", total_loss / len(self.train_dataloader), current_epoch
            )
            self.writer.add_scalar(
                "Validation Loss per Epoch",
                total_val_loss,
                current_epoch,
            )

        if self.save_ckpt:
            ## Validation score 가장 높은 5개 저장
            current_time = datetime.datetime.now() + datetime.timedelta(hours=9)
            current_time = current_time.strftime("%m-%d-%H:%M")
            model_name = self.cfg["model-name"]
            ckpt_name = f"{model_name}_{current_epoch}_{current_time}_{total_val_loss}"
            if len(self.best_ckpt) < 5:
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_dir}/{ckpt_name}",
                )
                self.best_ckpt.append({"ckpt_name": ckpt_name, "score": total_val_loss})

            else:
                score_list = [obj["score"] for obj in self.best_ckpt]
                if total_val_loss < max(score_list):
                    drop_idx = score_list.index((min(score_list)))
                    drop_name = self.best_ckpt[drop_idx]["ckpt_name"]
                    os.remove(f"{self.save_dir}/{drop_name}")
                    self.best_ckpt[drop_idx] = {"ckpt_name": ckpt_name, "score": total_val_loss}
                    torch.save(
                        self.model.state_dict(),
                        f"{self.save_dir}/{ckpt_name}",
                    )

    def validate(self):
        total_val_loss = 0
        total_dice_score = 0
        self.model.eval()
        print("Validation")
        with torch.no_grad():
            for data, target in tqdm(self.val_dataloader):
                data, target = data.to(self.device, dtype=torch.float32), target.to(
                    self.device, dtype=torch.float32
                )
                output = F.sigmoid(self.model(data))
                # output = F.interpolate(output, (data.shape[1], data.shape[0]), mode="bilinear")
                target = target.unsqueeze(dim=1)
                # val_loss, val_losses = self._get_loss(output=outputs, target=target)
                # total_val_loss += val_loss.item()
                # output = torch.concat(outputs, dim=1).mean(dim=1).unsqueeze(1)
                # print(output[0])
                cv2.imwrite(
                    "outputmask.png",
                    output2mask(output[0], threshold=0.5).squeeze(0).cpu().numpy() * 255,
                )
                cv2.imwrite(
                    "target.png",
                    target[0].squeeze().cpu().numpy() * 255,
                )
                # print(data.shape)
                cv2.imwrite(
                    "input.png",
                    data[0].permute(1, 2, 0).cpu().numpy() * 255,
                )
                # print(output.shape)
                # print(target.shape)
                # dice = dice_loss(
                #     input=output2mask(output, threshold=0.5).squeeze(), target=target.squeeze()
                # ).item()
                dice = dice_coeff_batch(
                    input=output2mask(output), target=target.unsqueeze(dim=1)
                ).item()
                # print(dice)
                total_dice_score += dice
                # break
        print("final dice loss", total_dice_score / len(self.val_dataloader))

    def inference(self, x):
        self.model.eval()
        self.model(x)

    def test(model, device, test_loader, criterion, args=None):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    def _validate(self, current_epoch):
        if self.val_dataset is None:
            return
        self.model.eval()
        validation_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validation_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        validation_loss /= len(self.test_dataset)
        if self.writer is not None:
            self.writer.add_scalar("Validation Loss", validation_loss, current_epoch)
            self.writer.add_scalar(
                "Validation Accuracy",
                100.0 * correct / len(self.test_dataset.dataset),
                current_epoch,
            )

    def _get_loss(self, output, target):
        len_output = 1
        loss = self.criterion(output, target)
        return loss, {"Loss": loss / len_output}


if __name__ == "__main__":
    train_class = Trainer(base_dir="/home/ubuntu/yaiconvally/VIT")
    train_class.set_model(VIT)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_class.set_dataset("cifar10", transform=transform, val=True)
    train_class.enable_tensorboard()
    train_class.train(epoch=10)