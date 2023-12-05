import torch
import os

from embedding.models.imagebind import ImageBindEmbedding
from ImageBind.imagebind.models.imagebind_model import ModalityType
from embedding.text_dataset import TextDataset
from show_1.showone.pipelines.pipeline_for_embedding import TextEmbedding
from tqdm import tqdm

import numpy as np

pretrained_model_path = "show_1/show-1-base"
imagebind_model = ImageBindEmbedding()
# t5_model = T5Embedder("cuda")
t5_model = TextEmbedding.from_pretrained(
    pretrained_model_path, torch_dtype=torch.float16, variant="fp16"
)
t5_model.enable_model_cpu_offload()


def save_embedding(
    train_loader,
):
    j=0
    for data in tqdm(train_loader):
        with torch.no_grad():
            imagebind_embedding = imagebind_model(data, "text")[ModalityType.TEXT].cpu().numpy()
            t5_embedding = t5_model.encode_prompt(data, device="cuda")[0].cpu().numpy()
            
        for k in tqdm(range(imagebind_embedding.shape[0])):
            np.save(os.path.join("embedding/data/imagebind_embeddings", "ib_embeddings_each/{0:07d}.npy".format(j)), imagebind_embedding[k])
            np.save(os.path.join("embedding/data/t5_embeddings", "t5_embeddings_each/{0:07d}.npy".format(j)), t5_embedding[k])
            j += 1
        

        # if batch_idx % test_interval == test_interval - 1 or batch_idx == len(train_loader) - 1:
        #     test(model, device, test_loader, criterion, args)


if __name__ == "__main__":
    batch_size = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = TextDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    save_embedding(dataloader)