import torch.nn as nn
from einops import rearrange, repeat, reduce
import torch


class EmbeddingTransformer(nn.Module):
    def __init__(
        self,
        **cfgs,
        # batch_size = 8,
        # device: str = "cpu",
    ):
        super(EmbeddingTransformer, self).__init__()
        heads = 8
        mlp_dim = 2048
        encoder_depth = 2
        self.embedding = LinearEmbeddingLayer(
            token_num = 77,
            vector_size = 2048,
            batch_size = cfgs["batch_size"],
            device=cfgs["device"],
        )
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    dim=2048,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    linear_drop_rate=0,
                )
                for _ in range(encoder_depth)
            ]
        )
        
        self.final_linear = nn.Linear(2048,4096)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.final_linear(x)
        return x


class LinearEmbeddingLayer(nn.Module):
    def __init__(self, vector_size = 1024, token_num = 77, batch_size = 8, device="cpu"):
        super(LinearEmbeddingLayer, self).__init__()
        self.linear = nn.Linear(vector_size, vector_size)
        # self.positional_encoding = PositionalEncoding(
        #     dim=vector_size,
        #     patch_num=token_num,
        #     batch_size = batch_size,
        #     device=device,
        # )

    def forward(self, x):
        x = self.linear(x)
        # x = self.positional_encoding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim=768, patch_num=16, batch_size=8, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(patch_num, dim).to(device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, patch_num, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, dim, 2, dtype=torch.float)
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / dim))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / dim))
        self.encoding = repeat(self.encoding, "n d -> b n d", b=batch_size)
        

    def forward(self, x):
        
        return x + self.encoding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, att_drop_rate=0.1, out_drop_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.scale = dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_drop = nn.Dropout(att_drop_rate)
        self.out_drop = nn.Dropout(out_drop_rate)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        # print(self.dim)
        # print(self.heads)
        # print(self.dim % (self.heads * 3))
        # assert self.dim % (self.heads * 3) == 0
        x = self.qkv(x)
        q, k, v = rearrange(x, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        qk = nn.functional.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        qk = self.attention_drop(qk)
        out = qk @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.o(out)
        out = self.out_drop(out)
        return out


class MlpLayer(nn.Module):
    def __init__(self, dim, mlp_dim, linear_drop_rate=0.1):
        super(MlpLayer, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.drop1 = nn.Dropout(linear_drop_rate)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop2 = nn.Dropout(linear_drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.gelu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim=768, heads=8, mlp_dim=3072, linear_drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.msa_residual = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(dim),
                MultiHeadSelfAttention(
                    dim, heads, att_drop_rate=linear_drop_rate, out_drop_rate=linear_drop_rate
                ),
            )
        )
        self.mlp_residual = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(dim),
                MlpLayer(dim=dim, mlp_dim=mlp_dim, linear_drop_rate=linear_drop_rate),
            )
        )

    def forward(self, x):
        x = self.msa_residual(x)
        x = self.mlp_residual(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, network):
        super(ResidualBlock, self).__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)


class ClassifiacationHead(nn.Module):
    def __init__(self, dim=768, classes=100) -> None:
        super(ClassifiacationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, classes)

    def forward(self, x):
        # x = reduce(x, "b n d -> b d", "mean")
        x = x[:, 0, :]
        x = self.layer_norm(x)
        x = self.linear(x)
        return x


# from torchsummary import summary

if __name__ == "__main__":
    # summary(VIT(batch_size=2), (3, 64, 64), device="cpu")
    print(2048 % (24))
    model = EmbeddingTransformer()
    data = torch.zeros((10,77,2048))
    result = model(data)
    print(result.shape)
