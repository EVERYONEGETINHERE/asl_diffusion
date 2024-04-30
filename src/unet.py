import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              padding=1,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        return y

class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 context_emb_dim: int):
        super().__init__()
        hidden_channels = max(in_channels, out_channels)
        self.time_emb_mlp = EmbeddingFC(time_emb_dim, hidden_channels)
        self.context_emb_mlp = EmbeddingFC(context_emb_dim, hidden_channels)
        self.block1 = Block(in_channels, hidden_channels)
        self.block2 = Block(hidden_channels, out_channels)
        self.res = nn.Identity() if in_channels==out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        y = self.block1(x)
        t_emb = self.time_emb_mlp(time_embedding)
        if context_embedding is not None:
            t_emb += self.context_emb_mlp(context_embedding)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        y = y + t_emb
        y = self.block2(y) + self.res(x)
        return y

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.query = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.key = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.value = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.projection = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor):
        h = self.norm(x)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        B, C, H, W = q.shape
        q = q.reshape(B, C, H*W)
        k = k.reshape(B, C, H*W)
        
        q = q.permute(0, 2, 1)              # -> (B, H*W, C)
        scores = torch.bmm(q, k)            # (B, H*W, C) @ (B, C, H*W) -> (B, H*W, H*W)
        scores = scores * (int(C)**(-0.5))  # normalization for unit std
        scores = F.softmax(scores, dim=2)
        
        v = v.reshape(B, C, H*W)
        scores = scores.permute(0, 2, 1)    # -> (B, H*W, H*W)
        h = torch.bmm(v, scores)            # (B, C, H*W) @ (B, H*W, H*W) -> (B, C, H*W)
        h = h.reshape(B, C, H, W)
        h = self.projection(h)
        return x + h

class Downsample(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=2,
                              stride=2)
    def forward(self, x: torch.Tensor):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self,
                 in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
    def forward(self, x):
        h = F.interpolate(x, scale_factor=2)
        return self.conv(h)

class UNetDown(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 context_emb_dim: int):
        super().__init__()
        self.resblock = ResnetBlock(in_channels, out_channels, time_emb_dim, context_emb_dim)
        self.downsample = Downsample(out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        h = self.resblock(x, time_embedding, context_embedding)
        y = self.downsample(h)
        return y, h

class UNetMiddle(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 context_emb_dim: int):
        super().__init__()
        self.resblock1 = ResnetBlock(in_channels, out_channels, time_emb_dim, context_emb_dim)
        self.attblock = SelfAttentionBlock(out_channels)
        self.resblock2 = ResnetBlock(out_channels, out_channels, time_emb_dim, context_emb_dim)
    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        y = self.resblock1(x, time_embedding, context_embedding)
        y = self.attblock(y)
        y = self.resblock2(y, time_embedding, context_embedding)
        return y

class UNetUp(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 context_emb_dim: int):
        super().__init__()
        self.upsample = Upsample(in_channels)
        self.resblock = ResnetBlock(in_channels+in_channels, out_channels, time_emb_dim, context_emb_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        y = self.upsample(x)
        y = torch.cat((y, h), dim=1)
        #y = self.attblock(y)
        y = self.resblock(y, time_embedding, context_embedding)
        return y

class TimeEmbedding(nn.Module):
    def __init__(self,
                 dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(200) / half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class EmbeddingFC(nn.Module):
    def __init__(self,
                emb_dim_in: int,
                emb_dim_out: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim_in, emb_dim_out),
            nn.SiLU()
        )
        self.emb_dim = emb_dim_in
        
    def forward(self, x: torch.Tensor):
        return self.fc(x.view(-1, self.emb_dim))

class UNet(nn.Module):
    def __init__(self,
                dim: int = 28,
                input_channels: int = 1,
                channels: int = 4,
                channel_mults = [1, 2, 4],
                time_emb_dim: int = 8,
                n_classes: int = 26,
                context_emb_dim = 8):
        super().__init__()
        self.increase_depth = nn.Conv2d(input_channels,
                              channels,
                              kernel_size=3,
                              padding=1)
        self.down_blocks = nn.ModuleList([UNetDown(channels*ch_in, channels*ch_out, time_emb_dim, context_emb_dim) for ch_in, ch_out in zip(channel_mults[:-1], channel_mults[1:])])
        self.middle_block = UNetMiddle(channels*channel_mults[-1], channels*channel_mults[-1], time_emb_dim, context_emb_dim)
        self.up_blocks = nn.ModuleList([UNetUp(channels*ch_in, channels*ch_out, time_emb_dim, context_emb_dim) for ch_in, ch_out in zip(channel_mults[:0:-1], channel_mults[-2::-1])])
        self.decrease_depth = nn.Conv2d(channels,
                              input_channels,
                              kernel_size=3,
                              padding=1)
            
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            EmbeddingFC(time_emb_dim, time_emb_dim)
        )
        self.context_mlp = nn.Sequential(
            nn.Embedding(n_classes, context_emb_dim),
            EmbeddingFC(context_emb_dim, context_emb_dim)
        )
        

    def forward(self, x: torch.Tensor, time: torch.Tensor, label = None):
        t = self.time_mlp(time)
        if label is not None:
            c = self.context_mlp(label)
        else:
            c = None
        #c = None
        h_deque = []
        y = x
        y = self.increase_depth(y)
        for block in self.down_blocks:
            y, h = block(y, t, c)
            h_deque.append(h)
        y = self.middle_block(y, t, c)
        for block in self.up_blocks:
            h = h_deque.pop()
            y = block(y, h, t, c)
        y = self.decrease_depth(y)
        return y














        