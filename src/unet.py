import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


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

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, device='cuda'):
        super().__init__()
        self.channels = channels
        half_channels = channels // 2
        self.encodings = torch.exp(torch.arange(0., half_channels, 2, device=device) * -(math.log(10000.0) / half_channels))
        
    def forward(self, x: torch.Tensor):
        device = x.device
        B, C, H, W = x.shape
        assert C == self.channels
        half_C = C // 2
        pos_w = torch.arange(0., W, device=device)
        pos_h = torch.arange(0., H, device=device)
        sin_inp_w = torch.einsum("i,j->ji", pos_w, self.encodings)
        sin_inp_h = torch.einsum("i,j->ji", pos_h, self.encodings)
        encodings_w = torch.cat((sin_inp_w.sin(), sin_inp_w.cos()), dim=0).unsqueeze(2)
        encodings_h = torch.cat((sin_inp_h.sin(), sin_inp_h.cos()), dim=0).unsqueeze(1)
        emb = torch.zeros((C, H, W), device=device)
        emb[:half_C, :, :] = encodings_h
        emb[half_C:C , :, :] = encodings_w
        return emb.repeat(B, 1, 1, 1)

class Attention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int):
        super().__init__()
        assert in_channels % num_heads == 0
        self.projection = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.num_heads = num_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, C, H, W = v.shape
        head_dim = C//self.num_heads
        q = q.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, head_dim).transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(B, C, H, W)
        h = self.projection(h)
        h = self.norm(h)
        return h

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int):
        super().__init__()
        assert in_channels % num_heads == 0
        self.fused_qkv = torch.nn.Conv2d(in_channels,        # x -> (q, k, v) as a single vector 
                                 in_channels*3,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.attention = Attention(in_channels, num_heads)
        self.encodings = PositionalEncoding2D(in_channels)
        self.encodings_mlp = nn.Sequential(
            nn.Conv2d(in_channels, 
                      in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor):
        enc = self.encodings(x)
        h = self.encodings_mlp(enc) + x
        B, C, H, W = h.shape
        qkv = self.fused_qkv(h)
        #qkv = qkv.view(B, 3*C, H, W)
        q, k, v = qkv.chunk(3, 1)
        
        h = self.attention(q, k, v)
        return x + h


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
                 context_emb_dim: int,
                 num_heads: Optional[int] = None):
        super().__init__()
        self.resblock = ResnetBlock(in_channels, out_channels, time_emb_dim, context_emb_dim)
        if num_heads is not None:
            self.attblock = SelfAttention(out_channels, num_heads)
        else:
            self.attblock = nn.Identity()
        self.downsample = Downsample(out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        h = self.resblock(x, time_embedding, context_embedding)
        h = self.attblock(h)
        y = self.downsample(h)
        return y, h

class UNetMiddle(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 context_emb_dim: int,
                 num_heads: Optional[int] = None):
        super().__init__()
        self.resblock1 = ResnetBlock(in_channels, out_channels, time_emb_dim, context_emb_dim)
        if num_heads is not None:
            self.attblock = SelfAttention(out_channels, num_heads)
        else:
            self.attblock = nn.Identity()
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
                 context_emb_dim: int,
                 num_heads: Optional[int] = None):
        super().__init__()
        self.upsample = Upsample(in_channels)
        if num_heads is not None:
            self.attblock = SelfAttention(in_channels+in_channels, num_heads)
        else:
            self.attblock = nn.Identity()
        self.resblock = ResnetBlock(in_channels+in_channels, out_channels, time_emb_dim, context_emb_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, time_embedding: torch.Tensor, context_embedding=None):
        y = self.upsample(x)
        y = torch.cat((y, h), dim=1)
        y = self.attblock(y)
        y = self.resblock(y, time_embedding, context_embedding)
        return y

class UNet(nn.Module):
    def __init__(self,
                dim: int = 28,
                channels: int = 1,
                channel_mults = [8, 16, 32],
                time_emb_dim: int = 8,
                n_classes: int = 26,
                context_emb_dim: int = 12,
                attention_heads = [None, None, None]):   #num of attention heads for down, middle and up passes
        super().__init__()
        assert len(attention_heads)==3
        self.increase_depth = nn.Conv2d(channels,
                              channels*channel_mults[0],
                              kernel_size=3,
                              padding=1)
        self.down_blocks = nn.ModuleList([UNetDown(channels*ch_in, channels*ch_out, time_emb_dim, context_emb_dim, attention_heads[0]) for ch_in, ch_out in zip(channel_mults[:-1], channel_mults[1:])])
        self.middle_block = UNetMiddle(channels*channel_mults[-1], channels*channel_mults[-1], time_emb_dim, context_emb_dim, attention_heads[1])
        self.up_blocks = nn.ModuleList([UNetUp(channels*ch_in, channels*ch_out, time_emb_dim, context_emb_dim, attention_heads[2]) for ch_in, ch_out in zip(channel_mults[:0:-1], channel_mults[-2::-1])])
        self.decrease_depth = nn.Conv2d(channels*channel_mults[0],
                              channels,
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














        