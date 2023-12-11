import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum(t, self.inv_freq, 'i, j -> i j')
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale


class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum(x, self.gamma, '... d, h d -> ... h d') + self.beta
        return out.unbind(dim = -2)


class ConvolutionModule(nn.Module):
    def __init__(self,
                 in_dim:int=512,
                 out_dim:int=512,
                 drop_out_rate:float=0.1,
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU()
        )
        self.depth_wise_conv = nn.Conv1d(out_dim,
                                         out_dim,
                                         kernel_size=3,
                                         padding=1,
                                         groups=out_dim)
        self.drop_out = nn.Dropout(drop_out_rate)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        conv_in = x.transpose(-1, -2)
        conv_in = self.depth_wise_conv(conv_in)
        x = x + conv_in.transpose(-1, -2)
        x = self.drop_out(x)
        return x


class GLU(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.parallel_liner1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.parallel_liner2 = nn.Conv1d(dim, dim, kernel_size=1)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x1 = self.parallel_liner1(x)
        x2 = self.parallel_liner2(x)
        x = torch.sigmoid(x1) * x2
        return x


class MossFormerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        hidden_dim = 128,
        expansion_factor = 2.,
        dropout = 0.1,
    ):
        """
        MossFormerBlock is a module that represents a single block of the MossFormer model.
        
        Args:
            dim (int): The input dimension of the block.
            group_size (int, optional): The size of the groups used for quadratic attention. Defaults to 256.
            query_key_dim (int, optional): The dimension of the query and key vectors. Defaults to 128.
            hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 128.
            expansion_factor (float, optional): The expansion factor for the hidden dimension. Defaults to 2.0.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.group_size = group_size
        hidden_dim = int(dim * expansion_factor)

        self.attn_fn = ReLUSquared()
        self.rotary_pos_emb = RotaryEmbedding(dim = query_key_dim)
        self.dropout = nn.Dropout(dropout)

        self.to_u = ConvolutionModule(
            in_dim=dim,
            out_dim=hidden_dim,
            drop_out_rate=dropout,
        )

        self.to_v = ConvolutionModule(
            in_dim=dim,
            out_dim=hidden_dim,
            drop_out_rate=dropout,
        )

        self.to_qk = ConvolutionModule(
            in_dim=dim,
            out_dim=query_key_dim,
            drop_out_rate=dropout,
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)
        self.to_out = ConvolutionModule(
            in_dim=hidden_dim,
            out_dim=dim,
            drop_out_rate=dropout)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass of the MossFormerBlock.
        
        Args:
            x (torch.Tensor): The input tensor with shape [B, T, C].
        
        Returns:
            torch.Tensor: The output tensorwith shape[B, T, C].
        """
        seq_len, group_size = x.shape[-2], self.group_size

        # initial projection
        u = self.to_u(x)
        v = self.to_v(x)
        Z = self.to_qk(x)

        # offset and scale

        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(Z)

        quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys,
                                           (quad_q, lin_q, quad_k, lin_k))
        
        # linear gloabl attention
        lin_kv = einsum(lin_k, v, 'b n d, b n e -> b d e') / seq_len
        lin_ku = einsum(lin_k, u, 'b n d, b n e -> b d e') / seq_len
        lin_attn_out_v = einsum(lin_q, lin_kv, 'b n d, b d e -> b n e')
        lin_attn_out_u = einsum(lin_q, lin_ku, 'b n d, b d e -> b n e')

        padding = padding_to_multiple_of(seq_len, group_size)

        if padding > 0:
            quad_q, quad_k, u, v \
            = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.),
                  (quad_q, quad_k, u, v))

        # group along sequence
        quad_q, quad_k, u, v \
        = map(lambda t: rearrange(t, 'b (n g) d -> b n g d', g = self.group_size),
              (quad_q, quad_k, u, v))

        # calculate quadratic attention output

        sim = einsum(quad_q, quad_k, '... i d, ... j d -> ... i j') / group_size
        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        quad_out_v = einsum(attn, v, '... i j, ... j d -> ... i d')
        quad_out_u = einsum(attn, u, '... i j, ... j d -> ... i d')

        # fold back groups into full sequence, and excise out padding

        quad_attn_out_v, quad_attn_out_u, u, v\
        = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :seq_len],
              (quad_out_v, quad_out_u, u, v))
        
        v_hat = quad_attn_out_v + lin_attn_out_v
        u_hat = quad_attn_out_u + lin_attn_out_u

        O_hat = F.sigmoid(u*v_hat)
        O_hat_hat = u_hat * v

        return self.to_out(O_hat_hat * O_hat) + x

class MossFormer(nn.Module):
    def __init__(self,
                 in_dim:int=1,
                 hidden_dim:int=512,
                 kernel_size:int=8,
                 stride:int=4,
                 speaker_num:int=4,
                 MFB_num:int=4,
                 drop_out_rate:float=0.1,
                 ) -> None:
        """
        MossFormer model implementation.

        Args:
            in_dim (int): Number of input dimensions. Default is 1.
            hidden_dim (int): Dimension of hidden layers. Default is 512.
            kernel_size (int): Size of the convolutional kernel. Default is 8.
            stride (int): Stride value for the convolutional layers. Default is 4.
            speaker_num (int): Number of speakers. Default is 4.
            MFB_num (int): Number of MossFormer blocks. Default is 4.
            drop_out_rate (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.speaker_num = speaker_num
        self.MFB_num = MFB_num
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_dim,
                      hidden_dim,
                      kernel_size=kernel_size,
                      stride=stride),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.abs_pos_emb = ScaledSinuEmbedding(hidden_dim)

        self.in_point_wise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        for i in range(MFB_num):
            setattr(self, f"MFB_{i}", MossFormerBlock(dim=hidden_dim,
                                                      group_size=256,
                                                      query_key_dim=128,
                                                      expansion_factor=2.,
                                                      dropout=drop_out_rate))
        
        self.split_speaker_conv = nn.Conv1d(hidden_dim,
                                            self.speaker_num*hidden_dim,
                                            kernel_size=1)
        self.glu = GLU(hidden_dim)
        self.out_point_wise_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.out_conv = nn.ConvTranspose1d(hidden_dim,
                                           in_dim,
                                           kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MossFormer model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 1, T].

        Returns:
            torch.Tensor: Output tensor with shape [BxC, 1, T].
        """
        in_len = x.shape[-1]
        x_in = self.in_conv(x)
        
        x_trans = x_in.transpose(-1, -2)
        x_norm = self.ln1(x_trans)
        abs_pos_emb = self.abs_pos_emb(x_norm)
        x_pos = abs_pos_emb + x_norm
        x_pos = x_pos.transpose(-1, -2)
        
        x_MFB_in = self.in_point_wise_conv(x_pos)
        x_MFB_in = x_MFB_in.transpose(-1, -2)
        for i in range(self.MFB_num):
            x_MFB_in = getattr(self, f"MFB_{i}")(x_MFB_in)
        x_MFB_out = x_MFB_in.transpose(-1, -2)
        x_MFB_out = F.relu(x_MFB_out)
        
        x_split = self.split_speaker_conv(x_MFB_out)
        x_split = rearrange(x_split, 'b (c n) s -> (b c) n s', c=self.speaker_num)
        x_split = self.glu(x_split)
        mask = self.out_point_wise_conv(x_split)
        x_in = x_in.repeat_interleave(self.speaker_num, dim = 0)
        split_sound =  self.out_conv(mask * x_in)[...,:in_len]
        return split_sound


if __name__ == "__main__":
    x = torch.randn(2, 1, 12090)
    x = x.to("cuda:0")
    mf = MossFormer()
    mf.to("cuda:0")
    x = mf(x)
