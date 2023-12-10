import torch
import torch.nn as nn

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
        self.depth_wise_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        self.drop_out = nn.Dropout(drop_out_rate)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        conv_in = x.transpose(-1, -2)
        conv_in = self.depth_wise_conv(conv_in)
        x = x + conv_in.transpose(-1, -2)
        x = self.drop_out(x)
        return x

class MossFormer(nn.Module):
    def __init__(self,
                 in_dim:int=512,
                 out_dim:int=512,
                 drop_out_rate:float=0.1,
                 ) -> None:
        super().__init__()
        self.conv_module = ConvolutionModule(in_dim, out_dim, drop_out_rate)
        self.norm = nn.LayerNorm(out_dim)
        self.drop_out = nn.Dropout(drop_out_rate)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_module(x)
        x = self.norm(x)
        x = self.drop_out(x)
        return x
if __name__ == "__main__":
    x = torch.randn(2, 1023, 512)
    conv = ConvolutionModule()
    y = conv(x)
    print(y.shape)