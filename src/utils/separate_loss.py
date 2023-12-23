import itertools
from typing import Any

import torch


class SeparateLoss:
    def __init__(self, mix_num, device):
        self.mix_num = mix_num
        self.permutations = [torch.tensor(perm, device=device) 
                             for perm in itertools.permutations(range(mix_num))]
    

    def cal_seploss(self, clean:torch.tensor, est:torch.tensor, loss_fn):
        """
        Calculate the separation loss between clean and estimated tensors.

        Args:
            clean (torch.tensor): The clean tensor with shape [B,C,T].
            est (torch.tensor): The estimated tensor with shape [B, C, T].
            loss_fn (function): The loss function to calculate the loss.

        Returns:
            torch.tensor: The mean separation loss.
        """
        losses = []
        for perm in self.permutations:
            loss = 0
            clean_perm = clean.index_select(1, perm)
            loss = loss_fn(clean_perm, est)
            losses.append(torch.mean(loss))
        losses = torch.stack(losses, dim=-1)
        loss, _ = torch.min(losses, dim=-1)
        return loss
    
    def __call__(self, **kwargs) -> Any:
        return self.cal_seploss(**kwargs)
    

if __name__ == "__main__":
    sep_loss = SeparateLoss(3,"cuda:0")
    clean = [torch.rand((4,3,12,34)).to("cuda:0"),
             torch.rand((4,3,431)).to("cuda:0")]
    est = [torch.rand((4,3,12,34)).to("cuda:0"),
             torch.rand((4,3,431)).to("cuda:0")]
    print(sep_loss.cal_seploss(clean, est, [lambda x, y: torch.mean(torch.abs(x-y), dim=(1,2,3)), lambda x, y: torch.mean(torch.abs(x-y), dim=(1,2))],[0.5,0.5]))