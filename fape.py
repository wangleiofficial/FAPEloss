"""
Frame aligned point error 
"""

from torch import nn
import torch
from einops import rearrange

class FAPEloss(nn.Module):
    """Frame aligned point error loss

    Args:
        Z (int, optional): [description]. Defaults to 10.
        clamp (int, optional): [description]. Defaults to 10.
        epsion (float, optional): [description]. Defaults to -1e4.
    """
    def __init__(self, Z=10.0, clamp=10.0, epsion=-1e4):

        super().__init__()
        self.z = Z
        self.epsion = epsion
        self.clamp = clamp

    def forward(self, predict_T, transformation, pdb_mask=None, padding_mask=None, device='cpu'):
        """
        Args:
            predict_T (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            transformation (`tensor`, `tensor`): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            pdb_mask (`tensor`, optional): pdb mask. size: [batch, N_seq, N_seq]. Defaults to None.
            padding_mask (`tensor`, optional): padding mask. size: [batch, N_seq, N_seq]. Defaults to None.
        """
        predict_R, predict_Trans = predict_T
        RotaionMatrix, translation = transformation
        delta_predict_Trans = rearrange(predict_Trans, 'b j t -> b j () t') - rearrange(predict_Trans, 'b i t -> b () i t')
        delta_Trans = rearrange(translation, 'b j t -> b j () t') - rearrange(translation, 'b i t -> b () i t')

        X_hat = torch.einsum('bikq, bjik->bijq', predict_R, delta_predict_Trans)
        X = torch.einsum('bikq, bjik->bijq', RotaionMatrix, delta_Trans)

        distance = torch.norm(X_hat-X, dim=-1)
        distance = torch.clamp(distance, max=self.clamp) * (1/self.z)

        if pdb_mask is not None:
            distance = distance * pdb_mask
        if padding_mask is not None:
            distance = distance * padding_mask

        FAPE_loss = torch.mean(distance)

        return FAPE_loss
