"""
Frame aligned point error 
"""

from torch import nn
import torch
from einops import rearrange

class FAPEloss(nn.Module):
    """Frame aligned point error loss
    """
    def __init__(self, Z=10, clamp=10, epsion=-1e4):
        super().__init__()
        self.z = Z
        self.epsion = epsion
        self.clamp = clamp

    def forward(self, predict_T, transformation):
        """
        Args:
            predict_T ([tuple]): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
            transformation ([type]): ([batch, N_seq, 3, 3], [batch, N_seq, 3])
        """
        predict_R, predict_Trans = predict_T
        RotaionMatrix, translation = transformation
        delta_predict_Trans = rearrange(predict_Trans, 'b n t -> b n t ()') - rearrange(predict_Trans, 'b n t -> b n () t')
        delta_Trans = rearrange(translation, 'b n t -> b n t ()') - rearrange(translation, 'b n t -> b n () t')

        X_hat = torch.einsum('bnki, bnjk->bnji', predict_R, delta_predict_Trans)
        X = torch.einsum('bnki, bnjk->bnji', RotaionMatrix, delta_Trans)

        distance = torch.norm(X_hat-X, dim=-1)
        distance = torch.where(distance>self.clamp, distance, self.clamp) * (1/self.clamp)

        FAPE_loss = torch.mean(distance)

        return FAPE_loss
