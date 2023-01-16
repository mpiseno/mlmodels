import torch
import torch.nn as nn

from mlmodels.utils import create_patches
from mlmodels.transformer import Transformer


class ViT(nn.Module):
    def __init__(self, input_shape, patch_size, dim=512):
        '''
        Implementation of Vision Transformer (Dosovitskiy et al., ICLR 2021)

        input_shape: shape of each image
        patch_size: Each patch from the image will be shape (patch_size, patch_size, C)
        dim: The dimension of the transformer.
        '''
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim
        self.H, self.W, self.C = input_shape
        self.N = (self.H * self.W) // (self.patch_size ** 2)    # Number of patches

        self.patch_embedding = nn.Linear((self.patch_size ** 2) * self.C, self.dim)
        scale = self.dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.N + 1, self.dim))

        self.transformer = Transformer(n_layers=2, n_heads=8, dim=self.dim)
        self.ln = nn.LayerNorm(self.dim)
    
    def forward(self, x):
        x = create_patches(x, self.patch_size)
        x = self.patch_embedding(x)

        class_emb = torch.zeros(x.shape[0], 1, x.shape[-1]) + self.class_embedding
        x = torch.cat((class_emb, x), dim=1) + self.positional_embedding
        x = self.transformer(x)
        x = self.ln(x[:, 0, :])
        return x


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    model = ViT(input_shape=(256, 256, 3), patch_size=16, dim=512)

    img = Image.open('test/assets/cat.png').convert('RGB')
    img = np.resize(np.asarray(img), (256, 256, 3))
    img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    model(img)