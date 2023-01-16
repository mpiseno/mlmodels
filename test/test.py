from PIL import Image

import torch
import numpy as np

from mlmodels.utils import create_patches


def test_create_patches():
    img = Image.open('test/assets/cat.png').convert('RGB')

    p = 16
    img = np.resize(np.asarray(img), (p**2, p**2, 3))
    img = torch.as_tensor(img)
    patches = create_patches(img.clone(), p=p)
    patches_rec = [patch.view(p, p, 3) for patch in patches]

    rows = []
    for i in range(p):
        row = patches_rec[i*p:(i+1)*p]
        row = torch.concat(row, dim=1)
        rows.append(row)
    
    img_rec = torch.concat(rows, dim=0)
    correct = torch.all(img == img_rec)
    return correct


def main():
    print(test_create_patches())


if __name__ == '__main__':
    main()