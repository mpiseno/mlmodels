import torch.nn.functional as F
	

def create_patches(imgs, p=16):
	'''
	Turns an batch of images of shape (*, C, H, W} to patches of shape (*, N, P^2 C)
	'''
	assert(len(imgs.shape) == 4)
	assert(imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0)

	patches = F.unfold(imgs, kernel_size=p, stride=p)
	return patches.permute(0, 2, 1)
