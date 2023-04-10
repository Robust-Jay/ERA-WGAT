"""
Mayo 数据集
Poisson noise was inserted into the projection data for each case in the library
to reach a noise level that corresponded to 25% of the full dose (i.e. "quarter-dose" data were simulated).

row data: quarter-dose data
ground truth: full-dose data
"""

from torch.utils.data import Dataset
import torch
import numpy as np


class CTDataset(Dataset):
	def __init__(self, noise_level="quarter_dose", flag="train"):
		super(CTDataset, self).__init__()
		self.root = "../data/Mayo"
		self.files = np.load(f"./dataset/split_datasets/{flag}.npy", allow_pickle=False)
		self.noise_level = noise_level
	
	def __getitem__(self, index: int):
		noise = np.load(f"{self.root}/{self.noise_level}/{self.files[index]}")
		gt = np.load(f"{self.root}/full_dose/{self.files[index]}")
		
		return torch.from_numpy(noise).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)
	
	def __len__(self):
		return len(self.files)

