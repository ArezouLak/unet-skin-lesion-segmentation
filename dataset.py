# import the necessary packages
from torch.utils.data import Dataset
import cv2

class dataset(Dataset):
	def __init__(self,imagePaths,maskPaths,transforms):
		super().__init__()

		self.imagePaths=imagePaths
		self.maskPaths=maskPaths
		self.transforms=transforms

	def __len__(self):
		return len(self.imagePaths)
	
	def __getitem__(self, index):
		imagepath=self.imagePaths[index]
		maskpath=self.maskPaths[index]

		image=cv2.imread(imagepath)
		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		mask=cv2.imread(maskpath,0)

		if self.transforms is not None:

			image=self.transforms(image)
			mask=self.transforms(mask)
		return (image,mask)
		


                