import os
from numpy import imag
import pandas as pd
import click
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import cuda
from torch._C import device
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2

@click.command()
@click.option("--test_data_path",
			  required=True,
			  help="Experiments for which the visualization are to be generated")

@click.option("--model_path",
			  required=True,
			  help="Specify the model to be used for inference.")
@click.option(
	"--normalize",
	default=False,
	type=bool,
	help="Specify if you want to normalize the input image")

def calc_metrics(test_data_path,model_path,normalize):
	# Set the model to evaluate mode
	model = torch.load(model_path)
	model.cuda()
	model.eval()
	count = 0
	correct_pixel_count = 0
	total_pixel_count = 0
	for image in os.listdir(test_data_path+'/Masks'):
		image_path = test_data_path+'/Images/'+image
		mask_path = test_data_path+'/Masks/'+image
		img = cv2.imread(image_path)
		mask = cv2.imread(mask_path)
		if normalize:
			data_transforms = transforms.Compose([
											transforms.ToTensor(),
											transforms.Resize((256,256),interpolation = InterpolationMode.NEAREST),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		else:
			data_transforms = transforms.Compose([
											transforms.ToTensor(),
											transforms.Resize((256,256),interpolation = InterpolationMode.NEAREST)])
		mask_transforms = transforms.Compose([
											transforms.ToTensor(),
											transforms.Resize((256,256),interpolation = InterpolationMode.NEAREST)])
		img_torch = data_transforms(img)
		img_torch = torch.unsqueeze(img_torch,0).type(torch.cuda.FloatTensor)
		with torch.no_grad():
			a = model(img_torch)
			_,indices = torch.max(a['out'],dim=1)
		mask_output = 2*torch.ones((256,256))
		mask_output = mask_output.to(device='cuda')
		mask = mask_transforms(mask)
		mask = mask.to(device='cuda')
		mask = mask*255
		mask = mask.long()
		mask_output[(mask[0]==0) & (mask[1]==0) & (mask[2]==0)]=0
		mask_output[(mask[0]==0) & (mask[1]==0) & (mask[2]==128)]=1
		correct_pixel_count += torch.sum((mask_output == indices[0])*(mask_output == 2))
		total_pixel_count += torch.sum(mask_output==2)
		count+=1
		print(count)
	
	print(total_pixel_count)
	print('Global Pixel Wise Accuracy = ',correct_pixel_count/total_pixel_count)

calc_metrics()