import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import click
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from matplotlib import colors

@click.command()
@click.option("--image_path",
              required=True,
              help="Specify the image")
@click.option("--mask_path",
              required=True,
              help="Specify the mask image")
@click.option("--model_path",
              required=True,
              help="Specify the model to be used for inference.")
@click.option(
    "--normalize",
    default=False,
    type=bool,
    help="Specify if you want to normalize the input image")
def inference(image_path,mask_path,model_path,normalize):
    # Load the trained model
    model = torch.load(model_path)
    model.cuda()
    # Set the model to evaluate mode
    model.eval()

    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    if normalize:
        data_transforms = transforms.Compose([
                                        transforms.Resize((256,256),interpolation = InterpolationMode.NEAREST),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([
                                        transforms.Resize((256,256),interpolation = InterpolationMode.NEAREST),
                                        transforms.ToTensor()])
    img_torch = data_transforms(img)
    img_torch = torch.unsqueeze(img_torch,0).type(torch.cuda.FloatTensor)
    with torch.no_grad():
        a = model(img_torch)
        _,indices = torch.max(a['out'],dim=1)

    plt.hist(a['out'].data.cpu().numpy().flatten())
    indices = indices.cpu().detach().numpy()[0]
    
    img = img.resize((256,256))
    mask = mask.resize((256,256))


    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(indices)
    plt.title('Segmentation Output')
    plt.axis('off')
    
    plt.savefig('results/WeightedCrossEntropy25Epochs/SegmentationOutput.png',bbox_inches='tight')

inference()