import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Load the trained model 
model = torch.load('./CFExp/weights.pt')
model.cuda()
# Set the model to evaluate mode
model.eval()
df = pd.read_csv('./CFExp/log.csv')
#df.plot(x='epoch',figsize=(15,8))
#print(df[['Train_auroc','Test_auroc']].max())


ino = 2
img = cv2.imread(f'./CrackForest/Images/001.jpg')
img = cv2.resize(img,(256,256)).transpose(2,0,1).reshape(1,3,256,256)
mask = cv2.imread(f'./CrackForest/Masks/001_label.PNG')
mask = cv2.resize(mask,(256,256))
with torch.no_grad():
    a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)

plt.hist(a['out'].data.cpu().numpy().flatten())

plt.figure(figsize=(10,10))
plt.subplot(131)
plt.imshow(img[0,...].transpose(1,2,0))
plt.title('Image')
plt.axis('off')
plt.subplot(132)
plt.imshow(mask)
plt.title('Ground Truth')
plt.axis('off')
plt.subplot(133)
plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.2)
plt.title('Segmentation Output')
plt.axis('off')
plt.savefig('./CFExp/SegmentationOutput.png',bbox_inches='tight')