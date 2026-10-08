from dataset import dataset
import os
from imutils import paths
import cv2
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNet

#get the images and masks paths
imagepth=os.path.join("/home/arezou/UBONTO/my_own_projects/pytorch/unet_lesion_detection/dataset","images")
imagepaths=sorted(list(paths.list_images(imagepth)))
maskpth=os.path.join("/home/arezou/UBONTO/my_own_projects/pytorch/unet_lesion_detection/dataset","masks")
maskpaths=sorted(list(paths.list_images(maskpth)))

#define transforms
Transforms=transforms.Compose([transforms.ToPILImage(),transforms.Resize((256,256)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(degrees=15),
                               transforms.RandomVerticalFlip(),transforms.ToTensor()])
#
#split dataset into train and test sets
(TrainImages,TestImages,TrainMasks,TestMasks)=train_test_split(imagepaths, maskpaths,test_size=0.15, random_state=42)

#save the test path to the disk to use them for prediction later
test_path=os.path.sep.join(["/home/arezou/UBONTO/my_own_projects/pytorch/unet_lesion_detection/output_aug","test_path.txt"])
f=open(test_path,"w")
f.write("\n".join(TestImages))
f.close()

#get the images and masks using defined dataset class
trainData=dataset(imagePaths=TrainImages,maskPaths=TrainMasks,transforms=Transforms)
testData=dataset(imagePaths=TestImages,maskPaths=TestMasks,transforms=Transforms)
#print the number of training and testing sets
print(f"[INFO] found {len(trainData)} examples in the training set...")
print(f"[INFO] found {len(testData)} examples in the test set...")

#define parameters needs for training
BS=32
Epochs=50
device="cuda" if torch.cuda.is_available else "cpu"
#create dataloader
trainDs=DataLoader(trainData,batch_size=BS,shuffle=True, pin_memory=True if device == "cuda" else False, num_workers=os.cpu_count())
testDs=DataLoader(testData,shuffle=False,batch_size=BS,pin_memory=True if device == "cuda" else False, num_workers=os.cpu_count())

#define train and test steps
trainSteps= len(trainData)//BS
testSteps=len(testData)//BS

#define U-net model ,optimizer and loss
model=UNet().to(device)
opt=Adam(model.parameters(),lr=0.001)
lossFn=BCEWithLogitsLoss()
#define a dictionary for storing train and test loss
H={"train_loss":[], "test_loss":[]}

print("start training...")
#start training
for e in tqdm(range (Epochs)):

    model.train()
    #initialized train and test loss
    train_loss=0
    test_loss=0
    for (image,mask) in trainDs:
        image= image.to(device)
        mask= mask.to(device)
        #get the prediction and calculate loss
        predict=model(image)
        loss=lossFn(predict,mask)

        #apply zero gradient, perform backpropagation and update step
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss +=loss

    with torch.no_grad():
        #set the model to evaluation mode
        model.eval()
        for (image,mask) in testDs:
            
            image= image.to(device)
            mask=mask.to(device)
            predict=model(image)
            loss=lossFn(predict,mask)
            test_loss+=loss
    
    avg_trainLoss=train_loss/trainSteps
    avg_testLoss=test_loss/testSteps
    H["train_loss"].append(avg_trainLoss.cpu().detach().numpy())
    H["test_loss"].append(avg_testLoss.cpu().detach().numpy())

    print( " for {}/{} epoch : train_loss:{}, test_loss:{}".format(e+1,Epochs,avg_trainLoss,avg_testLoss))
 

#define plot path and model path:
PLOT_PATH = os.path.sep.join(["/home/arezou/UBONTO/my_own_projects/pytorch/unet_lesion_detection/output_aug", "plot.png"])
MODEL_PATH = os.path.sep.join(["/home/arezou/UBONTO/my_own_projects/pytorch/unet_lesion_detection/output_aug","unet_lesion.pth"])
 # plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)
# serialize the model to disk
torch.save(model,MODEL_PATH)


    




