import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

def apply_custom_resnet_transforms(mean,std_dev):
    train_transforms = A.Compose([A.Normalize(mean=mean, std=std_dev, always_apply=True),
                                  A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # padding of 4 on each side of 32x32 image
                                  A.RandomCrop(height=32, width=32, always_apply=True),
                                  A.Cutout(num_holes=1,max_h_size=8, max_w_size=8, fill_value=mean, always_apply= True),
                                  ToTensorV2()
                                 ])

    test_transforms = A.Compose([A.Normalize(mean=mean, std=std_dev, always_apply=True),
                                 ToTensorV2(),
                                 ])

    return lambda img: train_transforms(image=np.array(img))["image"], lambda img: test_transforms(image=np.array(img))["image"]

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1
class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResBlock,self).__init__()
    self.res_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

  def forward (self, x):
    x = self.res_block(x)
    return x


class LayerBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(LayerBlock,self).__init__()
    self.layer_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

  def forward (self, x):
    x = self.layer_block(x)
    return x

class custom_resnet_s10(nn.Module):
  def __init__(self, num_classes=10):
    super(custom_resnet_s10,self).__init__()

    self.PrepLayer = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels=64, kernel_size = 3, stride = 1, padding =1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    self.Layer1 = LayerBlock(in_channels = 64, out_channels=128)
    self.resblock1 = ResBlock(in_channels =128, out_channels=128)
    self.Layer2 = LayerBlock(in_channels = 128, out_channels=256)
    self.resblock2 = ResBlock(in_channels =256, out_channels=256)
    self.Layer3 = LayerBlock(in_channels = 256, out_channels=512)
    self.resblock3 = ResBlock(in_channels =512, out_channels=512)
    self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4) # 512,512, 4/4 = 512,512,1
    self.fc = nn.Linear(512,num_classes)

  def forward(self,x):
    x = self.PrepLayer(x)
    #################
    x = self.Layer1(x)
    # print("x..l1",x.shape)
    resl1 = self.resblock1(x)
    # print("resl1",resl1.shape)
    x = x+resl1
    # print("x..l1+resl1",x.shape)
    #################
    x = self.Layer2(x)
    # print("x..l2",x.shape)
    resl2 = self.resblock2(x)
    # print("resl2",resl2.shape)
    x = x+resl2
    # print("x..l2+resl2",x.shape)
    #################
    x = self.Layer3(x)
    # print("x..l3",x.shape)
    resl3 = self.resblock3(x)
    # print("resl3",resl3.shape)
    x = x+resl3
    # print("x..l3+resl3",x.shape)
    #################
    x = self.max_pool4(x)
    # print("x..max_pool4",x.shape)
    x = x.view(x.size(0),-1)
    # print("x..flat",x.shape)
    x = self.fc(x)
    return F.log_softmax(x, dim=-1)
  
import torch
import torch.nn as nn
from tqdm import tqdm # for beautiful model training updates


def trainer(model,device, trainloader, testloader, optimizer,epochs,criterion,scheduler):
  train_losses = [] # to capture train losses over training epochs
  train_accuracy = [] # to capture train accuracy over training epochs
  test_losses = [] # to capture test losses
  test_accuracy = [] # to capture test accuracy
  for epoch in range(epochs):
    print("EPOCH:", epoch+1)
    train(model, device, trainloader, optimizer, epoch,criterion,train_accuracy,train_losses,scheduler) # Training Function
    test(model, device, testloader,criterion,test_accuracy,test_losses)   # Test Function

  return train_accuracy, train_losses, test_accuracy, test_losses


# # Training Function
def train(model, device, train_loader, optimizer, epoch,criterion,train_accuracy,train_losses,scheduler = None):
  model.train() # setting the model in training
  pbar = tqdm(train_loader) # putting the iterator in pbar
  correct = 0 # for accuracy numerator
  processed =0 # for accuracy denominator

  for batch_idx, (images,labels) in enumerate(pbar):
    images, labels = images.to(device),labels.to(device)#sending data to CPU or GPU as per device
    optimizer.zero_grad() # setting gradients to zero to avoid accumulation

    y_preds = model(images) # forward pass, result captured in y_preds (plural as there are many images in a batch)
    # the predictions are in one hot vector

    loss = criterion(y_preds,labels) # capturing loss

    train_losses.append(loss) # to capture loss over many epochs

    loss.backward() # backpropagation
    optimizer.step() # updating the params

    if scheduler:
      if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    preds = y_preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += preds.eq(labels.view_as(preds)).sum().item()
    processed += len(images)


    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_accuracy.append(100*correct/processed)


# # Test Function
def test(model, device, test_loader,criterion,test_accuracy,test_losses) :
  model.eval() # setting the model in evaluation mode
  test_loss = 0
  correct = 0 # for accuracy numerator

  with torch.no_grad():
    for (images,labels) in test_loader:
      images, labels = images.to(device),labels.to(device)#sending data to CPU or GPU as per device
      outputs = model(images) # forward pass, result captured in outputs (plural as there are many images in a batch)
      # the outputs are in batch size x one hot vector

      test_loss = criterion(outputs,labels).item()  # sum up batch loss
      preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset) # average test loss
    test_losses.append(test_loss) # to capture loss over many batches

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

    test_accuracy.append(100*correct/len(test_loader.dataset))

import seaborn as sns

def plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses):
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25,6)

    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(train_losses), 'b', label="Train Loss")

    # Label the plot.
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(train_accuracy), 'b', label="Train Accuracy")

    # Label the plot.
    ax2.set_title("Train Accuracy")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()

    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(test_losses), 'b', label="Test Loss")

    # Label the plot.
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(test_accuracy), 'b', label="Test Accuracy")

    # Label the plot.
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()

def evaluate_classwise_accuracy(model, device, classes, test_loader):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        

def plot_misclassified_images(wrong_predictions, mean, std, n_images=20, class_names=None):
    """
    Plot the misclassified images.
    """
    if class_names is None:
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig = plt.figure(figsize=(10, 12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:n_images]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j] * std[j]) + mean[j]

        img = np.transpose(img, (1, 2, 0))
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        ax.set_title(f"\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}", fontsize=10)
        ax.imshow(img)

    plt.show()

def misclassified_images(model, test_loader, device, mean, std, class_names=None, n_images=20):
    """
    Get misclassified images.
    """
    wrong_images = []
    wrong_label = []
    correct_label = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = pred.eq(target.view_as(pred)) == False
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f"Total wrong predictions are {len(wrong_predictions)}")

        plot_misclassified_images(wrong_predictions, mean, std, n_images=n_images, class_names=class_names)

    return wrong_predictions


