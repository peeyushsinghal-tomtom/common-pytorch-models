{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNAKp/6AoE2K6iVTyzuyCaX",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/peeyushsinghal/common-pytorch-models/blob/main/custom_resnet.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "iaxGqsBdISyt"
      },
      "outputs": [],
      "source": [
        "import albumentations as A\n",
        "import numpy as np\n",
        "from albumentations.pytorch.transforms import ToTensorV2\n",
        "\n",
        "def apply_custom_resnet_transforms(mean,std_dev):\n",
        "    train_transforms = A.Compose([A.Normalize(mean=mean, std=std_dev, always_apply=True),\n",
        "                                  A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),  # padding of 4 on each side of 32x32 image\n",
        "                                  A.RandomCrop(height=32, width=32, always_apply=True),\n",
        "                                  A.Cutout(num_holes=1,max_h_size=8, max_w_size=8, fill_value=mean, always_apply= True),\n",
        "                                  ToTensorV2()\n",
        "                                 ])\n",
        "\n",
        "    test_transforms = A.Compose([A.Normalize(mean=mean, std=std_dev, always_apply=True),\n",
        "                                 ToTensorV2(),\n",
        "                                 ])\n",
        "\n",
        "    return lambda img: train_transforms(image=np.array(img))[\"image\"], lambda img: test_transforms(image=np.array(img))[\"image\"]\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "def imshow(img):\n",
        "    img = img / 2 + 0.5     # unnormalize\n",
        "    npimg = img.numpy()\n",
        "    plt.imshow(np.transpose(npimg, (1, 2, 0)))\n",
        "    plt.show()\n",
        "\n",
        "\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "dropout_value = 0.1\n",
        "class ResBlock(nn.Module):\n",
        "  def __init__(self, in_channels, out_channels):\n",
        "    super(ResBlock,self).__init__()\n",
        "    self.res_block = nn.Sequential(\n",
        "        nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),\n",
        "        nn.BatchNorm2d(out_channels),\n",
        "        nn.ReLU(),\n",
        "        nn.Conv2d(in_channels=out_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),\n",
        "        nn.BatchNorm2d(out_channels),\n",
        "        nn.ReLU(),\n",
        "    )\n",
        "\n",
        "  def forward (self, x):\n",
        "    x = self.res_block(x)\n",
        "    return x\n",
        "\n",
        "\n",
        "class LayerBlock(nn.Module):\n",
        "  def __init__(self, in_channels, out_channels):\n",
        "    super(LayerBlock,self).__init__()\n",
        "    self.layer_block = nn.Sequential(\n",
        "        nn.Conv2d(in_channels=in_channels, out_channels = out_channels, kernel_size=3, stride =1 , padding =1),\n",
        "        nn.MaxPool2d(kernel_size=2,stride=2),\n",
        "        nn.BatchNorm2d(out_channels),\n",
        "        nn.ReLU(),\n",
        "    )\n",
        "\n",
        "  def forward (self, x):\n",
        "    x = self.layer_block(x)\n",
        "    return x\n",
        "\n",
        "class custom_resnet_s10(nn.Module):\n",
        "  def __init__(self, num_classes=10):\n",
        "    super(custom_resnet_s10,self).__init__()\n",
        "\n",
        "    self.PrepLayer = nn.Sequential(\n",
        "        nn.Conv2d(in_channels = 3, out_channels=64, kernel_size = 3, stride = 1, padding =1),\n",
        "        nn.BatchNorm2d(64),\n",
        "        nn.ReLU(),\n",
        "    )\n",
        "    self.Layer1 = LayerBlock(in_channels = 64, out_channels=128)\n",
        "    self.resblock1 = ResBlock(in_channels =128, out_channels=128)\n",
        "    self.Layer2 = LayerBlock(in_channels = 128, out_channels=256)\n",
        "    self.resblock2 = ResBlock(in_channels =256, out_channels=256)\n",
        "    self.Layer3 = LayerBlock(in_channels = 256, out_channels=512)\n",
        "    self.resblock3 = ResBlock(in_channels =512, out_channels=512)\n",
        "    self.max_pool4 = nn.MaxPool2d(kernel_size=4, stride=4) # 512,512, 4/4 = 512,512,1\n",
        "    self.fc = nn.Linear(512,num_classes)\n",
        "\n",
        "  def forward(self,x):\n",
        "    x = self.PrepLayer(x)\n",
        "    #################\n",
        "    x = self.Layer1(x)\n",
        "    # print(\"x..l1\",x.shape)\n",
        "    resl1 = self.resblock1(x)\n",
        "    # print(\"resl1\",resl1.shape)\n",
        "    x = x+resl1\n",
        "    # print(\"x..l1+resl1\",x.shape)\n",
        "    #################\n",
        "    x = self.Layer2(x)\n",
        "    # print(\"x..l2\",x.shape)\n",
        "    resl2 = self.resblock2(x)\n",
        "    # print(\"resl2\",resl2.shape)\n",
        "    x = x+resl2\n",
        "    # print(\"x..l2+resl2\",x.shape)\n",
        "    #################\n",
        "    x = self.Layer3(x)\n",
        "    # print(\"x..l3\",x.shape)\n",
        "    resl3 = self.resblock3(x)\n",
        "    # print(\"resl3\",resl3.shape)\n",
        "    x = x+resl3\n",
        "    # print(\"x..l3+resl3\",x.shape)\n",
        "    #################\n",
        "    x = self.max_pool4(x)\n",
        "    # print(\"x..max_pool4\",x.shape)\n",
        "    x = x.view(x.size(0),-1)\n",
        "    # print(\"x..flat\",x.shape)\n",
        "    x = self.fc(x)\n",
        "    return F.log_softmax(x, dim=-1)\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from tqdm import tqdm # for beautiful model training updates\n",
        "\n",
        "\n",
        "def trainer(model,device, trainloader, testloader, optimizer,epochs,criterion,scheduler):\n",
        "  train_losses = [] # to capture train losses over training epochs\n",
        "  train_accuracy = [] # to capture train accuracy over training epochs\n",
        "  test_losses = [] # to capture test losses\n",
        "  test_accuracy = [] # to capture test accuracy\n",
        "  for epoch in range(epochs):\n",
        "    print(\"EPOCH:\", epoch+1)\n",
        "    train(model, device, trainloader, optimizer, epoch,criterion,train_accuracy,train_losses,scheduler) # Training Function\n",
        "    test(model, device, testloader,criterion,test_accuracy,test_losses)   # Test Function\n",
        "\n",
        "  return train_accuracy, train_losses, test_accuracy, test_losses\n",
        "\n",
        "\n",
        "# # Training Function\n",
        "def train(model, device, train_loader, optimizer, epoch,criterion,train_accuracy,train_losses,scheduler = None):\n",
        "  model.train() # setting the model in training\n",
        "  pbar = tqdm(train_loader) # putting the iterator in pbar\n",
        "  correct = 0 # for accuracy numerator\n",
        "  processed =0 # for accuracy denominator\n",
        "\n",
        "  for batch_idx, (images,labels) in enumerate(pbar):\n",
        "    images, labels = images.to(device),labels.to(device)#sending data to CPU or GPU as per device\n",
        "    optimizer.zero_grad() # setting gradients to zero to avoid accumulation\n",
        "\n",
        "    y_preds = model(images) # forward pass, result captured in y_preds (plural as there are many images in a batch)\n",
        "    # the predictions are in one hot vector\n",
        "\n",
        "    loss = criterion(y_preds,labels) # capturing loss\n",
        "\n",
        "    train_losses.append(loss) # to capture loss over many epochs\n",
        "\n",
        "    loss.backward() # backpropagation\n",
        "    optimizer.step() # updating the params\n",
        "\n",
        "    if scheduler:\n",
        "      if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):\n",
        "        scheduler.step()\n",
        "\n",
        "    preds = y_preds.argmax(dim=1, keepdim=True)  # get the index of the max log-probability\n",
        "    correct += preds.eq(labels.view_as(preds)).sum().item()\n",
        "    processed += len(images)\n",
        "\n",
        "\n",
        "    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')\n",
        "\n",
        "    train_accuracy.append(100*correct/processed)\n",
        "\n",
        "\n",
        "# # Test Function\n",
        "def test(model, device, test_loader,criterion,test_accuracy,test_losses) :\n",
        "  model.eval() # setting the model in evaluation mode\n",
        "  test_loss = 0\n",
        "  correct = 0 # for accuracy numerator\n",
        "\n",
        "  with torch.no_grad():\n",
        "    for (images,labels) in test_loader:\n",
        "      images, labels = images.to(device),labels.to(device)#sending data to CPU or GPU as per device\n",
        "      outputs = model(images) # forward pass, result captured in outputs (plural as there are many images in a batch)\n",
        "      # the outputs are in batch size x one hot vector\n",
        "\n",
        "      test_loss = criterion(outputs,labels).item()  # sum up batch loss\n",
        "      preds = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability\n",
        "      correct += preds.eq(labels.view_as(preds)).sum().item()\n",
        "\n",
        "    test_loss /= len(test_loader.dataset) # average test loss\n",
        "    test_losses.append(test_loss) # to capture loss over many batches\n",
        "\n",
        "    print('\\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\\n'.format(\n",
        "    test_loss, correct, len(test_loader.dataset),\n",
        "    100. * correct / len(test_loader.dataset)))\n",
        "\n",
        "    test_accuracy.append(100*correct/len(test_loader.dataset))\n",
        "\n",
        "import seaborn as sns\n",
        "\n",
        "def plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses):\n",
        "    sns.set(font_scale=1)\n",
        "    plt.rcParams[\"figure.figsize\"] = (25,6)\n",
        "\n",
        "    # Plot the learning curve.\n",
        "    fig, (ax1,ax2) = plt.subplots(1,2)\n",
        "    ax1.plot(np.array(train_losses), 'b', label=\"Train Loss\")\n",
        "\n",
        "    # Label the plot.\n",
        "    ax1.set_title(\"Train Loss\")\n",
        "    ax1.set_xlabel(\"Iterations\")\n",
        "    ax1.set_ylabel(\"Loss\")\n",
        "    ax1.legend()\n",
        "\n",
        "    ax2.plot(np.array(train_accuracy), 'b', label=\"Train Accuracy\")\n",
        "\n",
        "    # Label the plot.\n",
        "    ax2.set_title(\"Train Accuracy\")\n",
        "    ax2.set_xlabel(\"Iterations\")\n",
        "    ax2.set_ylabel(\"Loss\")\n",
        "    ax2.legend()\n",
        "\n",
        "    plt.show()\n",
        "\n",
        "    # Plot the learning curve.\n",
        "    fig, (ax1,ax2) = plt.subplots(1,2)\n",
        "    ax1.plot(np.array(test_losses), 'b', label=\"Test Loss\")\n",
        "\n",
        "    # Label the plot.\n",
        "    ax1.set_title(\"Test Loss\")\n",
        "    ax1.set_xlabel(\"Epoch\")\n",
        "    ax1.set_ylabel(\"Loss\")\n",
        "    ax1.legend()\n",
        "\n",
        "    ax2.plot(np.array(test_accuracy), 'b', label=\"Test Accuracy\")\n",
        "\n",
        "    # Label the plot.\n",
        "    ax2.set_title(\"Test Accuracy\")\n",
        "    ax2.set_xlabel(\"Epoch\")\n",
        "    ax2.set_ylabel(\"Loss\")\n",
        "    ax2.legend()\n",
        "\n",
        "    plt.show()\n",
        "\n",
        "def evaluate_classwise_accuracy(model, device, classes, test_loader):\n",
        "    class_correct = list(0. for i in range(len(classes)))\n",
        "    class_total = list(0. for i in range(len(classes)))\n",
        "    with torch.no_grad():\n",
        "        for images, labels in test_loader:\n",
        "            images, labels = images.to(device), labels.to(device)\n",
        "            outputs = model(images)\n",
        "            _, predicted = torch.max(outputs, 1)\n",
        "            c = (predicted == labels).squeeze()\n",
        "            for i in range(4):\n",
        "                label = labels[i]\n",
        "                class_correct[label] += c[i].item()\n",
        "                class_total[label] += 1\n",
        "\n",
        "    for i in range(len(classes)):\n",
        "        print('Accuracy of %5s : %2d %%' % (\n",
        "            classes[i], 100 * class_correct[i] / class_total[i]))\n",
        "\n",
        "\n",
        "def plot_misclassified_images(wrong_predictions, mean, std, n_images=20, class_names=None):\n",
        "    \"\"\"\n",
        "    Plot the misclassified images.\n",
        "    \"\"\"\n",
        "    if class_names is None:\n",
        "        class_names = [\"airplane\", \"automobile\", \"bird\", \"cat\", \"deer\", \"dog\", \"frog\", \"horse\", \"ship\", \"truck\"]\n",
        "    fig = plt.figure(figsize=(10, 12))\n",
        "    fig.tight_layout()\n",
        "    for i, (img, pred, correct) in enumerate(wrong_predictions[:n_images]):\n",
        "        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()\n",
        "        for j in range(img.shape[0]):\n",
        "            img[j] = (img[j] * std[j]) + mean[j]\n",
        "\n",
        "        img = np.transpose(img, (1, 2, 0))\n",
        "        ax = fig.add_subplot(5, 5, i + 1)\n",
        "        ax.axis(\"off\")\n",
        "        ax.set_title(f\"\\nactual : {class_names[target.item()]}\\npredicted : {class_names[pred.item()]}\", fontsize=10)\n",
        "        ax.imshow(img)\n",
        "\n",
        "    plt.show()\n",
        "\n",
        "def misclassified_images(model, test_loader, device, mean, std, class_names=None, n_images=20):\n",
        "    \"\"\"\n",
        "    Get misclassified images.\n",
        "    \"\"\"\n",
        "    wrong_images = []\n",
        "    wrong_label = []\n",
        "    correct_label = []\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        for data, target in test_loader:\n",
        "            data, target = data.to(device), target.to(device)\n",
        "            output = model(data)\n",
        "            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability\n",
        "\n",
        "            wrong_pred = pred.eq(target.view_as(pred)) == False\n",
        "            wrong_images.append(data[wrong_pred])\n",
        "            wrong_label.append(pred[wrong_pred])\n",
        "            correct_label.append(target.view_as(pred)[wrong_pred])\n",
        "\n",
        "            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))\n",
        "        print(f\"Total wrong predictions are {len(wrong_predictions)}\")\n",
        "\n",
        "        plot_misclassified_images(wrong_predictions, mean, std, n_images=n_images, class_names=class_names)\n",
        "\n",
        "    return wrong_predictions\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "mhje9_fXIeok"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}