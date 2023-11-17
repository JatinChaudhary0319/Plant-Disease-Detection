#Importing Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

'''Importing Dataset
   Using Plant Village Dataset from following link:
   https://data.mendeley.com/datasets/tywbtsjrjv/1'''

transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

dataset = datasets.ImageFolder("Dataset", transform = transform)

dataset

indices = list(range(len(dataset)))

split = int(np.floor(0.85*len(dataset))) # Train-Size

validation = int(np.floor(0.8 * split)) #Validation

print(0, validation, split, len(dataset))

print(f'length of train size: {validation}')
print(f'length of validation data: {split - validation}')
print(f'length of test size: {len(dataset) - validation}')

np.random.shuffle(indices)

# Spliting into Training and Testing data

train_indices, validation_indices, test_indices = (indices[:validation], indices[validation:split], indices[split:])

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(dataset.class_to_idx)

'''Modeling
   Convolution Aithmetic Equation : </b>(W - F + 2P) / S + 1
   where, W = Input Size
          F = Filter Size
          P = Padding Size
          S = Stride'''

class CNN(nn.Module):
  def __init__(self, K):
    super(CNN, self).__init__()
    self.conv_layers = nn.Sequential(
        # Conv1
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),

        # Conv2
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),

        # Conv3
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),

        # Conv4
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2),
    )

    self.dense_layers = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(50176, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, K),
    )

  def forward(self, X):
    out = self.conv_layers(X)

    #Flatten
    out = out.view(-1, 50176)

    #Fully Connected
    out = self.dense_layers(out)

    return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

device = 'cpu'

model = CNN(targets_size)

model.to(device)

from torchsummary import summary
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss() #softmax + cross entropy loss
optimizer = torch.optim.Adam(model.parameters())

# Batch Gradient Descent

def batch_gd(model, criterion, train_loader, validation_loader, epochs):
  train_losses = np.zeros(epochs)
  validation_losses = np.zeros(epochs)

  for e in range(epochs):
    t0 = datetime.now()
    train_loss = []

    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)

      optimizer.zero_grad()

      output = model(inputs)

      loss = criterion(output, targets)

      train_loss.append(loss.item())

      loss.backward()
      optimizer.step()

    train_loss = np.mean(train_loss)

    validation_loss = []

    for inputs, targets in validation_loader:
      inputs, targets = inputs.to(device), targets.to(device)

      output = model(inputs)

      loss = criterion(output, targets)

      validation_loss.append(loss.item())

    validation_loss = np.mean(validation_loss)

    train_losses[e] = train_loss
    validation_losses[e] = validation_loss

    dt = datetime.now() - t0

    print(f"Epoch : {e+1}/{epochs} Train_loss : {train_loss:.3f} Test_loss : {validation_loss:.3f} Duration : {dt}")

    return train_losses, validation_losses

device = "cpu"

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size = batch_size, sampler = train_sampler
)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size = batch_size, sampler = test_sampler
)
validation_loader = torch.utils.data.DataLoader(
    dataset, batch_size = batch_size, sampler = validation_sampler
)

train_losses, validation_losses = batch_gd(model, criterion, train_loader, validation_loader, 5)

# Saving the Model

torch.save(model.state_dict(), 'plant_disease_model_1_latest.pt')


'''Loading Model

targets_size = 39
model = CNN(targets_size)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()
'''
# Ploting the loss

plt.plot(train_losses, label = 'train_loss')
plt.plot(validation_losses, label = 'validation_loss')
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy

def accuracy(loader):
  n_correct = 0
  n_total = 0

  for inputs, targets in loader:
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)

    _, predictions = torch.max(outputs, 1)

    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

  acc = n_correct/n_total

  return acc

train_acc = accuracy(train_loader)
test_acc = accuracy(test_loader)
validation_acc = accuracy(validation_loader)

print(f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}")

# Single Image Prediction

transform_index_to_disease = dataset.class_to_idx

transform_index_to_disease = dict([(value, key) for key, value in transform_index_to_disease.items()])

transform_index_to_disease

# data = pd.read_csv("disease_info.csv", encoding="cp1252")

from PIL import Image
import torchvision.transforms.functional as TF

def single_prediction(image_path):
  image = Image.open(image_path)
  image = image.resize((224, 224))
  input_data = TF.to_tensor(image)
  input_data = input_data.view((-1, 3, 224, 224))
  output = model(input_data)
  output = output.detach().numpy()
  index = np.argmax(output)
  print("Original : ", image_path[12:-4])
  pred = transform_index_to_disease[index]
  plt.imshow(image)
  plt.title("Disease Prediction : " + pred)
  plt.show()

single_prediction("test_images/Apple_ceder_apple_rust.JPG")

# Wrong Prediction

single_prediction("test_images/Apple_scab.JPG")

# Image outside dataset

single_prediction("test_images/background_without_leaves.jpg")