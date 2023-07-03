from brain_anomaly_detector import BrainAnomalyDetector
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import torch, torchvision
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import torchvision.models as models
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

class CNN():
  def __init__(self, bad: BrainAnomalyDetector) -> None:
    self.bad = bad

  def create_resnet(self):
    resnet_model = models.resnet18(weights=None)
    # rescriem primul layer convolutional astfel incat sa accepte imagini grayscale
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    features_number = resnet_model.fc.in_features
    # si actualizam outputul din fully connected
    resnet_model.fc = nn.Linear(in_features=features_number, out_features=2)
    return resnet_model

  def validate(self, cnn, device):
    prediction = []
    
    # Cream un DataLoader pentru datele de validare
    tensor_images = torch.from_numpy(self.bad.validation_data)
    # desi nu foloses labelurile, da eroare daca nu le adaug in dataset
    tensor_labels = torch.from_numpy(self.bad.validation_labels) 
    validation_data = TensorDataset(tensor_images.to(device), tensor_labels.to(device))
    validation_dl = DataLoader(validation_data, batch_size=32)
    
    for images, _ in validation_dl:
      # parcurg batch-urile si salvez predictiile
      images = images.float().cuda() 
      output = cnn(images)
      _, batch_pred = torch.max(output, 1)
      batch_pred = batch_pred.data.cpu()
      prediction = prediction + batch_pred.tolist()
    
    return self.bad.get_f1_score(prediction)
    
  def test(self, output_folder, cnn, device):
    prediction = []
    
    tensor_images = torch.from_numpy(self.bad.test_data)
    # din nou, iau o lista de 0-uri drept labeluri doar ca sa nu dea erorare
    tensor_labels = torch.tensor([0 for _ in range(len(self.bad.test_data))])
    test_data = TensorDataset(tensor_images.to(device), tensor_labels.to(device))
    test_dl = DataLoader(test_data, batch_size=32)

    for images, _ in test_dl:
      images = images.float().cuda() 
      output = cnn(images)
      _, batch_pred = torch.max(output, 1)
      batch_pred = batch_pred.data.cpu()
      prediction = prediction + batch_pred.tolist()

    with open(os.path.join(output_folder, 'submission.csv'), 'a') as output_file:
      output_file.write('id,class\n')
      for idx, result in enumerate(prediction):
        output_file.write(f'0{idx + 17001},{result}\n')
  
  def train(self, num_epochs = 3, learning_rate = 0.001, device = 'cpu', make_plot=False):
    result = None
    scores = []
    max_score = 0

    tensor_images = torch.from_numpy(self.bad.train_data)
    tensor_labels = torch.from_numpy(self.bad.train_labels)
    train_data = TensorDataset(tensor_images.to(device), tensor_labels.to(device))
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    
    # cream modelul
    cnn_model = self.create_resnet().to(device)
    # definim functia de loss
    loss_function = nn.CrossEntropyLoss()
    # definim o functie de optimizare pentru a minimiza loss-ul
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
      cnn_model.train() # activam modul de train
      for images, labels in train_dl:
        # Resetarea gradientului înainte de optimizare pentru independența fiecărui mini-batch.
        optimizer.zero_grad()
        images = images.float().cuda() 
        batch_pred = cnn_model(images)
        loss = loss_function(batch_pred, labels)
        loss.backward()
        optimizer.step()
      
      cnn_model.eval() # activam modul de evaluare
      current_score = self.validate(cnn_model, device=device)
      scores.append(current_score)
      # salvam modelul daca e cel mai bun de pana acum
      if current_score > max_score:
        result = copy.deepcopy(cnn_model)
        max_score = current_score
        print('Best accuracy: ', current_score)
      print(f'Epoch: {epoch} | Accuracy: {current_score}% | Loss: {loss}')
    
    if make_plot:
      plt.plot(scores)
    return result


if __name__ == '__main__':
  if torch.cuda.is_available():
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")
  
  bad = BrainAnomalyDetector('input', bidimensional=True)
  cnn = CNN(bad)
  best_model = None
  max_score = 0
  scores = []
  options = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
  for i in options:
    model, score = cnn.train(50, i, device)
    scores.append(score)
    if score > max_score:
      max_score = score
      best_model = model
  plt.plot(options, scores)
  plt.show()