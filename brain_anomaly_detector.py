import numpy as np
import csv
import cv2
import os
from sklearn.metrics import f1_score


class BrainAnomalyDetector:
  train_data: np.array
  validation_data: np.array
  test_data: np.array

  train_labels: np.array
  validation_labels: np.array

  def __init__(self, folder_path: str, image_size = None, flip_images = False, normalize_images = False, bidimensional = False) -> None:
    self.read_data(folder_path, image_size, flip_images, normalize_images, bidimensional)

  def read_image(self, file_path: str, image_size: int, flip_image: bool, normalize_image: bool, bidimensional: bool) -> np.array:
    img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image_size: # daca i-am dat ca parametru o noua dimensiune
      img_data = cv2.resize(img_data, (image_size, image_size))

    if flip_image:
      img_data = cv2.flip(img_data, 1)

    if normalize_image:
      img_data = np.float32(img_data)
      img_data = cv2.normalize(img_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
      # img_data = np.uint8(img_data * 255)

    bidimensional_image = np.array(img_data)

    # pentru a transforma tabloul bidimensional intr-unul unidimensional
    if bidimensional:
      return [bidimensional_image]
    else:
      return np.reshape(bidimensional_image, (len(bidimensional_image[0]) ** 2))

  def read_data(self, folder_path: str, image_size: int , flip_images: bool, normalize_images: bool, bidimensional: bool) -> None:
    train_data = []
    validation_data = []
    test_data = []
    train_labels = []
    validation_labels = []

    # parcurg folderul cu imagini
    images_folder_path = os.path.join(folder_path, 'data')
    image_idx = 0
    print('Reading data...')
    files = sorted(os.listdir(images_folder_path))
    for filename in files:
      image_path = os.path.join(images_folder_path, filename)
      img_data = self.read_image(image_path, image_size, False, normalize_images, bidimensional)
      
      # in functie de indexul imaginii pe care o citim, o adaugam la categoria de care apartine
      data = train_data
      if image_idx >= 15000 and image_idx < 17000:
        data = validation_data
      elif image_idx >= 17000:
        data = test_data

      if image_idx % 1000 == 0:
        print('\r', end='')
        print(f'Loading {int(image_idx * 100 / len(files))}%', end='')
      data.append(img_data)
      if flip_images and image_idx < 15000:
        # o adaugam de 2 ori la train daca flip_images = true
        img_data = self.read_image(image_path, image_size, True, normalize_images, bidimensional)
        data.append(img_data)

      image_idx += 1
    print('\rLoading 100%')

    # citesc labelurile de train
    train_labels_path = os.path.join(folder_path, 'train_labels.txt')
    with open(train_labels_path, 'r') as file:
      reader = csv.reader(file)
      for idx, row in enumerate(reader):
        if idx == 0:
          continue
        train_labels.append(int(row[1]))
        if flip_images: # daca flip_images = true atunci adaugam labelul de 2 ori in lista
          train_labels.append(int(row[1]))


    # citesc labelurile de validare
    validation_labels_path = os.path.join(folder_path, 'validation_labels.txt')
    with open(validation_labels_path, 'r') as file:
      reader = csv.reader(file)
      for idx, row in enumerate(reader):
        if idx == 0:
          continue
        validation_labels.append(int(row[1]))

    self.train_data = np.array(train_data)
    self.validation_data = np.array(validation_data)
    self.test_data = np.array(test_data)
    self.train_labels = np.array(train_labels)
    self.validation_labels = np.array(validation_labels)

  def get_f1_score(self, prediction) -> float:
    return f1_score(self.validation_labels, prediction)