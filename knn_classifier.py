from brain_anomaly_detector import BrainAnomalyDetector
import matplotlib.pyplot as plt
import numpy as np
from typing import List

class DistanceMetric:
  L1 = 'L1'
  L2 = 'L2'

class KnnClassifierModel(BrainAnomalyDetector):
  def __init__(self, folder_path: str, image_size = None, flip_images = False, normalize_images = False) -> None:
    super().__init__(folder_path, image_size, flip_images, normalize_images)

  def get_distance(self, img1: np.array, img2: np.array, metric = DistanceMetric.L1):
    # Calculeaza distanta L1 sau L2 dintre 2 imagini
    if metric == DistanceMetric.L1:
      return np.sum(np.abs(img1 - img2))
    elif metric == DistanceMetric.L2:
      return np.sqrt(np.sum((img1 - img2)**2))
    else:
      raise Exception("Metric value should be L1 or L2. Try using the DistanceMetric class")


  def classify_image(self, test_image: np.array, neighbours_number = 3, metric = DistanceMetric.L1):
    # vom avea un vector unde salvam cei mai apropriati vecini, cu formatul [clasa imaginii, distanta]
    closest_neighbours = [[float('inf'), float('inf')] for _ in range(neighbours_number)]

    for idx, image in enumerate(self.train_data):
      image_class = self.train_labels[idx]
      # calculam distanta pt catre fiecare imagine de train
      dist = self.get_distance(test_image, image, metric)
      # primele 'neighbours_number' imagini le bagam din prima in lista
      if idx < neighbours_number:
        # adaugam imaginea si sortam lista
        closest_neighbours[idx] = [image_class, dist]
        closest_neighbours.sort(key=lambda x: x[1])
      else:
        for i in range(len(closest_neighbours)):
          if dist < closest_neighbours[i][1]:
            # tre sa dam shift la dreapta valorilor de la i incolo
            for j in range(len(closest_neighbours) - 1, i - 1, -1):
              closest_neighbours[j] = closest_neighbours[j-1]
            closest_neighbours[i] = [image_class, dist]
            break
    
    # selectam clasele imaginilor
    result_classes = [closest_neighbours[i][0] for i in range(len(closest_neighbours))]
    # si returnam cea mai frecventa clasa
    # return np.argmax(np.bincount(result_classes))

    # aici testez returnez pozitiv daca macar unul dintre vecini e pozitiv.
    if len(np.bincount(result_classes)) > 1 and np.bincount(result_classes)[1] >= 2 : return 1
    else: return 0

  def validate_model(self, neighbours_number = 3, metric = DistanceMetric.L1) -> float:
    prediction = []
    confusion_matrix = [
      [0, 0], 
      [0, 0]
    ]
    for idx, image in enumerate(self.validation_data):
      result = self.classify_image(image, neighbours_number, metric)
      prediction.append(result)

      if (result == self.validation_labels[idx]):
        # Daca rezultatul e corect
        if (result == 1): 
          confusion_matrix[0][0] += 1
        else:
          confusion_matrix[1][1] += 1

        print(f"Image {idx}: valid")
      else:
        # Daca rezultatul e incorect
        if (result == 1):
          confusion_matrix[1][0] += 1
        else:
          confusion_matrix[0][1] += 1

        print(f"Image {idx}: invalid")
    print(confusion_matrix)
    return self.get_f1_score(prediction)

  def test_model(self, output_file_name: str, neighbours_number = 3, metric = DistanceMetric.L1):
    with open(output_file_name, 'a') as output_file:
      output_file.write('id,class\n')
      for idx, image in enumerate(self.test_data):
        result = self.classify_image(image, neighbours_number, metric)
        output_file.write(f'0{idx + 17001},{result}\n')

  def neighbours_score_chart(self, neighbours_number_values: List[int]):
    l1_scores = []
    l2_scores = []
    for nr in neighbours_number_values:
      print(f"\nStarting L1 validation for {nr} neighbours...")
      score = self.validate_model(nr, DistanceMetric.L1)
      l1_scores.append(score)
      print(score)

    for nr in neighbours_number_values:
      print(f"\nStarting L2 validation for {nr} neighbours...")
      score = self.validate_model(nr, DistanceMetric.L2)
      l2_scores.append(score)
      print(score)
      
    plt.plot(neighbours_number_values, l1_scores, label='L1')
    plt.plot(neighbours_number_values, l2_scores, label='L2')
    plt.title("Knn Classifier")
    plt.xlabel("Bins number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
  bad = KnnClassifierModel('input', 100, False, True)
  print(bad.validate_model(5, DistanceMetric.L1))

