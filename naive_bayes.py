from sklearn.naive_bayes import MultinomialNB
from brain_anomaly_detector import BrainAnomalyDetector
import matplotlib.pyplot as plt
import numpy as np


class NaiveBayesModel(BrainAnomalyDetector):
  def __init__(self, folder_path: str, image_size = None, flip_images = False) -> None:
    super().__init__(folder_path, image_size, flip_images)

  def train_model(self, bins_number = 5) -> MultinomialNB:
    print("Training Naive Bayes Model...")
    naive_bayes_model = MultinomialNB()
    bins = np.linspace(start=0, stop=255, num=bins_number)

    # folosim partial fit, pe cate 1000 imagini pentru ca nu avem destul ram sa le folosim pe toate din prima
    # am facut primul pas deasupra for-ului pentru a putea atribui parametrul classes la primul partia_fit, altfel nu functionaza
    step = 500
    first_elements = self.train_data[0:step]
    train_data = np.digitize(first_elements, bins) - 1
    naive_bayes_model.partial_fit(train_data, self.train_labels[0:step], classes=[0,1])

    print("Loading 0%", flush=True, end='')
    for i in range(step, len(self.train_data), step):
      print('\r', end='')
      next_elements = self.train_data[i:i+step]
      train_data = np.digitize(next_elements, bins) - 1
      naive_bayes_model.partial_fit(train_data, self.train_labels[i:i+step])
      print(f"Loading {int(i * 100 / len(self.train_data))}%", end='', flush=True)
    print("\rLoading 100%")

    return naive_bayes_model
  
  def validate_model(self, naive_bayes_model: MultinomialNB, bins_number = 5):
    bins = np.linspace(start=0, stop=255, num=bins_number)
    validation_data = np.digitize(self.validation_data, bins) - 1
    prediction = naive_bayes_model.predict(validation_data)
    return prediction
  
  def test_model(self, output_file_name:str, naive_bayes_model:MultinomialNB, bins_number = 5):
    # facem o predictie pe datele de test
    bins = np.linspace(start=0, stop=255, num=bins_number)
    test_data = np.digitize(self.test_data, bins) - 1
    prediction = naive_bayes_model.predict(test_data)

    # apoi cream csv-ul cu rezultatul
    with open(output_file_name, 'a') as output_file:
      output_file.write('id,class\n')
      for idx, result in enumerate(prediction):
        output_file.write(f'0{idx + 17001},{result}\n')
  
  def bins_score_chart(self, bins_options:list):
    # si vom salva pt fiecare valoare acuratetea
    scores = []
    for bins_number in bins_options:
      print(f"Testing for {bins_number} bins...")
      naive_bayes_model = self.train_model(bins_number)
      prediction = self.validate_model(naive_bayes_model, bins_number)
      score = self.get_f1_score(prediction)
      scores.append(score)
      print(f"Score {score}")
    
    plt.plot(bins_options, scores)
    plt.title("Naive Bayes")
    plt.xlabel("Bins number")
    plt.ylabel("Accuracy")
    plt.show()
  
      


if __name__ == '__main__':
  bad = NaiveBayesModel('input', 100)
  bad.bins_score_chart([2, 4, 6, 8, 10, 12, 14])