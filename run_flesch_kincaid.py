import textstat
from datasets import Dataset
import numpy as np

dataset = Dataset.from_csv("weebit-cache-test.csv")
dataset = dataset.to_dict()
correct_guesses = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
errors = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_guesses = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
grade_labels = []

for (text, label) in zip(dataset["text"], dataset["label"]):
    predicted_level = textstat.flesch_kincaid_grade(text)

    # 2nd grade, 3rd grade, 4th grade, 6-8th grade, 10th grade
    if (label == 0 and 1 <= predicted_level <= 3) or (label == 1 and 2 <= predicted_level <= 4) or (label == 2 and 3 <= predicted_level <= 5) or (label == 3 and 5 <= predicted_level <= 9) or (label == 4 and 9 <= predicted_level <= 11):
        correct_guesses[label] += 1
        correct_guesses[5] += 1  # 5 is an extra label used for describing overall performance
    elif label == 0:
        errors[label] += min(np.abs(predicted_level - 1), np.abs(predicted_level - 3)) ** 2
        errors[5] += min(np.abs(predicted_level - 1), np.abs(predicted_level - 3)) ** 2
    elif label == 1:
        errors[label] += min(np.abs(predicted_level - 2), np.abs(predicted_level - 4)) ** 2
        errors[5] += min(np.abs(predicted_level - 2), np.abs(predicted_level - 4)) ** 2
    elif label == 2:
        errors[label] += min(np.abs(predicted_level - 3), np.abs(predicted_level - 5)) ** 2
        errors[5] += min(np.abs(predicted_level - 3), np.abs(predicted_level - 5)) ** 2
    elif label == 3:
        errors[label] += min(np.abs(predicted_level - 5), np.abs(predicted_level - 9)) ** 2
        errors[5] += min(np.abs(predicted_level - 5), np.abs(predicted_level - 9)) ** 2
    elif label == 4:
        errors[label] += min(np.abs(predicted_level - 9), np.abs(predicted_level - 11)) ** 2
        errors[5] += min(np.abs(predicted_level - 9), np.abs(predicted_level - 11)) ** 2

    total_guesses[label] += 1
    total_guesses[5] += 1

print("Accuracy:")
for label in correct_guesses:
    print(str(label) + ": " + str(correct_guesses[label] / total_guesses[label]))

print("RMSE:")
for label in errors:
    print(str(label) + ": " + str(np.sqrt(errors[label] / total_guesses[label])))
