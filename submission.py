import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from skimage.transform import resize

test_dir = r'C:\Users\ragha\Desktop\New folder\ietemlcompetition\Dataset\test'
model = load_model('test1.h5')
test_csv = pd.read_csv('Test.csv')
number_to_class = ['0', '1', '2', '3', '4', '5', '6', '7']
final = []

for file in test_csv['Id']:
    path = os.path.join(test_dir, file)
    my_image = plt.imread(path)
    my_image_resized = resize(my_image, (64, 64, 3))
    probabilities = model.predict(np.array([my_image_resized, ]))
    index = np.argsort(probabilities[0, :])
    final.append(number_to_class[index[7]])

df = pd.DataFrame(final, columns = ['Label'])
df.to_csv('final_submission.csv')









