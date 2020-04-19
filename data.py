import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

data_file = pd.read_csv(r'ietecomp\train.csv')

root_dir = r'C:\Users\ragha\Desktop\New folder\ietemlcompetition'
base_dir = 'Dataset'

path_root = os.path.join(root_dir, base_dir)
try:
    os.mkdir(path_root)
except:
    pass

train_dir = os.path.join(path_root, 'train')
try:
    os.mkdir(train_dir)
except:
    pass

valid_dir = os.path.join(path_root, 'valid')
try:
    os.mkdir(valid_dir)
except:
    pass
try:
    for i in range(8):
        os.mkdir(os.path.join(train_dir, str(i)))
        os.mkdir(os.path.join(valid_dir, str(i)))
except:
    pass

src = r'C:\Users\ragha\Desktop\New folder\ietemlcompetition\ietecomp\ietecompdata'

for i in range(8):
    label = data_file.loc[data_file['Label1'] == i]
    print("Number of Images in Label "+str(i)+" is "+str(len(label)))
    try:
        train, valid = train_test_split(label, test_size=0.2, random_state=42)
    except:
        pass
    print("Training set Images : "+str(len(train)))
    print("Validation set Images : "+str(len(valid)))
    train_path = os.path.join(train_dir, str(i))
    valid_path = os.path.join(valid_dir, str(i))
    print("STARTED COPYING..............")
    for file in train['Filename']:
        path = os.path.join(src, file)
        shutil.copy(path, train_path)
    for file in valid['Filename']:
        path = os.path.join(src, file)
        shutil.copy(path, valid_path)
    print("FINISHED COPYING............")
    print(" ")

