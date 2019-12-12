from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import os

_cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset = fetch_mldata(dataname="MNIST Original")

features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()

clf.fit(hog_features, labels)

model_path = os.path.join(_cur_dir, 'model/digits_cls.pkl')
joblib.dump(clf, model_path, compress=3)
