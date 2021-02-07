from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

from collections import Counter

print(sorted(Counter(y_resampled).items()))

from sklearn.svm import LinearSVC

clf = LinearSVC()
clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS

# In addition, RandomOverSampler allows to sample heterogeneous data (e.g. containing some strings):

import numpy as np

X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
                    dtype=np.object)
y_hetero = np.array([0, 0, 1])
X_resampled, y_resampled = ros.fit_resample(X_hetero, y_hetero)
print(X_resampled)

print(y_resampled)

# It would also work with pandas dataframe:
from sklearn.datasets import fetch_openml
df_adult, y_adult = fetch_openml(
    'adult', version=2, as_frame=True, return_X_y=True)
df_adult.head()  
df_resampled, y_resampled = ros.fit_resample(df_adult, y_adult)
df_resampled.head()  
