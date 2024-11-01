import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Memuat dataset dan menambahkan nama kolom
column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv("wdbc.data", header=None, names=column_names)

# Menghapus kolom ID yang tidak relevan dan mengubah label menjadi numerik
data = data.drop(columns=['ID'])
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})  # M untuk Malignant, B untuk Benign

# Memisahkan fitur dan label
X = data.drop(columns=['Diagnosis']).values
y = data['Diagnosis'].values

# Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi menghitung jarak Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Fungsi K-NN
def knn(X_train, y_train, X_test, k):
    distances = [euclidean_distance(X_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Melakukan prediksi pada data uji
k = 5
predictions = [knn(X_train, y_train, x_test, k) for x_test in X_test]

# Menghitung akurasi
accuracy = accuracy_score(y_test, predictions)
print("Akurasi model K-NN:", accuracy)
