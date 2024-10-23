# Langkah 1: Preprocessing dataset
# Menghapus kolom 'User ID' karena tidak relevan untuk klasifikasi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Membaca dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Menghapus kolom 'User ID'
dataset_cleaned = dataset.drop('User ID', axis=1)

# Mengonversi kolom 'Gender' menjadi nilai numerik menggunakan LabelEncoder
label_encoder = LabelEncoder()
dataset_cleaned['Gender'] = label_encoder.fit_transform(dataset_cleaned['Gender'])

# Langkah 2: Memisahkan fitur (X) dan target (y)
X = dataset_cleaned.drop('Purchased', axis=1)
y = dataset_cleaned['Purchased']

# Langkah 3: Memisahkan dataset menjadi data latih dan data uji (80% data latih, 20% data uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Langkah 4: Menerapkan model Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Langkah 5: Melakukan prediksi pada data uji
y_pred = naive_bayes_model.predict(X_test)

# Langkah 6: Mengevaluasi kinerja model
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")
