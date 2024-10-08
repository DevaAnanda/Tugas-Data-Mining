# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Membuat dataset berdasarkan data yang dikirim
data = {
    'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', 'Spain', 'France', 'Germany', 'France'],
    'Age': [44, 27, 30, 38, 40, 35, 52, 48, 50, 37],
    'Salary': [72000, 48000, 54000, 61000, 61000, 58000, 52000, 79000, 83000, 67000],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Encoding data kategori (Country dan Purchased)
labelencoder_country = LabelEncoder()
df['Country'] = labelencoder_country.fit_transform(df['Country'])

labelencoder_purchased = LabelEncoder()
df['Purchased'] = labelencoder_purchased.fit_transform(df['Purchased'])

# Memisahkan data fitur dan target
X = df[['Country', 'Age', 'Salary']]
y = df['Purchased']

# Membagi dataset menjadi training set dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling (untuk Age dan Salary)
scaler = StandardScaler()
X_train[['Age', 'Salary']] = scaler.fit_transform(X_train[['Age', 'Salary']])
X_test[['Age', 'Salary']] = scaler.transform(X_test[['Age', 'Salary']])

# Menampilkan hasil preprocessing
print("Training Set:\n", X_train)
print("Test Set:\n", X_test)
