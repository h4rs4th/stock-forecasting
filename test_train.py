import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('/content/sample_data/tesla-stock-price.csv')

# Assuming the dataset has columns: 'Open', 'High', 'Low', 'Close', 'Volume', and 'Target'
# Adjust these names based on your actual column names

# Feature columns
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)
#df['volume'] = df['volume'].str.replace(',', '').astype(float)
# Feature Engineering: Use 'Close' price to predict future price movement
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Drop the last row which will have NaN target
#df.dropna(inplace=True)
# Target column
y = df['Target']
df=df[:-1]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM classifier
clf = SVC(kernel='linear')  # You can change the kernel if needed (e.g., 'rbf', 'poly')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))









