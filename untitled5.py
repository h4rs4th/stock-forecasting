import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

faetures=pd.read_csv("/content/sample_data/tesla-stock-price.csv")


print(np.shape(features))
if isinstance(features, pd.DataFrame):
    print(features.isnull().sum())
features = features.fillna(0)

if isinstance(features, pd.DataFrame):
  features = features.apply(pd.to_numeric, errors='coerce')
  features['open-close']  = features['open'] - features['close']
features['low-high']  = features['low'] - features['high']
features['target'] = np.where(features['close'].shift(-1) > ['close'], 1, 0)


scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

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



import pandas as pd
df=pd.read_csv("/content/sample_data/tesla-stock-price.csv")
print(df)
df.describe
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
prnt(df)





