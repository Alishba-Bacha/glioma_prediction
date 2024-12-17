import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from utils import prepare_data  # Import the preprocessing function

# Load dataset (assumed to be CSV for this example)
data = pd.read_csv('glioma_dataset.csv')

# Preprocess data using the function from utils.py
X, y = prepare_data(data)  # This returns preprocessed features and target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/glioma_model.pkl')

# Optionally, evaluate the model and print accuracy
accuracy = model.score(X_test, y_test)
print(f'Model accuracy on test data: {accuracy}')
