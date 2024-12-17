from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import shap
import joblib
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to use the non-interactive 'Agg' backend
import matplotlib.pyplot as plt
from utils import get_shap_explanation  # Assuming this is where get_shap_explanation is located
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load pre-trained model
model = joblib.load('models/glioma_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

def convert_age_to_days(age_str):
    """
    Convert age from string format 'XX years YY days' to total days.
    If the format is invalid, return None (which will be handled later).
    """
    match = re.match(r'(\d+)\s*years?\s*(\d+)\s*days?', age_str)
    if match:
        years = int(match.group(1))
        days = int(match.group(2))
        return years * 365 + days
    # Return None if the age format is not as expected
    return None

def encode_mutation_status(data, mutation_columns):
    """
    Encode mutation status columns (MUTATED/NOT_MUTATED) to 1/0.
    """
    for col in mutation_columns:
        if col in data.columns:
            data[col] = data[col].map({'MUTATED': 1, 'NOT_MUTATED': 0}).fillna(0)
    return data

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and file.filename.endswith('.csv'):
        # Load CSV data
        data = pd.read_csv(file)

        # Drop unnecessary columns that were not used during model training
        columns_to_drop = ['Case_ID', 'Primary_Diagnosis', 'Project']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        # Convert 'Age_at_diagnosis' to numeric (total number of days or years)
        if 'Age_at_diagnosis' in data.columns:
            data['Age_at_diagnosis'] = data['Age_at_diagnosis'].apply(convert_age_to_days)

        # Handle missing 'Age_at_diagnosis' by replacing None with the mean value (or another strategy)
        data['Age_at_diagnosis'] = data['Age_at_diagnosis'].fillna(data['Age_at_diagnosis'].mean())
        
        # Encode categorical columns: Gender, Race, and Grade
        label_encoder = LabelEncoder()
        
        # Encode 'Gender' and 'Race' as these are categorical features
        if 'Gender' in data.columns:
            data['Gender'] = label_encoder.fit_transform(data['Gender'])
        if 'Race' in data.columns:
            data['Race'] = label_encoder.fit_transform(data['Race'])
        
        # Encode 'Grade' if it's the target column (target variable for classification)
        if 'Grade' in data.columns:
            data['Grade'] = label_encoder.fit_transform(data['Grade'])

        # Encode mutation columns (like IDH1, TP53, etc.)
        mutation_columns = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA',
                            'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4',
                            'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
        data = encode_mutation_status(data, mutation_columns)

        # Separate features and target variable
        X = data.drop('Grade', axis=1)  # Features
        y = data['Grade']  # Target variable (classification labels)

        # Optionally, scale continuous features like 'Age_at_diagnosis' and others (e.g., genetic markers)
        scaler = StandardScaler()
        continuous_features = ['Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
        
        # Check for the existence of the continuous features in the dataset and apply scaling
        continuous_features = [feature for feature in continuous_features if feature in data.columns]
        X[continuous_features] = scaler.fit_transform(X[continuous_features])
        
        # Model prediction
        predictions = model.predict(X)
        
        # Check that predictions length matches data length
        if len(predictions) != len(data):
            return jsonify({"error": f"Mismatch between the number of predictions ({len(predictions)}) and input data rows ({len(data)})"})

        # SHAP explanation
        shap_values = get_shap_explanation(model, X)
        
        # Visualization of SHAP values
        shap_summary_plot = 'static/images/shap_summary.png'
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)  # Set show=False to prevent GUI pop-up
        plt.savefig(shap_summary_plot)  # Save plot to static directory
        plt.close()  # Close the plot to free memory
        
        # Prediction results
        data['Risk Prediction'] = predictions
        risk_levels = ['LGG' if pred == 0 else 'GBM' for pred in predictions]
        data['Risk Level'] = risk_levels

        # Accuracy metrics (for demo purposes, assuming ground truth is available)
        accuracy = model.score(X, y)
        
        return render_template('index.html', 
                       table_html=data.to_html(classes='data', header=True), 
                       shap_image=shap_summary_plot, accuracy=accuracy)
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."})
    
if __name__ == '__main__':
    app.run(debug=True)
