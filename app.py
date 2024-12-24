from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import shap
import joblib
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from utils import get_shap_explanation  
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
model = joblib.load('models/glioma_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

def convert_age_to_days(age_str):
    
    match = re.match(r'(\d+)\s*years?\s*(\d+)\s*days?', age_str)
    if match:
        years = int(match.group(1))
        days = int(match.group(2))
        return years * 365 + days
    return None

def encode_mutation_status(data, mutation_columns):
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
        data = pd.read_csv(file)
        columns_to_drop = ['Case_ID', 'Primary_Diagnosis', 'Project']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        if 'Age_at_diagnosis' in data.columns:
            data['Age_at_diagnosis'] = data['Age_at_diagnosis'].apply(convert_age_to_days)

        data['Age_at_diagnosis'] = data['Age_at_diagnosis'].fillna(data['Age_at_diagnosis'].mean())
        label_encoder = LabelEncoder()
        if 'Gender' in data.columns:
            data['Gender'] = label_encoder.fit_transform(data['Gender'])
        if 'Race' in data.columns:
            data['Race'] = label_encoder.fit_transform(data['Race'])
        if 'Grade' in data.columns:
            data['Grade'] = label_encoder.fit_transform(data['Grade'])
        mutation_columns = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA',
                            'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4',
                            'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
        data = encode_mutation_status(data, mutation_columns)

        X = data.drop('Grade', axis=1)  
        y = data['Grade'] 
        scaler = StandardScaler()
        continuous_features = ['Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
        continuous_features = [feature for feature in continuous_features if feature in data.columns]
        X[continuous_features] = scaler.fit_transform(X[continuous_features])
        predictions = model.predict(X)
        
        if len(predictions) != len(data):
            return jsonify({"error": f"Mismatch between the number of predictions ({len(predictions)}) and input data rows ({len(data)})"})

        shap_values = get_shap_explanation(model, X)

        shap_summary_plot = 'static/images/shap_summary.png'
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)  # Set show=False to prevent GUI pop-up
        plt.savefig(shap_summary_plot) 
        plt.close()  
        data['Risk Prediction'] = predictions
        risk_levels = ['LGG' if pred == 0 else 'GBM' for pred in predictions]
        data['Risk Level'] = risk_levels
        accuracy = model.score(X, y)
        
        return render_template('index.html', 
                       table_html=data.to_html(classes='data', header=True), 
                       shap_image=shap_summary_plot, accuracy=accuracy)
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."})
    
if __name__ == '__main__':
    app.run(debug=True)
