import pandas as pd
import shap
import matplotlib.pyplot as plt

# SHAP Explanation Function (Without Preprocessing)
def get_shap_explanation(model, X):
    """
    Generate SHAP values and explanations for the given model and dataset.
    
    Args:
        model: Trained machine learning model (e.g., RandomForestClassifier).
        X: Features dataset for which SHAP values need to be generated.
    
    Returns:
        shap_values: SHAP values for the given model and data.
    """
    # Initialize SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Get SHAP values for the features
    shap_values = explainer.shap_values(X)

    # Save the SHAP summary plot to a file (avoid plt.show() in non-interactive environments)
    shap_summary_plot_path = 'static/images/shap_summary.png'
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(shap_summary_plot_path)
    plt.close()

    return shap_values
