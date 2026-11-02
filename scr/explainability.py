
import shap
import pandas as pd
import numpy as np


class ChurnExplainer:

    def __init__(self, model, background_data):

        self.model = model

        self.background_data = background_data

        self.feature_names = background_data.columns.tolist()

        # Model prediction function
        # Returns churn probability instead of class labels

        def predict_probability(data):

            if not isinstance(data, pd.DataFrame):

                data = pd.DataFrame(
                    data,
                    columns=self.feature_names
                )

            return self.model.predict_proba(data)[:, 1]

        self.predict_probability = predict_probability

        # Create independent background masker
        masker = shap.maskers.Independent(
            self.background_data
        )

        # Create model-agnostic SHAP explainer
        self.explainer = shap.PermutationExplainer(
            self.predict_probability,
            masker
        )

    def explain(self, customer_data):

        # Convert input into DataFrame
        customer_data = pd.DataFrame(
            [customer_data],
            columns=self.feature_names
        )

        # Calculate SHAP values
        shap_values = self.explainer(
            customer_data,
            max_evals=2 * len(self.feature_names) + 1
        )

        # Extract SHAP values
        feature_values = shap_values.values[0]

        base_value = float(
            np.asarray(shap_values.base_values).reshape(-1)[0]
        )

        # Get actual prediction probability
        probability = float(
            self.predict_probability(customer_data)[0]
        )

        explanations = []

        for feature, value, shap_value in zip(
            self.feature_names,
            customer_data.iloc[0].values,
            feature_values
        ):

            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()

            explanations.append({

                "feature": feature,

                "value": value,

                "shap_value": float(shap_value),

                "impact": (
                    "increases churn risk"
                    if shap_value > 0
                    else "decreases churn risk"
                    if shap_value < 0
                    else "no impact"
                )
            })

        # Sort features by absolute SHAP importance
        explanations.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )

        return {

            "churn_probability": probability,

            "base_value": base_value,

            "explanations": explanations

        }
