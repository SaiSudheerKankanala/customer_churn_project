
import joblib
import pandas as pd

from scr.explainability import ChurnExplainer


class prediction:

    def __init__(self):
        artifact = joblib.load("artifacts/best_model.pkl")
        self.model = artifact["model"]
        self.background_data = artifact["background_data"]
        self.explainer = ChurnExplainer(
            self.model,
            self.background_data
        )

    def predict(self, data):

        input_data = pd.DataFrame(
            [data],
            columns=self.background_data.columns
        )

        prediction_result = int(
            self.model.predict(input_data)[0]
        )

        explanation = self.explainer.explain(data)

        return {

            "prediction": prediction_result,

            "churn_probability": explanation["churn_probability"],

            "explanation": explanation["explanations"][:5]

        }
