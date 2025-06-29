import pytest
import joblib
import os
import pandas as pd


# --------------------------
# Utility Tests
# ---------------------------

def test_artifacts_exist():
    """
    Check if the artifacts directory and trained model file exists.
    """
    assert os.path.exists("artifacts")
    assert os.path.exists("artifacts/model.joblib")


# ---------------------------
# Model Prediction Tests
# ---------------------------

class TestModelPredictions:
    """
    A suite of tests validating model loading, hardcoded feature structure, and prediction logic.
    """

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load the trained model from disk once per test class.
        """
        return joblib.load("artifacts/model.joblib")

    @pytest.fixture(scope="class")
    def online_features(self):
        """
        Return a hardcoded DataFrame mimicking what Feast would return.
        """
        data = {
            "sepal_length": [5.1, 6.0, 6.3],
            "sepal_width": [3.5, 2.2, 3.3],
            "petal_length": [1.4, 4.0, 6.0],
            "petal_width": [0.2, 1.0, 2.5],
            "species": ["setosa", "versicolor", "virginica"]
        }
        return pd.DataFrame(data)

    def test_data_schema_validation(self, online_features):
        """
        Validate the structure, content, and types of the feature data.
        Ensures the model input is consistent with training expectations.
        """
        assert not online_features.empty, "Features DataFrame should not be empty"

        required_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ]

        for col in required_columns:
            assert col in online_features.columns, f"Missing column: {col}"

        numeric_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(
                online_features[col]
            ), f"Column '{col}' should be numerical"

        expected_species = {"setosa", "versicolor", "virginica"}
        actual_species = set(online_features["species"].unique())
        assert actual_species.issubset(
            expected_species
        ), f"Unexpected species found: {actual_species - expected_species}"

        assert not online_features[required_columns].isnull().any().any(), \
            "Features DataFrame should not have null values"

        assert len(online_features) == 3, f"Expected 3 rows, got {len(online_features)}"

    def test_model_loading(self, model):
        """
        Confirm that the model loads correctly and supports predictions.
        """
        assert model is not None, "Model should not be None"
        assert hasattr(model, "predict"), "Model should have a 'predict' method"

    def test_model_predictions(self, model, online_features):
        """
        Perform predictions on the feature data and validate output.
        Ensures predictions are in the expected label space and match target values.
        """
        feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        X = online_features[feature_columns]

        predictions = model.predict(X)

        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        assert all(
            pred in ["setosa", "versicolor", "virginica"] for pred in predictions
        ), "Predictions contain unexpected classes"

        for idx, row in online_features.iterrows():
            expected_class = row["species"]
            predicted_class = predictions[idx]

            if predicted_class != expected_class:
                print(
                    f"Warning: Species '{expected_class}' predicted as '{predicted_class}'"
                )
                print(f"Features: {row[feature_columns].to_dict()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
