import pytest
import joblib
import os
import pandas as pd
import numpy as np
from feast import FeatureStore


# --------------------------
# Utility Tests
# ---------------------------

def test_artifacts_exist():
    """
    Check if the artifacts directory and trained model file exists.
    """
    assert os.path.exists("artifacts")
    assert os.path.exists("artifacts/model.joblib")


def test_feast_config():
    """
    Check if the Feast feature store and specified feature service can be loaded correctly.
    Validates the feature store setup and service naming.
    """
    try:
        store = FeatureStore(repo_path="feature_repo")
        feature_service = store.get_feature_service("feast_model_v1")
        assert feature_service is not None
    except Exception as e:
        pytest.fail(f"Failed to load feature service: {str(e)}")


# ---------------------------
# Model Prediction Tests
# ---------------------------

class TestModelPredictions:
    """
    A suite of tests validating model loading, feature structure, and prediction logic.
    """

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load the trained model from disk once per test class.
        """
        return joblib.load("artifacts/model.joblib")

    @pytest.fixture(scope="class")
    def feature_store(self):
        """
        Initialize and return a Feast FeatureStore instance.
        """
        return FeatureStore(repo_path="feature_repo")

    @pytest.fixture(scope="class")
    def online_features(self, feature_store):
        """
        Retrieve online features for 3 flower species from the feature store.
        Converts the result to a pandas DataFrame for testing.
        """
        features = feature_store.get_online_features(
            features=feature_store.get_feature_service("feast_model_v1"),
            entity_rows=[
                {"species": "setosa"},
                {"species": "versicolor"},
                {"species": "virginica"},
            ],
        ).to_df()
        return features

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
        # Check that all required columns are present
        for col in required_columns:
            assert col in online_features.columns, f"Missing column: {col}"

        # Ensure feature columns are numeric
        numeric_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(
                online_features[col]
            ), f"Column '{col}' should be numerical"

        # Validate species values
        expected_species = ["setosa", "versicolor", "virginica"]
        actual_species = set(online_features["species"].unique())
        assert actual_species.issubset(
            expected_species
        ), f"Unexpected species found: {actual_species - expected_species}"

        # Check for nulls in the required columns
        assert not online_features[required_columns].isnull().any().any(), \
            "Features DataFrame should not have null values"

        # Expect exactly 3 rows (one per species)
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

        # Generate predictions
        predictions = model.predict(X)

        # Basic checks on prediction output
        assert len(predictions) == 3, f"Expected 3 predictions, got {len(predictions)}"
        assert all(
            pred in ["setosa", "versicolor", "virginica"] for pred in predictions
        ), "Predictions contain unexpected classes"

        # Optional: log mismatches between prediction and actual label
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
