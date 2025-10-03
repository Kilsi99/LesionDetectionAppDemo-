import joblib
import os
import pandas as pd
from app.models.meta_model import meta_model
from app.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
preprocessor_path = os.path.join(BASE_DIR, 'models', 'meta_preprocessor.pkl')

_preprocessor = None  # Global variable for caching

def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        try:
            _preprocessor = joblib.load(preprocessor_path)
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise RuntimeError("Metadata preprocessor could not be loaded.")
    return _preprocessor

def predict_metadata(age: int, sex, localisation):
    """Run metadata model prediction safely."""
    features = pd.DataFrame([[float(age), str(sex), str(localisation)]],
                            columns=['age', 'sex', 'localization'])
    preprocessor = get_preprocessor()  # Lazy load here
    probs_meta = meta_model.predict_proba(preprocessor.transform(features))
    pred_class_meta = int(probs_meta.argmax(axis=1)[0])
    probs_meta_list = probs_meta.astype(float).tolist()
    return pred_class_meta, probs_meta_list
