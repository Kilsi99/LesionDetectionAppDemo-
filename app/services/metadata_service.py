import numpy as np
from app.models.meta_model import meta_model
from app.utils.logger import get_logger
import cloudpickle  # <- use cloudpickle instead of joblib
import pandas as pd
import os

logger = get_logger(__name__)

# Set base directory (root of your project)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

preprocessor_path = os.path.join(BASE_DIR, 'models', 'meta_preprocessor.pkl')

# Load preprocessor safely
try:
    with open(preprocessor_path, 'rb') as f:
        preprocessor = cloudpickle.load(f)
    logger.info("Meta preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load meta preprocessor: {e}")
    preprocessor = None  # fallback, so the app doesn't crash immediately

def predict_metadata(age: int, sex, localisation):
    """Run metadata model prediction safely."""
    if preprocessor is None:
        logger.error("Preprocessor not loaded, cannot run prediction")
        return None, [0] * 7  # Return dummy values

    features = pd.DataFrame(
        [[float(age), str(sex), str(localisation)]],
        columns=['age', 'sex', 'localization']
    )

    probs_meta = meta_model.predict_proba(preprocessor.transform(features))
    
    pred_class_meta = int(probs_meta.argmax(axis=1)[0])
    probs_meta_list = probs_meta.astype(float).tolist()
    
    return pred_class_meta, probs_meta_list
