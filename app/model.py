import joblib
import sys
from pathlib import Path
from app.utils import get_logger
from app.config import MODEL_PATH

logger = get_logger("ModelLoader")

class SpamModel:
    def __init__(self, model_path: str = MODEL_PATH):
        logger.info(f"Loading model from {model_path}")
        
        # Import TrainingConfig to ensure it's available during unpickling
        try:
            from app.ham_spam_classifier import TrainingConfig
            # Make it available in __main__ for joblib unpickling
            sys.modules['__main__'].TrainingConfig = TrainingConfig
        except Exception as e:
            logger.warning(f"Could not import TrainingConfig: {e}")
        
        data = joblib.load(model_path)

        self.pipeline = data["pipeline"]
        self.config = data.get("config", None)

        logger.info("Model successfully loaded.")

    def predict(self, text: str):
        label = self.pipeline.predict([text])[0]

        prob_spam = None
        prob_ham = None

        clf = getattr(self.pipeline, "named_steps", {}).get("clf", None)
        if clf is not None and hasattr(self.pipeline, "predict_proba"):
            try:
                proba = self.pipeline.predict_proba([text])[0]
                # Assuming binary order [ham, spam]
                prob_ham = float(proba[0])
                prob_spam = float(proba[1])
            except Exception as e:
                logger.warning(f"Could not compute probabilities: {e}")

        return label, prob_spam, prob_ham
