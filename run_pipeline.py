"""Top-level CLI: run the full retraining + prediction pipeline."""
from src.predict import run_full_pipeline
from src.utils import get_logger

log = get_logger("run_pipeline")


if __name__ == "__main__":
    df = run_full_pipeline(retrain=True)
    print("\n=== SEGMENT DISTRIBUTION ===")
    print(df["segment"].value_counts())
    print("\nDone.")
