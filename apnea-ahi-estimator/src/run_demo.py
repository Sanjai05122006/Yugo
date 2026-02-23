
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import argparse
from pipeline import run_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--csv", type=str, default="data/epochs_features.csv")
    args = parser.parse_args()

    if not args.offline:
        raise SystemExit("Use --offline")

    csv_path = Path(args.csv)
    print("Running pipeline on:", csv_path)

    results = run_pipeline(csv_path)

    print("=== DONE ===")
    print("AHI:", results["ahi"])
    print("Severity:", results["severity"])
    print("Apnea epochs:", results["apnea_epochs"])
    print("Outputs saved to /outputs/")
