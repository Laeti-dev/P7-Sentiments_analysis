import csv
import pandas as pd
from pathlib import Path

FEEDBACK_FILE = Path(__file__).parent / "feedback" / "feedback_log.csv"

if FEEDBACK_FILE.exists():
    df = pd.read_csv(FEEDBACK_FILE)
    print("\n📊 Feedback Summary:")
    print(f"Total feedback: {len(df)}")
    print(f"\nCorrect predictions: {df['is_correct'].sum()}")
    print(f"Incorrect predictions: {(~df['is_correct']).sum()}")
    print(f"\nAccuracy: {df['is_correct'].mean() * 100:.2f}%")

    print("\n📋 Recent Feedback:")
    print(df.tail(10))
else:
    print("No feedback data yet")
