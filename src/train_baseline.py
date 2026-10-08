"""Week 2 starter: baseline classifier on a public tabular dataset.

Replace the dataset loader with your chosen OpenML/UCI source.
Goal: train a baseline, then add an explanation path in explain.py.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
report = classification_report(y_test, pred, output_dict=True)
(OUT / "baseline_report.json").write_text(json.dumps(report, indent=2))
print("Wrote", OUT / "baseline_report.json")
print(classification_report(y_test, pred))
