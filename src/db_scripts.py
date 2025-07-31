import optuna
import os
from tabulate import tabulate
import argparse

# === CONFIGURATION ===
history_path = "D:/Studiumj/Master Thesis/Data/Histories/"
DB_NAME = "vae_hyperopt_study_1.db"
# DB_PATH = os.path.join(history_path, DB_NAME)
STUDY_NAME = "optuna_study"  # Replace with your actual study name
ORDER_BY = "value"  # or "number"


parser = argparse.ArgumentParser(description="Inspect Optuna study in a SQLite .db file")
parser.add_argument("db_path", type=str, help="Path to the Optuna SQLite database (.db)")
args = parser.parse_args()
DB_PATH = os.path.abspath(args.db_path)


# === Load Study ===
if not os.path.isfile(DB_PATH):
    raise FileNotFoundError(f"Database file not found: {DB_PATH}")

storage_url = f"sqlite:///{DB_PATH}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage_url)

# === Extract Trials ===
trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

if ORDER_BY == "value":
    trials.sort(key=lambda t: t.value)
elif ORDER_BY == "number":
    trials.sort(key=lambda t: t.number)
else:
    raise ValueError(f"Unsupported ORDER_BY value: {ORDER_BY}")

# === Display Results ===
table = []
for t in trials:
    row = {
        "Trial #": t.number,
        "Value": t.value,
        "Params": t.params,
        "Duration": str(t.duration),
    }
    table.append(row)

print(f"\nStudy: {study.study_name}")
print(f"Best Trial: #{study.best_trial.number} (Value: {study.best_trial.value})")
print("\nTop Trials:\n")
print(tabulate(table, headers="keys", tablefmt="pretty"))