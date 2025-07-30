from data_utils import load_data, split_data, scale_data, load_yaml_file
import paths
import time
from vae.vae_utils import instantiate_vae_model, train_vae
import optuna
from metric_script import compute_catch_22_scores, compute_avg_wasserstein

def run_vae_pipeline(dataset_name: str, vae_type: str):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    # split data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

    # scale data
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model
    model_id = f"{vae_type}_{dataset_name}_{int(time.time())}"


    # load hyperparameters from yaml file
    config = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)
    hyperparameters = config[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        model_id=model_id,
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=config["common"]["max_epochs"],
        verbose=1,
    )

def objective(trial):
    latent_dim = trial.suggest_int('z_dim', 8, 64, step=8)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    units = trial.suggest_int('units', 32, 128, step=16)
    warumup_epochs = trial.suggest_int('warmup_epochs', 50, 200, step=50)
    max_epochs = trial.suggest_int('max_epochs', 200, 300, step=50)

    hyperparameters = {'latent_dim': latent_dim,
                    'bidirectional': bidirectional,
                    'units': units,
                    'hidden_layer_sizes': [48, 96, 192],
                    'reconstruction_wt': 1,
                    'reconstruction_wt_bound': 0.2,
                    'batch_size': 16,
                    'warmup_epochs': warumup_epochs}

    
    vae_type = "vae_lstm"
    dataset_name = "jerkEventSubset_20"
    sequence_length = 20
    feature_dim = 1

    print(hyperparameters)


    model_id = f"{vae_type}_{dataset_name}_{int(time.time())}"
    vae_model = instantiate_vae_model(
        model_id=model_id,
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)
    train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)
    scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=max_epochs,
        verbose=1,
    )

    # Compute loss metric

    prior_data = vae_model.get_prior_samples(1000)
    original_data = scaled_train_data[:1000]

    original_scores, prior_scores = compute_catch_22_scores(original_data, prior_data)
    wasserstein_distance = compute_avg_wasserstein(prior_scores, original_scores)
    
    return wasserstein_distance

import optuna
import os

# --- CONFIGURATION ---
STUDY_NAME = "vae_hyperopt_study"
STORAGE_PATH = "sqlite:///vae_hyperopt_study.db"

# Optional: ensure you're not duplicating studies
if os.path.exists("vae_hyperopt_study.db"):
    print(f"📂 Loading existing study '{STUDY_NAME}' from file.")
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
else:
    print(f"🆕 Creating new study '{STUDY_NAME}'...")
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
    )

# --- OPTIMIZATION LOOP ---
study.optimize(objective, n_trials=50)

# --- OPTIONAL: SAVE BEST PARAMS OR LOG ---
best_params = study.best_params
print("✅ Best Parameters:", best_params)

# Save best trial manually to a JSON file
import json
with open("best_trial.json", "w") as f:
    json.dump({
        "params": best_params,
        "value": study.best_value,
        "trial_number": study.best_trial.number
    }, f, indent=2)
