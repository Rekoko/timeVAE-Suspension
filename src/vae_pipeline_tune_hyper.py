import os, warnings
import numpy as np
import time
import pandas as pd
from itertools import product
from model_saving import append_row_to_csv

from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne

def generate_hyperparameter_list(config):
    """
    Generate a list of hyperparameter configurations from the provided config.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.

    Returns:
        list: A list of hyperparameter dictionaries.
    """
    hyperparameter_list = []
    for key, value in config.items():
        if isinstance(value, list):
            for v in value:
                hyperparameter_list.append({key: v})
        else:
            hyperparameter_list.append({key: value})
    
    return hyperparameter_list

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


    # load hyperparameters from yaml file
    config = load_yaml_file(paths.TUNE_CONFIGS_PATH)
    param_dict = config["vae_conv"]

    keys = list(param_dict)
    value_lists = list(param_dict.values())

    hyperparameter_list = [dict(zip(keys, values)) for values in product(*value_lists)]
    for hyperparameters in hyperparameter_list:
        # Create id per tune run
        model_id = f"{vae_type}_{dataset_name}_{int(time.time())}"


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

        # ----------------------------------------------------------------------------------
        # Save scaler and model
        model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name, model_id)
        # save scaler
        save_scaler(scaler=scaler, dir_path=model_save_dir)
        # Save vae
        save_vae_model(vae=vae_model, dir_path=model_save_dir)
        # Add model with parameters to the model list
        model_params = {
            "model_id": model_id,
            "model_type": vae_type,
            "dataset_name": dataset_name,
            "date": time.strftime("%Y-%m-%d"),
            }
        
        for key, value in hyperparameters.items():
            if isinstance(value, list):
                model_params[key] = [value]
            else:
                model_params[key] = value
        for key, value in config["common"].items():
            model_params[key] = value
        
        current_model = pd.DataFrame(model_params, index=[0])
        append_row_to_csv(current_model, paths.MODEL_LIST_PATH)

if __name__ == "__main__":
    # check `/data/` for available datasets
    dataset = "jerkEvents_20"

    # models: vae_dense, vae_conv, timeVAE
    model_name = "vae_conv"

    run_vae_pipeline(dataset, model_name)
