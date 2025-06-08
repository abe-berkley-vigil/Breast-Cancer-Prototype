import logging
import os.path
from pathlib import Path
import pickle
import sys
import time

import hydra
from loguru import logger

#imports for cml metrics and plots
import matplotlib.pyplot as plt  # REQUIRED for plotting
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from rich import print as rprint

# Rich imports for enhanced logging
from rich.console import Console
from rich.logging import RichHandler
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # REQUIRED for evaluation
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import typer

## Required otherwise it will error out in the makefile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')) )

from breast_cancer_classification.dataset import load_data

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("train")

app = typer.Typer()


def create_test_train_split(data: pd.DataFrame, debug: bool = False,test_size: float = 0.2, random_state: int = 21):
    """Create the test train split for the data.

    Args:
        data: The dataframe containing the data.
        debug: The option to turn on printing information about the training and test shapes.

    Returns:
        The X,y train and test splits.

    """
    X = data.drop(["diagnosis", "id"], axis=1)
    y = data["diagnosis"]
    # Split data into training and testing sets

    log.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    load_time = time.time() - start_time
    log.info(f"Data split successfully in {load_time:.2f} seconds")
    # Log dataset shapes
    log.info("Train/Test Split Complete: ")
    rprint(f"  • X_train: [yellow]{X_train.shape[0]} rows[/yellow], [yellow]{X_train.shape[1]} features[/yellow]")
    rprint(f"  • X_test: [yellow]{X_test.shape[0]} rows[/yellow], [yellow]{X_test.shape[1]} features[/yellow]")
    rprint(f"  • y_train: [yellow]{y_train.shape[0]} labels[/yellow] with distribution {dict(y_train.value_counts())}")
    rprint(f"  • y_test: [yellow]{y_test.shape[0]} labels[/yellow] with distribution {dict(y_test.value_counts())}")


    if debug:
        print(f"X_train shape:{X_train.shape}")
        print(f"X_test shape:{X_test.shape}")
        print(f"y_train shape:{y_train.shape}")
        print(f"y_test shape:{y_test.shape}")

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """Scales the X train and X_Test data to have a mean of 0 and standard deviation of 1.

    Args:
        X_train : The X training dataset.
        X_test: The X test dataset.

    Returns:
        The X_train and X_test data scaled.

    """
    log.info(f"Scaling data with train size={X_train.shape}, test size={X_test.shape}")
    start_time = time.time()
    # Scale features to ensure each has a mean of 0 and stdv of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    load_time = time.time() - start_time
    log.info(f"Data scaled successfully in {load_time:.2f} seconds")

    return X_train_scaled, X_test_scaled


def create_lr_model(max_iterv: int = 100) -> LogisticRegression:
    """Creates the logistic regression model.

    Args:
        max_iter : The max number of iterations before converging on the best solution.

    Returns:
        The logistic regression model.

    """
    log.info(f"Intializing logistic regression model with max_iter={max_iterv}")
    start_time = time.time()
    # Initialize logistic regression model
    lr_model = LogisticRegression(
        max_iter=max_iterv

    )  # default is 100, we were stopping at `100` iterations before converging on best solution
    load_time = time.time() - start_time
    log.info(f"LR model initialized successfully in {load_time:.2f} seconds")

    return lr_model


def fit_lr_model(lr_model: LogisticRegression, X_train, y_train):
    """Fits the logistic regression model based on the X_train and y_train data.

    Args:
        lr_model : The logisitic regression model.
        X_train : The X training dataset.
        y_train : The y training dataset.

    Returns:
        None

    """
    log.info(f"Fitting logistic regression model with {X_train.shape[0]} training samples")
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    load_time = time.time() - start_time
    log.info(f"Model fitted successfully in {load_time:.2f} seconds")
    log.info(f"Model coefficients: {lr_model.coef_}")
    log.info(f"Model intercept: {lr_model.intercept_}")
    log.info(f"Model score: {lr_model.score(X_train, y_train)}")

def save_trained_model(lr_model: LogisticRegression, file_path: Path):
    """Save the trained model to the /models directory.

    Args:
        lr_model : The logisitic regression model.
        file_name : The name of the output file.

    Returns:
        None.


    """
    log.info(f"Saving logistic regression model to: {file_path}")
    start_time = time.time()
    with open(file_path, "wb") as file:
        pickle.dump(lr_model, file)
    load_time = time.time() - start_time
    log.info(f"Model saved successfully in {load_time:.2f} seconds")

#@app.command()
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):



    """Main function to train a logistic regression model."""
    
    # Debug information
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    conf_dir = os.path.abspath(os.path.join(script_dir, "..", "conf"))
    
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config directory: {conf_dir}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Extract configuration values and resolve paths relative to the project root
        processed_data_path = os.path.join(project_root, cfg.data.processed_data_path.replace('../', ''))
        labels_path = os.path.join(project_root, cfg.data.labels_path.replace('../', ''))
        features_path = os.path.join(project_root, cfg.data.features_path.replace('../', ''))
        model_path = os.path.join(project_root, cfg.model.model_path.replace('../', ''))
        
        logger.info(f"Resolved data path: {processed_data_path}")
        logger.info(f"Resolved labels path: {labels_path}")
        logger.info(f"Resolved features path: {features_path}")
        logger.info(f"Resolved model path: {model_path}")
        
        # Model parameters
        max_iter = cfg.model.lr_params.max_iter
        test_size = cfg.train.test_size
        random_state = cfg.train.random_state
        debug = cfg.train.debug
        scale_data_flag = cfg.train.scale_data
        logger.info(f"Max_iter value: {max_iter}")
        logger.info(f"Loading data from: {processed_data_path}")
        data = load_data(processed_data_path)

        logger.info("Creating & Training logistic regression (LR) model...")
        X_train, X_test, y_train, y_test = create_test_train_split(
            data, test_size=test_size, random_state=random_state, debug=debug
        )
        
        if scale_data_flag:
            logger.info("Scaling data...")
            X_train, X_test = scale_data(X_train, X_test)
        
        lr_model = create_lr_model(max_iterv=max_iter)
        fit_lr_model(lr_model, X_train, y_train)

        # Save the Test_Data for testing in the predict section
        logger.info(f"Saving labels and features for LF model: {labels_path}, {features_path}")
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        # Convert arrays to DataFrames if needed
        if isinstance(y_test, pd.Series):
            y_test.to_csv(labels_path, index=False)
        else:
            pd.DataFrame(y_test, columns=['diagnosis']).to_csv(labels_path, index=False)

        if isinstance(X_test, pd.DataFrame):
            X_test.to_csv(features_path, index=False)
        else:
            # Convert numpy array back to DataFrame with original column names
            pd.DataFrame(X_test, columns=data.drop(["diagnosis", "id"], axis=1).columns).to_csv(features_path, index=False)

        logger.info(f"Saving LR model: {model_path}")
        save_trained_model(lr_model, model_path)

        # Metrics and plotting for cml.yml
        logger.info("Evaluating model and saving metrics and plot...")
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        with open("metrics.txt", "w") as f:
            f.write("### Model Evaluation\n")
            f.write(f"- Accuracy: {accuracy:.4f}\n")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig("plot.png")
        plt.close()




        logger.success("Model training complete.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    #app()
    main()
