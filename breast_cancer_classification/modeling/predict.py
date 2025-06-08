from datetime import datetime
import logging
import os.path
from pathlib import Path
import pickle
import sys
import time

from loguru import logger
import numpy as np
import pandas as pd
from rich import print as rprint

# Rich imports for enhanced logging
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
import typer

## Required otherwise it will error out in the makefile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))

from breast_cancer_classification.config import MODELS_DIR, PROCESSED_DATA_DIR

# Set up Rich console and logging
console = Console()
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"predict_{timestamp}.log"

# Create separate formatters for console and file
console_formatter = logging.Formatter("%(message)s", datefmt="[%X]")
file_formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create handlers with their respective formatters
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(file_formatter)

# Configure logging with both handlers
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
log = logging.getLogger("predict")

# Print the log file path so user knows where logs are being saved
log.info(f"Logs are being saved to: {log_file.absolute()}")
app = typer.Typer()


def evaluate_lr_model(lr_model: LogisticRegression, X_test, y_test):
    """Evalutes the accuracy of logistic regression model and prints the results.

    Args:
        lr_model : The logisitic regression model.
        X_train : The X training dataset.
        y_train : The y training dataset.

    Returns:
        The y_predictions, accuracy, confusion matrix, and the classification report.


    """
    """Evaluates the accuracy of logistic regression model and prints the results."""
    log.info("Evaluating model performance...")
    start_time = time.time()
    # Make predictions with progress tracking
    log.info("Making predictions on test data")

    # Make predictions
    y_pred = lr_model.predict(X_test)  # maybe split predictions out
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report_str = classification_report(y_test, y_pred)
    # Log the time taken for evaluation
    eval_time = time.time() - start_time
    log.info(f"Model evaluation completed in {eval_time:.2f} seconds")
    # Create rich table for confusion matrix
    cm_table = Table(title="Confusion Matrix")
    cm_table.add_column("", style="cyan")
    cm_table.add_column("Predicted Negative", style="magenta")
    cm_table.add_column("Predicted Positive", style="magenta")
    
    cm_table.add_row("Actual Negative", str(conf_matrix[0][0]), str(conf_matrix[0][1]))
    cm_table.add_row("Actual Positive", str(conf_matrix[1][0]), str(conf_matrix[1][1]))
    
    # Log accuracy with Rich formatting
    rprint(Panel(f"[bold green]Model Accuracy: {accuracy:.4f}[/bold green]", 
                title="Evaluation Results", 
                border_style="green"))
    
    # Display tables and raw classification report
    console.print(cm_table)
    console.print(Panel(class_report_str, title="Classification Report", border_style="blue"))
    # Print evaluation metrics
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"Classification Report:\n {class_report}")

    return y_pred, accuracy, conf_matrix, class_report


def generate_feature_importance(
    lr_model: LogisticRegression, original_df: pd.DataFrame
) -> pd.DataFrame:
    """Generates the feature importance report and outputs it.

    Args:
        lr_model : The logisitic regression model.
        original_df: The original dataframe with all the columns.

    Returns:
        The important features dataframe.


    """
    log.info("Generating feature importance analysis...")
    start_time = time.time()
    # Extract feature importance
    # Extract feature names from the dataset
    feature_names = original_df.columns
    # Get the coefficients
    coefficients = lr_model.coef_[0]  # shape (n_features,)

    # Pair each coefficient with its feature name
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),  # for easier sorting by strength
        }
    )

    # Sort features by importance (absolute value of coefficient)
    feature_importance = feature_importance.sort_values(by="abs_coefficient", ascending=False)
    # Create rich table for feature importance
    importance_table = Table(title="Feature Importance")
    importance_table.add_column("Rank", style="cyan", justify="right")
    importance_table.add_column("Feature", style="green")
    importance_table.add_column("Coefficient", style="magenta", justify="right")
    importance_table.add_column("Abs Coefficient", style="yellow", justify="right")  

    # Add top 10 features to the table
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        importance_table.add_row(
            f"{i+1}",
            row["feature"],
            f"{row['coefficient']:.6f}",
            f"{row['abs_coefficient']:.6f}"
        )
    
    console.print(importance_table)
    # Log the time taken for feature importance generation
    gen_time = time.time() - start_time
    log.info(f"Feature importance analysis completed in {gen_time:.2f} seconds")
    # Log summary of feature importance
    top_feature = feature_importance.iloc[0]["feature"]
    top_coefficient = feature_importance.iloc[0]["abs_coefficient"]
    log.info(f"Most important feature: {top_feature} (coefficient magnitude: {top_coefficient:.6f})")      

    print("Feature Importance Data:")
    print(feature_importance)

    return feature_importance


def load_lr_model(filepath: Path):
    """Load an already existing model from a pickle file.

    Args:
        file_path : The location of the logistic regression pickle file.

    Returns:
        Loaded model from file.


    """
    # Load the model
    log.info(f"Loading model from: {filepath}")
    
    start_time = time.time()
    
    try:
        with open(filepath, "rb") as file:
            lr_model = pickle.load(file)
        
        load_time = time.time() - start_time
        log.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model details
        model_info = {
            "Type": type(lr_model).__name__,
            "Max Iterations": lr_model.max_iter,
            "Solver": lr_model.solver,
            "Penalty": lr_model.penalty if hasattr(lr_model, 'penalty') else "None",
            "Classes": len(lr_model.classes_)
        }
        
        # Create a table for model details
        model_table = Table(title="Model Information")
        model_table.add_column("Parameter", style="cyan")
        model_table.add_column("Value", style="green")
        
        for param, value in model_info.items():
            model_table.add_row(param, str(value))
            
        console.print(model_table)
        
        return lr_model
    
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    processed_data_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    logger.info("Performing inference for model...")
    lr_model = load_lr_model(model_path)
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(labels_path)

    data = pd.read_csv(processed_data_path)
    X = data.drop(["diagnosis", "id"], axis=1)

    evaluate_lr_model(lr_model, X_test, y_test)
    generate_feature_importance(lr_model, X)

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
