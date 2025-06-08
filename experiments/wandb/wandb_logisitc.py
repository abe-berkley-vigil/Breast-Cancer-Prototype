import wandb
import sys
import os
from pathlib import Path
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from breast_cancer_classification.config import PROCESSED_DATA_DIR
from breast_cancer_classification.dataset import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 1. Start a new run
wandb.init(project="breast-cancer-classification", name="logisitc regression")

# 2. Initialize wandb config from sweep.yaml
config = wandb.config 

# 3. Load and preprocess data
data_path = PROCESSED_DATA_DIR / "dataset.csv"
df = load_data(data_path)
X = df.drop(["diagnosis", "id"], axis=1)
y = df["diagnosis"]
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# 4. Train logistic regression with sweepable parameters
model = LogisticRegression(
    C=config.C,
    solver=config.solver,
    max_iter=config.max_iter,
    random_state=21
)
model.fit(X_train, y_train)

# Compute loss on both train and test sets
train_loss = log_loss(y_train, model.predict_proba(X_train))
test_loss = log_loss(y_test, model.predict_proba(X_test))

# Compute accuracy on both sets
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

# 5. Make predictions and compute metrics
y_pred = model.predict(X_test)

metrics = {
    "train_loss": train_loss,
    "test_loss": test_loss,
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "f1_score": f1_score(y_test, y_pred),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred)
}
wandb.log(metrics)

# 6. Finish run
wandb.finish()
