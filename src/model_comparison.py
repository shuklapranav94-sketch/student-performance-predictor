import os
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# 1. BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =========================================================
# 2. DATASET PATH
# =========================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student_performance.csv"
)


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset loaded successfully!"
)

print(
    "Dataset shape:",
    df.shape
)


# =========================================================
# 4. DISPLAY COLUMNS
# =========================================================

print("\nDataset columns:")

print(
    df.columns.tolist()
)


# =========================================================
# 5. TARGET COLUMN
# =========================================================

TARGET_COLUMN = "FinalGrade"


# =========================================================
# 6. CHECK TARGET
# =========================================================

if TARGET_COLUMN not in df.columns:

    print(
        f"\n❌ Target column '{TARGET_COLUMN}' "
        "was not found."
    )

    print(
        "Available columns:",
        df.columns.tolist()
    )

    raise SystemExit


# =========================================================
# 7. FEATURES AND TARGET
# =========================================================

X = df.drop(
    TARGET_COLUMN,
    axis=1
)

y = df[TARGET_COLUMN]


# =========================================================
# 8. HANDLE MISSING VALUES
# =========================================================

X = X.fillna(
    X.median(
        numeric_only=True
    )
)


# =========================================================
# 9. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# 10. DEFINE MODELS
# =========================================================

models = {

    "Logistic Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]),


    "Decision Tree":

        DecisionTreeClassifier(
            random_state=42
        ),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=200,

            random_state=42,

            n_jobs=-1
        ),


    "K-Nearest Neighbors":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ]),


    "Support Vector Machine":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                SVC(
                    probability=True,
                    random_state=42
                )
            )
        ])
}


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

results = []


print(
    "\n======================================"
)

print(
    "MODEL COMPARISON"
)

print(
    "======================================"
)


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )


    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    results.append({

        "Model":
            model_name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1 Score":
            f1
    })


    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )


# =========================================================
# 12. CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# 13. SORT BY F1 SCORE
# =========================================================

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)


# =========================================================
# 14. DISPLAY FINAL RESULTS
# =========================================================

print(
    "\n======================================"
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "======================================\n"
)


print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# 15. BEST MODEL
# =========================================================

best_model_name = results_df.iloc[0]["Model"]

best_accuracy = results_df.iloc[0]["Accuracy"]

best_precision = results_df.iloc[0]["Precision"]

best_recall = results_df.iloc[0]["Recall"]

best_f1 = results_df.iloc[0]["F1 Score"]


print(
    "\n======================================"
)

print(
    "BEST MODEL"
)

print(
    "======================================"
)


print(
    "Best Model:",
    best_model_name
)

print(
    f"Accuracy: {best_accuracy:.4f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall: {best_recall:.4f}"
)

print(
    f"F1 Score: {best_f1:.4f}"
)


# =========================================================
# 16. SAVE COMPARISON RESULTS
# =========================================================

RESULT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_comparison_results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


print(
    "\n✅ Model comparison results saved to:"
)

print(
    RESULT_PATH
)


print(
    "\n✅ Step 21 completed successfully!"
)