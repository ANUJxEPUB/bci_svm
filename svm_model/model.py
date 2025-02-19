import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import time

def log_results_and_timings(cv_results, train_time, test_time, output_file='svm_results.xlsx'):
    """
    Logs cross-validation results, training time, and testing time to an Excel file.

    Args:
        cv_results (dict): Dictionary containing cross-validation results from GridSearchCV.
        train_time (float): Time taken for model training (in seconds).
        test_time (float): Time taken for model testing (in seconds).
        output_file (str, optional): Name of the Excel file to save the results (default: 'svm_results.xlsx').
    """

    writer = pd.ExcelWriter(output_file, engine='openpyxl')

    # Log cross-validation results
    # Accessing the parameters and results directly from cv_results
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_excel(writer, sheet_name='CV_Results', index=False)

    # Log training and testing times
    timings_df = pd.DataFrame({'Training Time (s)': [train_time], 'Testing Time (s)': [test_time]})
    timings_df.to_excel(writer, sheet_name='Timings', index=False)

    # Save the Excel file using the writer object, not the engine
    writer.close() # Use writer.close() to save and close the Excel file

# Dataset loading and preprocessing (assuming features in columns 0-34, target in column 35)
dataset = pd.read_excel('/Users/bittuxsun/Downloads/internship/work3/subject2.xlsx', engine='openpyxl')
df_feat = dataset.iloc[:, 0:35]
df_target = dataset.iloc[:, 36]

X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.10, random_state=101)

# Model training (with timing)
start_train_time = time.time()
model = SVC()
model.fit(X_train, y_train)
train_time = time.time() - start_train_time

# Assign the trained model to 'estimator'
estimator = model  # Now 'estimator' holds the fitted model

start = time.time()
y_tst_pred = estimator.predict(X_test)  # Use the fitted model to predict
end = time.time()
print(end-start)

# Model prediction (with timing)
start_test_time = time.time()
predictions = model.predict(X_test)
test_time = time.time() - start_test_time

print(classification_report(y_test, predictions))

# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

start_cv_time = time.time()  # Start time for cross-validation
grid.fit(X_train, y_train)
cv_time = time.time() - start_cv_time  # Time taken for cross-validation

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))

# Log results and timings to an Excel file
# Passing the entire cv_results_ dictionary
log_results_and_timings(grid.cv_results_, train_time, test_time, output_file='svm_results_with_timings91.xlsx')