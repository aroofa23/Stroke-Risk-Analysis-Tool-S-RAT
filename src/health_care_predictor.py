import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('src/healthcare-dataset-stroke-data.csv')
print(data.head())
print(data.info())

# Drop unnecessary columns
data = data.drop(['id'], axis=1)
data = data[data['gender'] != 'Other']

# Check missing values percentage and fill them
print((data.isna().sum() / len(data)) * 100)
mean_bmi = data['bmi'].mean()
data['bmi'] = data['bmi'].fillna(mean_bmi)
print((data.isna().sum() / len(data)) * 100)

# Differentiate between categorical and numerical values
numeric_columns = ['age', 'bmi', 'avg_glucose_level']
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']

# Transform categorical values to numerical labels
columns_temp = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']
gender_le = LabelEncoder()
ever_married_le = LabelEncoder()
work_type_le = LabelEncoder()
smoking_status_le = LabelEncoder()
residence_type_le = LabelEncoder()

# Separate features and target variable
X_temp = data.drop(columns='stroke')
y = data.stroke

# Resample the data to address class imbalance
ros = RandomOverSampler(sampling_strategy='minority')
X, y_resampled = ros.fit_resample(X_temp, y)

# Concatenate resampled features and target variable
resampled_data = pd.concat([pd.DataFrame(X, columns=X_temp.columns), pd.Series(y_resampled, name='stroke')], axis=1)
resampled_data.to_csv('src/resampled_data.csv', index=False)

# Encode categorical variables
X['gender_le'] = gender_le.fit_transform(X['gender'])
X['ever_married_le'] = ever_married_le.fit_transform(X['ever_married'])
X['work_type_le'] = work_type_le.fit_transform(X['work_type'])
X['smoking_status_le'] = smoking_status_le.fit_transform(X['smoking_status'])
X['residence_type_le'] = residence_type_le.fit_transform(X['Residence_type'])

# Drop original categorical variables
X = X.drop(['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type'], axis=1)

# Display encoded features
print(X.head())

# Save the preprocessed data to a temporary CSV file
#Dealing with imbalanced data
scaler_dump = MinMaxScaler().fit(X)
scaler = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(scaler, columns=X.columns)
X = X.astype(np.float64)


# Define classifiers and their hyperparameters for grid search
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
}

#------------------------Testing different random seeds for tain-test slpit---------------------------------------
# Define a list of random state values to try
random_states = [42, 100, 200, 300, 400, 500]

# Create an empty list to store the accuracy scores
accuracy_scores = []

# Iterate over the random state values
for random_state in random_states:
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, random_state=random_state, test_size=0.25)

    # Initialize best classifier and score
    bClassifier = None
    bScore = 0.0

    # Iterate over classifiers and perform grid search for hyperparameter tuning
    for name, clf in classifiers.items():
        grid_search = GridSearchCV(clf, param_grid[name], cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        if score > bScore:
            bScore = score
            bClassifier = best_clf

    # Append the best accuracy score to the list
    accuracy_scores.append(bScore)

    # Print the best classifier and accuracy score for this random state
    print(f"Random State: {random_state}")
    print(f"Best classifier: {bClassifier}\nBest Accuracy score: {bScore}")
    print("-" * 50)

# Print the list of accuracy scores
print("Accuracy scores for different random states:")
for i, score in enumerate(accuracy_scores):
    print(f"Random State: {random_states[i]}, Accuracy Score: {score}")

accs = np.array(accuracy_scores)


#-----------------------------------Final Model----------------------------------------------------

# resampled_data = pd.concat([pd.DataFrame(X, columns=X_temp.columns), pd.Series(y_resampled, name='target')], axis=1)
# resampled_data.to_csv('src/resampled_data.csv', index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, random_state=42, test_size=0.25)


# Initialize best classifier and score
best_classifier = None
best_score = 0.0

# Iterate over classifiers and perform grid search for hyperparameter tuning
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid[name], cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score > best_score:
        best_score = score
        best_classifier = best_clf





#Learning the roc curve for the obtained model
from sklearn.metrics import roc_curve
y_preds = best_classifier.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_preds)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(fpr, tpr, color='blue', label='ROC curve')
ax1.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')
ax1.legend(loc='lower right')

# Plotting Confusion Matrix
mythrehold = 0.5
from sklearn.metrics import confusion_matrix
y_pred = (best_classifier.predict(X_test) >= mythrehold).astype('int')
cm = confusion_matrix(y_test, y_pred)
ax2.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()

# Save the plot to an image file (e.g., PNG)
plt.savefig('figs/roc_curve.png')

# Save the best classifier and preprocessing objects
dump(best_classifier, 'src/best_classifier.joblib')
dump(scaler_dump, 'src/scaler.joblib')
dump(gender_le, 'src/le_gender.joblib')
dump(ever_married_le, 'src/le_ever_married.joblib')
dump(work_type_le, 'src/le_work_type.joblib')
dump(smoking_status_le, 'src/le_smoking_status.joblib')
dump(residence_type_le, 'src/le_residence_type.joblib')

print(f"Best classifier: {best_classifier}\nBest Accuracy score: {best_score}")

# Assuming your best classifier is a tree-based model
if isinstance(best_classifier, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
    # Get the feature importances
    feature_importances = best_classifier.feature_importances_
    
    # Get the feature names
    feature_names = X_train.columns
    
    # Create a DataFrame to store the feature importance scores
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    
    # Sort the DataFrame in descending order of importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    
    # Adjust the layout to avoid overlapping x-axis labels
    plt.tight_layout()
    
    # Save the plot to an image file (e.g., PNG)
    plt.savefig('figs/feature_importance.png')

else:
    print("The best classifier is not a tree-based model. Feature importance cannot be computed.")


#Plotting Numerical Features for Train-Test Sets
# Get the numerical feature names
numeric_features = ['age', 'bmi', 'avg_glucose_level']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_features), figsize=(15, 5))

# Iterate over the numerical features
for i, feature in enumerate(numeric_features):
    # Get the feature values for train and test sets
    feature_train = X_train[feature]
    feature_test = X_test[feature]
    
    # Plot the feature distributions
    axes[i].hist(feature_train, bins=20, alpha=0.5, label='Train Set', color='b')
    axes[i].hist(feature_test, bins=20, alpha=0.5, label='Test Set', color='g')
    
    # Set the title and axis labels
    axes[i].set_title(f'{feature.capitalize()} Distribution', fontsize=14)
    axes[i].set_xlabel(feature.capitalize(), fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    
    # Add a legend
    axes[i].legend(fontsize=10)

# Adjust the layout and display the plot
plt.tight_layout()

 # Save the plot to an image file (e.g., PNG)
plt.savefig('figs/train_test_set.png')


