# Import Python Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
# Set the style for plots
plt.style.use('fivethirtyeight')

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load Data
data = pd.read_csv("E:\\Diabetes\\diabetes.csv")

# Display the first few rows of the data to ensure it's loaded correctly
print(data.head())
print("===================================")

# Explore The Data
# Check the info of the dataset, including column types and non-null counts
print(data.info())
print("===================================")

# Get statistical summary of the dataset
print(data.describe())
print("===================================")

# Check for duplicated rows
print("Number of duplicated rows:", data.duplicated().sum())
print("===================================")

# Analysis of The Data
# Display the correlation matrix
print("Correlation Matrix:\n", data.corr())
print("===================================")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', linewidths=0.1)
plt.title('Correlation Heatmap')
plt.show()

# Count Plot for 'Pregnancies'
plt.figure(figsize=(10, 6))
sns.countplot(x='Pregnancies', data=data)
plt.title('Count Plot for Pregnancies')
plt.show()

# Distribution Plot for 'Pregnancies'
plt.figure(figsize=(10, 6))
sns.histplot(data["Pregnancies"], kde=True)
plt.title('Distribution Plot for Pregnancies')
plt.show()

# Box Plot for 'Pregnancies'
plt.figure(figsize=(10, 6))
sns.boxplot(y=data['Pregnancies'])
plt.title('Box Plot for Pregnancies')
plt.show()

# Create Model
# Split the dataset into features (X) and target (y)
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target

rm = RandomOverSampler(random_state = 41)
x_res,y_res = rm.fit_resample(X, y)
print("Old Data set shape {}".format(Counter(y)))
print("===================================")
print("Old Data set shape {}".format(Counter(y_res)))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize the models
model_one = LogisticRegression()
model_two = SVC()
model_three = RandomForestClassifier(n_estimators = 100,class_weight = 'balanced')
model_four = GradientBoostingClassifier(n_estimators=1000)

# Lists to store results for different models
columns = ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'GradientBoostingClassifier']
result_one = []
result_two = []
result_three = []

# Function to train the model, make predictions, and display the results
def cal(model):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    pred = model.predict(X_test)
    
    # Calculate accuracy, recall, and F1 score
    accuracy = accuracy_score(pred, y_test)
    recall = recall_score(pred, y_test)
    f1 = f1_score(pred, y_test)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(pred, y_test)
    
    # Store results
    result_one.append(accuracy)
    result_two.append(recall)
    result_three.append(f1)
    
    # Display the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()
    
    # Print model evaluation metrics
    print(model)
    print("===================================")
    print(f"Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("===================================")

# Evaluate each model using the cal function
cal(model_one)
cal(model_two)
cal(model_three)
cal(model_four)

# Create a DataFrame to display results
data_frame = pd.DataFrame({
    "Algorithm": columns,
    "Accuracy": result_one,
    "Recall": result_two,
    "F1_Score": result_three
})

# Print the DataFrame with the results
print(data_frame)

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_frame.Algorithm, result_one, label="Accuracy", marker='o')
ax.plot(data_frame.Algorithm, result_two, label="Recall", marker='o')
ax.plot(data_frame.Algorithm, result_three, label="F1_Score", marker='o')
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Score")
ax.legend()
plt.show()