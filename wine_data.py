import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
wine_data = pd.read_csv('wine_data.csv')

# Question 1
most_frequent_quality = wine_data['quality'].mode()[0]
highest_quality = wine_data['quality'].max()
lowest_quality = wine_data['quality'].min()

print("\nQuestion 1:")
print(f"Most Frequently Occurring Wine Quality: {most_frequent_quality}")
print(f"Highest Quality: {highest_quality}")
print(f"Lowest Quality: {lowest_quality}")

# Question 2
correlation_matrix = wine_data.corr()
fixed_acidity_correlation = correlation_matrix.loc['quality', 'fixed acidity']
alcohol_correlation = correlation_matrix.loc['quality', 'alcohol']
free_sulfur_dioxide_correlation = correlation_matrix.loc['quality', 'free sulfur dioxide']

print("\nQuestion 2:")
print(f"Correlation between Fixed Acidity and Quality: {fixed_acidity_correlation:.2f}")
print(f"Correlation between Alcohol and Quality: {alcohol_correlation:.2f}")
print(f"Correlation between Free Sulfur Dioxide and Quality: {free_sulfur_dioxide_correlation:.2f}")

# Question 3
best_quality_wine = wine_data[wine_data['quality'] == highest_quality]
lowest_quality_wine = wine_data[wine_data['quality'] == lowest_quality]

average_residual_sugar_best = best_quality_wine['residual sugar'].mean()
average_residual_sugar_lowest = lowest_quality_wine['residual sugar'].mean()

print("\nQuestion 3:")
print(f"Average Residual Sugar for Best Quality Wine: {average_residual_sugar_best:.2f} g/L")
print(f"Average Residual Sugar for Lowest Quality Wine: {average_residual_sugar_lowest:.2f} g/L")

# Question 4
plt.figure(figsize=(10, 6))
sns.scatterplot(x='volatile acidity', y='quality', data=wine_data)
plt.title('Volatile Acidity vs. Wine Quality')
plt.xlabel('Volatile Acidity (g/L)')
plt.ylabel('Quality')
plt.grid(True)
plt.show()

volatile_acidity_correlation = correlation_matrix.loc['quality', 'volatile acidity']
print("\nQuestion 4:")
print(f"Correlation between Volatile Acidity and Quality: {volatile_acidity_correlation:.2f}")

# Question 5
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_forest = random_forest_model.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)

print("\nQuestion 5:")
print(f"Decision Tree Model Accuracy: {accuracy_tree:.2f}")
print(f"Random Forest Model Accuracy: {accuracy_forest:.2f}")
