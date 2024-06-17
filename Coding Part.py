import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
heart_data_path = 'Heart_Attack_analysis.csv'
heart_data = pd.read_csv(heart_data_path)

# Data Cleaning and Preparation
# Convert the 'trtbps' column to numeric values and handle missing values
heart_data['trtbps'] = pd.to_numeric(heart_data['trtbps'], errors='coerce')
heart_data.dropna(inplace=True)  # Drop rows with missing values

# Set the style for the plots
sns.set(style='whitegrid')

# Function to plot a pie chart
def plot_pie_chart(data, column, labels, colors, title):
    counts = data[column].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True)
    plt.title(title)
    plt.axis('equal')
    plt.show()

# Function to plot a heatmap
def plot_heatmap(data, title):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(title)
    plt.show()

# Function to plot a line chart
def plot_line_chart(data, x_column, y_column, title, xlabel, ylabel, color='blue'):
    plt.figure(figsize=(12, 6))
    data_sorted = data.groupby(x_column)[y_column].mean().sort_index()
    data_sorted.plot(kind='line', marker='o', color=color, linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a statistical analysis (boxplot)
def plot_statistical_analysis(data):
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=data, orient="h", palette="Set2")
    plt.title('Statistical Analysis of All Features')
    plt.xlabel('Values')
    plt.show()

# Generate the required plots using the functions

# Pie Chart: Distribution of Chest Pain Types
cp_labels = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9']
plot_pie_chart(heart_data, 'cp', cp_labels, colors, 'Distribution of Chest Pain Types')

# Heatmap: Correlation Matrix
plot_heatmap(heart_data, 'Correlation Matrix')

# Line Chart: Average Maximum Heart Rate by Age
plot_line_chart(heart_data, 'age', 'thalachh', 'Average Maximum Heart Rate by Age', 'Age', 'Average Maximum Heart Rate (thalachh)', color='#88B04B')

# Statistical Analysis: Boxplot of all features
plot_statistical_analysis(heart_data)

# Statistical summary and correlation matrix
summary_stats = heart_data.describe()
correlation_matrix = heart_data.corr()

# Save the summary statistics and correlation matrix to CSV files
summary_stats.to_csv('summary_statistics.csv')
correlation_matrix.to_csv('correlation_matrix.csv')

# Display the summary statistics and correlation matrix
print("Summary Statistics:")
print(summary_stats)
print("\nCorrelation Matrix:")
print(correlation_matrix)
