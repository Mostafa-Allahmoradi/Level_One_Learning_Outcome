import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.categorical_features = None
        self.numerical_features = None

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(self.data_path)
    
    def feature_analysis(self):
        """Overal analyze of features in the dataset."""
        num_of_observations = self.data.shape[0]
        num_of_features = self.data.shape[1]
        num_of_categorical_features = self.data.select_dtypes(include=['object']).shape[1]
        num_of_numerical_features = self.data.select_dtypes(include=[np.number]).shape[1]

        self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        self.numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()

        print("\nDataset Overview:")
        print(f"\nThe data includes {num_of_observations} observations and {num_of_features} features.")

        print(f"\nOut of {num_of_features} features, {num_of_categorical_features} are categorical features and {num_of_numerical_features} are numerical features.")
        print(f"    Categorical features: {self.categorical_features}")
        print(f"    Numerical features: {self.numerical_features}")
    
    def missing_values_report(self):
        """Generate a report of missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        report = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
        return report[report['Missing Values'] > 0]

    def plot_distribution(self, column):
        """Plot the distribution of a specified column."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=80, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()