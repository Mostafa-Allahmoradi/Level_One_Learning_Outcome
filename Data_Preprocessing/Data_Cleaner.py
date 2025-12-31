from sklearn.impute import KNNImputer

class DataCleaner:
    def __init__(self, df):
        self.df = df
        
    def get_features_with_missing_values(self) -> list:
        """Identify features that contain at least one missing value."""
        columns_with_missing_values = self.df.columns[self.df.isnull().any()]
        return columns_with_missing_values.tolist()

    def handle_missing_values(self, strategy='mean'):
        if strategy == 'mean':
            self.df.fillna(self.df.mean(), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df.median(), inplace=True)
        elif strategy == 'mode':
            self.df.fillna(self.df.mode().iloc[0], inplace=True)
        elif strategy == 'drop_rows':
            self.df.dropna(inplace=True)
        elif strategy == 'knn': # Using KNN Imputer for one specific column
            imputer = KNNImputer(n_neighbors=5) # Neighbors can be adjusted but should be an odd number
            imputer.fit(self.df)
            imputer.transform(self.df)         
        else:
            raise ValueError("Strategy not recognized. Use 'mean', 'median', 'mode', 'drop_rows', or 'knn'.")

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def get_cleaned_data(self):
        return self.df