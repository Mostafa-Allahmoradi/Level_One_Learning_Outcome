
class DataCleaner:
    def __init__(self, df):
        self.df = df

    def handle_missing_values(self, strategy='mean', column=None):
        if strategy == 'mean':
            self.df.fillna(self.df[column].mean(), inplace=True)
        elif strategy == 'median':
            self.df.fillna(self.df[column].median(), inplace=True)
        elif strategy == 'mode':
            self.df.fillna(self.df[column].mode().iloc[0], inplace=True)
        else:
            raise ValueError("Strategy not recognized. Use 'mean', 'median', or 'mode'.")

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)

    def get_cleaned_data(self):
        return self.df