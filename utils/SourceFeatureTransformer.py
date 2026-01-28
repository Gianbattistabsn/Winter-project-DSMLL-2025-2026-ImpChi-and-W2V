import pandas as pd

class SourceFeatureTransformer:
    """
    Transformer to extract features from sources with fit/transform pattern.
    """
    
    def __init__(self, categories: dict):
        """
        Parameters:
        categories : dict
            Dictionary {category_name: label_value}
        """
        self.categories = categories
        self.stats = None
        self.totalNews_median = None
        
    def fit(self, X: pd.DataFrame, y=None) -> 'SourceFeatureTransformer':
        """
        Fit the transformer to compute source statistics.
        """

        if y is None:
            raise ValueError("y must be provided for this transformer.")
        
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True)
        df = pd.concat([X_reset, y_reset], axis=1)

        num_classes = len(set(self.categories.values()))
        
        # Counts per same source and label
        counts = df[['source', 'label']].value_counts().unstack(fill_value=0)
        # result is something like: source, label_0, label_1, ..., label_n 
        # total per source 'source':total
        total = df['source'].value_counts()
        # add 1 to each count and divide it by total + num_classes
        percentages = (counts + 1).div(total + num_classes, axis=0)
        # back to correct df format
        percentages = percentages.reset_index()
        # Rename label columns to include 'source percentage' prefix
        percentages = percentages.rename(columns={col: f'source percentage {col}' for col in percentages.columns if col != 'source'})
        
        # Total news count per source
        total_news_by_source = df['source'].value_counts().to_dict()
        percentages['totalNews'] = percentages['source'].map(total_news_by_source).astype(int)
        
        self.stats_ = percentages
        self.totalNews_median = percentages['totalNews'].median()
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply source features using pre-computed statistics.
    
        Parameters:
        df : pd.DataFrame
            DataFrame with 'source' column
            
        Returns:
        df : pd.DataFrame
            DataFrame with new features added
        """
        if self.stats_ is None:
            raise ValueError("Transformer must be fitted before transforming data. Call fit() first.")
        
        df = df.copy()
        df = df.merge(self.stats_, on='source', how='left').fillna(0)

        # Derived features using training median
        df['is_major_source'] = (df['totalNews'] > self.totalNews_median).astype(int)
        pct_cols = [col for col in df.columns if 'source percentage' in col and df[col].dtype in ['float64', 'int64']]
        df['max_category_pct'] = df[pct_cols].max(axis=1) if pct_cols else 0
        df['is_specialized_source'] = (df['max_category_pct'] > 0.5).astype(int)
        df['is_source_bcc'] = (df['source'] == 'BCC').fillna(0).astype(int)
        df['is_source_bbc'] = (df['source'] == 'BBC').fillna(0).astype(int)

        df['isCNN'] = df['source'].astype(str).str.contains('CNN', case=False).astype(int)
        df['isYahoo'] = df['source'].astype(str).str.contains('Yahoo', case=False).astype(int)
        df['containsNews'] = df['source'].str.contains('news', case=False, regex=False).astype(int)
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform in a single step.
        """
        return self.fit(X, y).transform(X)

