import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_selection import chi2

class NewsAgencySmartSelector:
     def __init__(self, top_k: int = 30, min_count: int = 10):
          """
          top_k: number of agencies to keep (most statistically relevant).
          min_count: minimum appearances to be considered in the test.
          """
          self.top_k = top_k
          self.min_count = min_count
          self.selected_agencies_ = [] 
          self.agency_scores_ = {}
          
     def _extract_candidates(self, df: pd.DataFrame) -> pd.Series:
          combined = (df['title'].fillna('').astype(str).str.lower() + ' ' + 
                    df['article'].fillna('').astype(str).str.lower())
          combined = combined.str.replace(r'\s+', ' ', regex=True)
          return combined.str.findall(r"\(([^)]+)\)")

     def fit(self, X: pd.DataFrame, y=None):
          if y is None: 
               raise ValueError("y is needed")          
          candidates_series = self._extract_candidates(X)
          
          all_tokens = []
          for lst in candidates_series:
               if isinstance(lst, list):
                    all_tokens.extend([s.strip() for s in lst if 2 < len(s.strip()) < 40])
          
          counts = Counter(all_tokens)
          
          valid_candidates_list = sorted([ag for ag, count in counts.items() if count >= self.min_count])
          
          
          if len(valid_candidates_list) == 0:
               print("No agency candidates meet the minimum count requirement.")
               return self

          vocab_map = {agency: i for i, agency in enumerate(valid_candidates_list)}
          n_samples = len(X)
          n_features = len(valid_candidates_list)
          
          X_binary = np.zeros((n_samples, n_features), dtype=int)
          

          for row_idx, candidates in enumerate(candidates_series):
               if isinstance(candidates, list):
                    for cand in candidates:
                         cand_clean = cand.strip()
                         if cand_clean in vocab_map:
                              col_idx = vocab_map[cand_clean]
                              X_binary[row_idx, col_idx] = 1
          
          
          chi2_scores, _ = chi2(X_binary, y)
          

          scores_df = pd.DataFrame({
               'agency': valid_candidates_list,
               'score': chi2_scores,
               'freq': [counts[ag] for ag in valid_candidates_list]
          })
          
          scores_df = scores_df.sort_values(by='score', ascending=False)
          
          self.selected_agencies_ = scores_df.head(self.top_k)['agency'].tolist()
          

          self.agency_scores_ = dict(zip(scores_df['agency'], scores_df['score']))    
          return self

     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
          if not self.selected_agencies_:
               return df
          
          df = df.copy()
          candidates_series = self._extract_candidates(df)
          
          candidates_sets = candidates_series.apply(
               lambda x: set(s.strip() for s in x) if isinstance(x, list) else set()
          )
          
          for agency in self.selected_agencies_:
               safe_name = re.sub(r'[^a-zA-Z0-9]', '_', agency)
               col_name = f'agency_{safe_name}'
               
               df[col_name] = candidates_sets.apply(lambda s: 1 if agency in s else 0)
               
          new_cols = [c for c in df.columns if c.startswith('agency_')]
          df['total_top_agencies'] = df[new_cols].sum(axis=1)
          
          has_any_parenthesis = candidates_series.str.len() > 0
          df['has_minor_agency'] = (has_any_parenthesis & (df['total_top_agencies'] ==0 )).astype(int)
               
          return df
     
     def fit_transform(self, X, y=None):
          return self.fit(X, y).transform(X)
     

