import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

class TextFeatureSelectorTransformer:
     """
     Applica TF-IDF e selezione feature (Chi2) in modo distinto per Titolo e Articolo.
     Permette di specificare K diversi per le due sorgenti.
     """
     
     def __init__(self, 
                    use_title: bool = True,
                    k_per_label_title: int = 50,
                    use_article: bool = True,
                    k_per_label_article: int = 200, 
                    min_df: int = 3,  
                    ngram_range: tuple = (1, 2)):
          
          self.use_title = use_title
          self.k_title = k_per_label_title
          self.use_article = use_article
          self.k_article = k_per_label_article
          self.min_df = min_df
          self.ngram_range = ngram_range
          

          self.tfidf_title_ = None
          self.indices_title_ = None 
          self.feats_title_ = None
          
          self.tfidf_article_ = None
          self.indices_article_ = None 
          self.feats_article_ = None
          
     def _select_features(self, X_text, y_arr, k, name="text"):
          """Helper method to fit tf-idf and selecting top chi2 features"""
          print(f"[{name}] Fitting TF-IDF (min_df={self.min_df})...")
          
          # inizializing Vectorizer
          tfidf = TfidfVectorizer(
               input='content', encoding='utf-8', lowercase=True,
               stop_words='english', 
               min_df=self.min_df, 
               ngram_range=self.ngram_range, 
          )
          
          try:
               X_tfidf = tfidf.fit_transform(X_text)
          except ValueError:
               print(f"[{name}] Warning: Vocabulary is empty.")
               return None, [], []
          
          print(f"[{name}] Selecting top {k} features per label via Chi2...")
          unique_classes = np.unique(y_arr)
          #selected_indices_set = set()
          feature_to_labels = {}

          for label in unique_classes:
               y_binary = (y_arr == label).astype(int)
               chi2_scores, _ = chi2(X_tfidf, y_binary)
               chi2_scores = np.nan_to_num(chi2_scores)
               
               n_features = X_tfidf.shape[1]
               k_safe = min(k, n_features)

               if (label == 2 and name == 'ARTICLE') or ():
                    k_safe = min(k * 3, n_features)
                    print(f"[{name}] increasing k to {k_safe}.")
               elif label == 0 and name == 'ARTICLE':
                    k_safe = min(k * 2, n_features)
                    print(f"[{name}] increasing k to {k_safe}.")
               elif label == 4 and name == 'ARTICLE':
                    k_safe = min(k * 2, n_features)
                    print(f"[{name}] increasing k to {k_safe}.")
               if k_safe > 0:
                    top_k_indices = np.argsort(chi2_scores)[-k_safe:]
                    for idx in top_k_indices:
                         if idx not in feature_to_labels:
                              feature_to_labels[idx] = set()
                         feature_to_labels[idx].add(label)

          selected_indices = sorted(list(feature_to_labels.keys()))
          
          if not selected_indices:
               print(f"[{name}] Warning: No features selected.")
               return tfidf, [], []
               
          raw_feature_names = tfidf.get_feature_names_out()
          custom_feature_names = []
          
          for idx in selected_indices:
               word = raw_feature_names[idx]
               # so that we can evaluate which label are correlated with
               labels_suffix = "_".join(sorted([str(l) for l in feature_to_labels[idx]]))
               custom_feature_names.append(f"{word}_{labels_suffix}")
               
          print(f"[{name}] Total unique features selected: {len(selected_indices)}")
          
          return tfidf, selected_indices, custom_feature_names

     def fit(self, X: pd.DataFrame, y) -> 'TextFeatureSelectorTransformer':
          if y is None:
               raise ValueError("label is required, insert y also.")
          
          y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
          df = X.copy()
          
          # TITLE FITTING
          if self.use_title:
               title_text = df['title'].fillna('').astype(str).replace(r'\\N', '', regex=True)
               self.tfidf_title_, self.indices_title_, self.feats_title_ = \
                    self._select_features(title_text, y_arr, self.k_title, name="TITLE")
                    
          # ARTICLE FITTING
          if self.use_article:
               article_text = df['article'].fillna('').astype(str).replace(r'\\N', '', regex=True)
               self.tfidf_article_, self.indices_article_, self.feats_article_ = \
                    self._select_features(article_text, y_arr, self.k_article, name="ARTICLE")
               
          return self
     
     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
          if (self.use_title and self.tfidf_title_ is None) and (self.use_article and self.tfidf_article_ is None):
               raise RuntimeError("Transformer not fitted.")

          df = X.copy()
          
          # Pulizia base e conversione stringa
          df['title'] = df['title'].fillna('').astype(str).replace(r'\\N', '', regex=True)
          df['article'] = df['article'].fillna('').astype(str).replace(r'\\N', '', regex=True)
          
          # Removing original columns
          dfs_to_concat = [df.drop(columns=['source', 'title', 'article', 'combined_text'], errors='ignore')]
          
          # TITLE
          if self.use_title and self.tfidf_title_ is not None and len(self.indices_title_) > 0:
               X_title_full = self.tfidf_title_.transform(df['title'])
               X_title_sel = X_title_full[:, self.indices_title_]
               
               cols = [f"title_{name}" for name in self.feats_title_]
               
               df_title = pd.DataFrame(
                    X_title_sel.toarray(),
                    columns=cols,
                    index=df.index
               )
               dfs_to_concat.append(df_title)

          # ARTICLE
          if self.use_article and self.tfidf_article_ is not None and len(self.indices_article_) > 0:
               X_art_full = self.tfidf_article_.transform(df['article'])
               X_art_sel = X_art_full[:, self.indices_article_]
               
               cols = [f"art_{name}" for name in self.feats_article_]

               df_art = pd.DataFrame(
                    X_art_sel.toarray(),
                    columns=cols,
                    index=df.index
               )
               dfs_to_concat.append(df_art)
               
          final_df = pd.concat(dfs_to_concat, axis=1)
          
          return final_df