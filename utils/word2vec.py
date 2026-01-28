import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin
import os 
os.environ['PYTHONHASHSEED'] = '42'

class TextFeatureWord2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Word2Vec. 
    Now includes logic to SAVE and LOAD models to avoid re-training every time.
    This fixes the non-determinism issue with multithreading!
    """
    
    def __init__(self, 
                 name: str = "w2v_model",
                 vector_size: int = 75,
                 window: int = 10,
                 min_count: int = 5,
                 workers: int = 1, # Kept 1 for reproducibility
                 epochs: int = 15,
                 sg: int = 1,
                 apply_to_combined: bool = True, 
                 eval_text: pd.Series = None):
        
        self.name = name 
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.apply_to_combined = apply_to_combined
        self.eval_text = eval_text
        
        self.model_ = None
        self.vocab_ = None
        
    def _tokenize(self, text_series):
        # Simple tokenizer helper
        return [simple_preprocess(text) for text in text_series.fillna('').astype(str)]

    def fit(self, X: pd.DataFrame, y=None) -> 'TextFeatureWord2VecTransformer':
        # Create a folder for models if it doesn't exist
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{self.name}.model"

        # If the model is already saved, load it! 
        # This saves time and ensures we use the EXACT same vectors as before.
        if os.path.exists(model_path):
            print(f"Found saved model at '{model_path}'. Loading it to ensure reproducibility...")
            self.model_ = Word2Vec.load(model_path)
            self.vocab_ = set(self.model_.wv.index_to_key)
            print("Model loaded successfully. Skipping training.")
            return self


        # If not found, train from scratch (this might take a while)
        print(f"Model not found. Training Word2Vec from scratch (size={self.vector_size}, sg={self.sg})...")
        
        df = X.copy()
        # Combine title and article for better context
        train_text = X['title'].replace('\\N', np.nan).fillna('') + ' ' + df['article'].replace('\\N', np.nan).fillna('')
        
        if self.eval_text is not None:
            # Using all available text (train + eval) to improve vocabulary coverage
            full_corpus_text = pd.concat([train_text, self.eval_text])
        else:
            full_corpus_text = train_text

        sentences = self._tokenize(full_corpus_text)

        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers, # Using 1 worker makes it slower but DETERMINISTIC
            sg=self.sg,
            epochs=self.epochs,
            seed=42
        )
        
        # Save the model immediately so we don't have to retrain next time
        self.model_.save(model_path)
        self.vocab_ = set(self.model_.wv.index_to_key)
        print(f"Training done. Saved model to '{model_path}' for future use.")
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit first.")
            
        df_input = df.copy()
        train_text = df_input['title'].replace('\\N', np.nan).fillna('') + ' ' + df_input['article'].replace('\\N', np.nan).fillna('')
        tokenized_docs = self._tokenize(train_text)
        
        doc_vectors = []
        for tokens in tokenized_docs:
            # Filters out words that are not in the vocabulary
            valid_tokens = [word for word in tokens if word in self.vocab_]
            
            if valid_tokens:
                # Averages the vectors of the words in the document
                vectors = self.model_.wv[valid_tokens]
                doc_mean = np.mean(vectors, axis=0)
            else:
                # If no words are known, use a zero vector
                doc_mean = np.zeros(self.vector_size)
            doc_vectors.append(doc_mean)
        
        X_w2v = np.array(doc_vectors)
        w2v_cols = [f'w2v_{i}' for i in range(self.vector_size)]
        
        # Return original dataframe + new vector columns
        w2v_df = pd.DataFrame(X_w2v, columns=w2v_cols, index=df_input.index)
        result_df = pd.concat([df_input, w2v_df], axis=1)
        
        return result_df