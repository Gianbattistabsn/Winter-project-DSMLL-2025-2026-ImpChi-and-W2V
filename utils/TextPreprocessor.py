import re
import html
import numpy as np
import pandas as pd

class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    Extracts deep semantic meaning from HTML structure, URLs, and alt tags.
    
    Convert messy HTML/news data into clean numerical and text features
    that capture both structure (length, links, images) and semantic content (some specific keywords).
    """
    
    def __init__(self, categories_inv: dict):
        self.categories_inv = categories_inv
        
        
    @staticmethod
    def html_to_text(text: str) -> str:
        """
        Extract meaningful text from messy HTML.
        Preserve semantic hints (alt text, image titles) - these indicate what's important
        Extract clean URLs - often contain domain keywords (e.g., 'cnet.com' -> 'cnet' = tech)
        """
        if pd.isna(text) or text == "":
            return ""
        text = str(text)
        
        # Decoding text using html parser...
        text = html.unescape(text)
        text = text.replace('""', '"') 
        
        alt_matches = re.findall(r'(?:alt|title)=["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
        alt_content = " ".join(alt_matches)
        
        text = text.replace('"', ' ').replace("'", "")
        
        urls = re.findall(r'https?://[^\s<>]+', text)
        
        url_keywords = []
        
        for url in urls:
            clean = re.sub(r'^https?://(www\.)?', '', url)
            clean = re.sub(r'[^a-zA-Z0-9]', ' ', clean)
            tokens = clean.split()
            for t in tokens:
                t_low = t.lower()
                if len(t) > 1:
                    url_keywords.append(t_low)
        
        url_content = " ".join(url_keywords)
        
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://[^\s]+', ' ', text)
        full_text = f"{url_content} {alt_content} {text}"
        full_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip().lower()
        
        return full_text
    
    @staticmethod
    def find_foreign_currencies(text: str) -> int:
        """Count foreign currency symbols."""
        if pd.isna(text) or text == "":
            return 0
        text = str(text)
        pattern = r"[£€¥₹]"
        matches = re.findall(pattern, text)
        return len(matches)
    
    def fit(self, X: pd.DataFrame, y=None) -> 'TextPreprocessor':
        """
        Does not need to be fitted.
        """
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data and extract features from text
        """
        df['article_missing'] = (df['article'] =='\\N').astype(int)
        df['source_missing'] = (df['source'] =='\\N').astype(int)
        df['article_and_source_missing'] = ((df['source']=='\\N') & (df['article']=='\\N')).astype(int)
        

        # Handle missing values string
        df[['title', 'article']] = df[['title', 'article']].replace("\\N", "")
        df[['title', 'article']] = df[['title', 'article']].fillna("")
        
        # Length features
        df['title_len'] = df['title'].apply(len)
        df['article_len'] = df['article'].apply(len)
        df['len_ratio'] = (df['title_len'] / (df['article_len'] + 1))
        df['log_article_len'] = np.log1p(df['article_len'])
        df['title_word_count'] = df['title'].str.split().str.len()
        df['article_word_count'] = df['article'].str.split().str.len()
        df['num_numbers'] = df['article'].str.count(r'\d+').fillna(0).astype(int)
        # html trash count
        df['contains_adv'] = df['title'].str.contains('adv',False,False).fillna(0).astype(int)
        df['contains_world'] = df['title'].str.contains('world',False,False).fillna(0).astype(int)
        df['contains_full_story'] = df['article'].str.contains('full story',False,False).fillna(0).astype(int)
        df['contains_newsisfree'] = df['article'].str.contains('newsisfree',False,False).fillna(0).astype(int)
        df['contains_topstories'] = df['article'].str.contains('topstories',False,False).fillna(0).astype(int)
        df['contains_president'] =  df['article'].str.contains('president',False,False).fillna(0).astype(int)
        df['count_washington'] = df['article'].str.count('washington', flags=re.IGNORECASE).fillna(0).astype(int)
        df['has_pheedo'] = df['article'].str.contains('pheedo|feedburner', case=False).fillna(0).astype(int)
        df['num_html_trash'] = df['article'].str.count(r'\b(&quot;|href|http|click|read|&amp|<br>|breaking|urgent|alert|)\b').fillna(0).astype(int)
        df['is_clickbait'] = df['article'].str.count(r'(?i)\b(full story|details|click|cnn|read story)\b').fillna(0).astype(int)
        
        
        # HTML structure features - signal of media quality/presentation
        df['num_imgs'] = df['article'].str.count('<img ').fillna(0).astype(int)
        df['num_links'] = df['article'].str.count('href=').fillna(0).astype(int)
        
        # Currency features
        foreign_currencies_title = df['title'].apply(self.find_foreign_currencies)
        foreign_currencies_article = df['article'].apply(self.find_foreign_currencies)
        df['num_foreign_currencies'] = foreign_currencies_title + foreign_currencies_article
        df['title'] = df['title'].apply(self.html_to_text)
        df['article'] = df['article'].apply(self.html_to_text)
        
        df['start_with_city_reuters'] = df['article'].str.contains(r'^\s*\w+(?:\s+\w+)?\s+Reuters', case=False, regex=True)
        # Combine title and article - final text for embedding models (Word2Vec, TF-IDF)
        df['combined_text'] = (df['title'] + ' ' + df['article']).str.lower()
        return df
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)