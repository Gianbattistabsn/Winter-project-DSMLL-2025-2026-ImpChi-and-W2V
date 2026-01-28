#public modules
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FunctionTransformer

#private modules
from utils.NewsAgencyFeatureTransformer import NewsAgencySmartSelector
from utils.SourceFeatureTransformer import SourceFeatureTransformer
from utils.TextFeatureSelectorTopK import TextFeatureSelectorTransformer
from utils.TextPreprocessor import TextPreprocessor 
from utils.word2vec import TextFeatureWord2VecTransformer
INCLUDE_TIMESTAMP = True
#seed settings
np.random.seed(42)
# model params
best_params = {
    'objective': 'multiclass',
    'num_class': 7,
    'class_weight':'balanced',
    'metric': 'multi_logloss',
    'n_estimators':600, 
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'verbosity': -1,
    'random_state': 42,
    'learning_rate': 0.02,
    'num_leaves': 217, 
    'max_depth': 14, 
    'min_child_samples': 67, 
    'reg_alpha': 0.15, 
    'reg_lambda': 4, 
    'min_split_gain': 0.08,
    'colsample_bytree': 0.4,
    'subsample': 0.75,
    'subsample_freq': 7
}

categories = {
  'International News':0,
  'Business': 1,
  'Technology': 2,
  'Entertainment': 3,
  'Sports': 4,
  'General News': 5,
  'Health': 6
  }

categories_inv = {
    0:'International News',
    1:'Business',
    2:'Technology',
    3:'Entertainment',
    4:'Sports',
    5:'General News',
    6:'Health'
}

def process_timestamp(X: pd.DataFrame, y=None):
    if INCLUDE_TIMESTAMP:
        df = process_timestamp_utility(X, y)
    else:
        df = X.drop(columns='timestamp')
    return df


def process_timestamp_utility(X: pd.DataFrame, y=None) -> pd.DataFrame:
    df = X.copy()

    df['timestamp'] = df['timestamp'].replace('0000-00-00 00:00:00', np.nan)
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp_missing'] = df['timestamp_dt'].isna().astype(int)


    df['year'] = df['timestamp_dt'].dt.year
    df['month'] = df['timestamp_dt'].dt.month
    df['day'] = df['timestamp_dt'].dt.day
    df['hour'] = df['timestamp_dt'].dt.hour
    df['weekday'] = df['timestamp_dt'].dt.dayofweek # range is (0,6)
    df['isWeekend'] = df['timestamp_dt'].dt.dayofweek.isin([5, 6]).astype(float)

    #Imputing
    df['year'] = df['year'].fillna(2007).astype(int) 
    df['month'] = df['month'].fillna(1).astype(int)
    df['hour'] = df['hour'].fillna(0).astype(int)
    df['weekday'] = df['weekday'].fillna(0).astype(int)
    df['isWeekend'] = df['isWeekend'].fillna(0).astype(int)

    df['years_since_2004'] = df['year'] - 2004
    df['is_old_article'] = (df['year'] <= 2005).astype(int)
    df['is_recent_article'] = (df['year'] >= 2007).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['weekday_sin'] = np.sin(2 * np.pi * (df['weekday']) / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * (df['weekday']) / 7)
    

    cols_to_drop = [
        'timestamp', 'timestamp_dt', 'hour', 'weekday']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df


def main():
    df = pd.read_csv('./data/development.csv')

    print(f'Original N rows: {df.shape[0]}')

    df = df.drop_duplicates(subset=['title', 'article', 'source', 'label'], keep='first')
    print(f'N rows after removing duplicates: {df.shape[0]}')


    df = df.drop_duplicates(subset=['source','title', 'article'], keep=False)
    print(f'N rows after removing ambiguity {df.shape[0]}')

    X = df.drop(columns=['label', 'Id'])
    y = df['label']

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)
    print("Split completed")


    dfEval = pd.read_csv('./data/evaluation.csv', na_values=['\\N']).fillna('')


    X_train_val_test =  pd.concat([X_train, X_val, X_test])
    # seed has already been set inside.
    w2v_train_val_test = TextFeatureWord2VecTransformer('train_val_test_50_10', 50, 10, 10, 1, 10, 1, True, dfEval['title'] + ' ' + dfEval['article'])

    # no need to seed
    source_transformer = SourceFeatureTransformer(categories) # extract 7 source percentages
    text_preprocessor = TextPreprocessor(categories_inv) # clean text and extract some faetures
    timestamp_transformer = FunctionTransformer(process_timestamp) # timestamp transformer
    newsAgency_transformer = NewsAgencySmartSelector(top_k = 40, min_count=5) # onehot + chi2 for top news agency
    # apply tfidf with impChi
    selectKbestss = TextFeatureSelectorTransformer(use_title=True, k_per_label_title=30, use_article=True, k_per_label_article=80, min_df=5, ngram_range=(1,2))

    preprocessing_pipeline_train_val_test = Pipeline([
        ('timestamp_transformer', timestamp_transformer),
        ('newsAgency_transformer', newsAgency_transformer),
        ('w2v_transformer', w2v_train_val_test),
        ('text_preprocessor', text_preprocessor),
        ('source_transformer', source_transformer),
        ('tfidftransf', selectKbestss)
    ])

    # preprocessing for training on full dataset
    y_train_val_test =  pd.concat([y_train, y_val, y_test])
    X_train_val_test_pp = preprocessing_pipeline_train_val_test.fit_transform(X_train_val_test, y_train_val_test)

    print('Shapes after preprocessing:')
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_dev shape: {X_train_val_test_pp.shape}")

    # training on full development set and prediction by X_eval of y_eval
    print("\nTraining model on X_train_val_test...")
    model_train_test_val = LGBMClassifier(**best_params)
    model_train_test_val = model_train_test_val.fit(X_train_val_test_pp, y_train_val_test)

    df_eval = pd.read_csv('./data/evaluation.csv')
    df_eval = df_eval.drop(columns='Id')
    df_eval = preprocessing_pipeline_train_val_test.transform(df_eval)
    y_pred_eval = model_train_test_val.predict(df_eval)

    submission = pd.DataFrame({
    'Id': np.arange(y_pred_eval.shape[0]),
    'Predicted': y_pred_eval
    })
    submission.to_csv('submission.csv', index=False)
    print('Submission is done')
    return 0

main()