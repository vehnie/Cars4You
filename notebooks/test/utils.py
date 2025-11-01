import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from rapidfuzz import process, fuzz
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from joblib import load
import os



def impute_numerical(df, num_cols):
    # store mean and std
    means = df[num_cols].mean()
    stds = df[num_cols].std()

    # normalize
    df_scaled = (df[num_cols] - means) / stds

    # impute on scaled data
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_imputed_scaled = pd.DataFrame(
        imputer.fit_transform(df_scaled),
        columns=num_cols,
        index=df.index
    )

    # inverse transform
    df[num_cols] = df_imputed_scaled * stds + means


    return df


def predict_missing_test(test_df, target_col, ignore_cols=['carID'], model_path="../assets/rfc_model.joblib"):
    """
    Usa o modelo treinado para imputar missing values no test_df.
    Retorna o DataFrame com target_col imputado.
    """

    # carregar modelo e colunas do treino
    rfc = load(model_path)
    X_train_cols = load(model_path.replace(".joblib","_cols.joblib"))

    predictors = test_df.columns.drop([target_col] + ignore_cols)
    missing_idx = test_df[test_df[target_col].isnull()].index

    if len(missing_idx) > 0:
        X_test = pd.get_dummies(test_df.loc[missing_idx, predictors], drop_first=True)
        X_test = X_test.reindex(columns=X_train_cols, fill_value=0)
        test_df.loc[missing_idx, target_col] = rfc.predict(X_test)
        print(f"Test: Imputed '{target_col}' with {len(missing_idx)} missing values")

    return test_df


# normalize strings: strip, lower
def clean_text(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower()


# standard values in model
def standardize_model(s):
    if pd.isna(s):
        return ''
    s = str(s).lower().strip()
    s = s.replace('-', ' ')
    s = ' '.join(s.split())
    return s

def safe_extract_one(query, choices, scorer):
    result = process.extractOne(query, choices, scorer=scorer)
    if result is None:
        return None, 0
    # Algumas versões retornam 2 valores, outras 3
    if len(result) == 2:
        match, score = result
    elif len(result) == 3:
        match, score, _ = result
    else:
        match, score = result[0], result[1]
    return match, score


def correct_brand(row, df_models):
    brand = row['brand_clean']
    model = row['model_fixed']

    # Filtra todas as marcas possíveis para este modelo
    matching_brands = df_models[df_models['model_clean'] == model]['brand_clean'].unique()

    if len(matching_brands) == 1:
        # Se existe apenas 1 marca para o modelo, força essa marca
        return matching_brands[0]
    elif len(matching_brands) > 1:
        # Se houver várias marcas (estranho, mas possível), escolhe a mais frequente
        return max(set(matching_brands), key=list(matching_brands).count)
    
    # Se o modelo não existe, fallback para original ou fuzzy
    if brand in df_models['brand_clean'].values:
        return brand

    all_brands = df_models['brand_clean'].unique()
    matched_brand, score = safe_extract_one(brand, all_brands, scorer=fuzz.token_sort_ratio)
    return matched_brand if score >= 40 else brand


# --- fix models ---
def correct_model(row, df_models):
    model = row['model_clean']
    brand = row['brand_clean']
    
    if ((df_models['brand_clean'] == brand) & (df_models['model_clean'] == model)).any():
        return model
    
    if brand in df_models['brand_clean'].values:
        possible_models = df_models[df_models['brand_clean']==brand]['model_clean'].tolist()
        matched_model, score = safe_extract_one(model, possible_models, scorer=fuzz.ratio)
        return matched_model if score >= 40 else model
    
    possible_models = df_models['model_clean'].tolist()
    matched_model, score = safe_extract_one(model, possible_models, scorer=fuzz.ratio)
    return matched_model if score >= 40 else model


def harmonize_brand(df):
    # --- etapa final: harmonizar modelos com marcas divergentes ---
    freqs = df.groupby(['model', 'brand']).size().reset_index(name='count')

    # para cada modelo, ver se há mais de uma marca associada
    for model, group in freqs.groupby('model'):
        if len(group) > 1:
            total = group['count'].sum()
            group = group.sort_values('count', ascending=False)
            major_brand, major_count = group.iloc[0]['brand'], group.iloc[0]['count']
            ratio_major = major_count / total

            # se maioria tiver mais de 60% das ocorrências → força
            if ratio_major >= 0.6:
                minor_brands = group.iloc[1:]['brand'].tolist()
                df.loc[(df['model'] == model) & (df['brand'].isin(minor_brands)), 'brand'] = major_brand
    return df


def fuzzy_match_column(series, canonical_values, score_cutoff=85, default='other'):
    """
    Fuzzy-matches each value in a pandas Series to a list of canonical values.
    Example:
        fuzzy_match_column(df['fuel_type_clean'], ['petrol', 'diesel', 'electric'])
    """
    def match_one(value):


        # exact match
        if value in canonical_values:
            return value

        # fuzzy match
        match = process.extractOne(value, canonical_values, scorer=fuzz.token_sort_ratio)
        if match is None:
            return "unknown"

        best_choice, score, _ = match
        return best_choice if score >= score_cutoff else default

    return series.apply(match_one)
    


def encode_test(df, cat_cols=['brand','fuel_type','transmission'], avg_price_path='../../data/avg_model_prices.csv', ohe_cols_path='../assets/train_ohe_cols.joblib'):
    """
    One-hot encode categorias no teste com base nas colunas do treino.
    Categorias novas = 0.
    """
    
    # One-hot encoding
    #for col in cat_cols:
        #if col in df.columns:
            #dummies = pd.get_dummies(df[col], prefix=col, drop_first=False).astype(int)
            #df = pd.concat([df, dummies], axis=1)
    
    # Merge avg_price
    if os.path.exists(avg_price_path):
        avg_price_lookup = pd.read_csv(avg_price_path)
        if {'brand','model','avg_price'}.issubset(avg_price_lookup.columns):
            df = df.merge(avg_price_lookup, on=['brand','model'], how='left')
            df['avg_price'] = df['avg_price'].fillna(0)
        else:
            df['avg_price'] = 0
    else:
        df['avg_price'] = 0
    
    # Drop originais
    drop_cols = [col for col in cat_cols + ['model'] if col in df.columns]
    df = df.drop(columns=drop_cols)
    
    # Reindex para colunas do treino
    train_cols = load(ohe_cols_path)
    df = df.reindex(columns=train_cols, fill_value=0)  # categorias novas = 0
    
    return df

