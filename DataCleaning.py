
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)


def trim_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    First, drop the columns with not_reported values > 10%
    Then, drop observations with not_reported or other values
    return cleaned dataframe
    """
    # Drop the columns with not_reported values > 10%
    drop = []
    demo = ['age', 'gender', 'race', 'ethnicity', 'lang',
            'employStatus', 'maritalStatus', 'chief_complaint']
    cols = [i for i in df.columns if i not in demo]
    for col in cols:
        ratio = df[col][df[col] == 'not_reported'].count()/df.shape[0]*100
        if ratio > 0.1:
            drop.append(col)
    df = df.drop(labels=drop, axis=1)

    # Drop observations with not_reported or other values
    df= df[~df.apply(lambda row: row =='not_reported').any(axis=1)]
    df= df[~df.apply(lambda row: row =='other').any(axis=1)]
    df= df[~df.apply(lambda row: row =='4+').any(axis=1)]

    # Convert numeric features to float
    num = ['ua_ph', 'ua_spec_grav', 'age']
    for col in num:
        mean = df[(df[col] != 'not_reported') & (df[col]!= 'other')][col].astype(
            'float').mean()
        df[col] = df[col].replace('not_reported', mean)
        df[col] = df[col].astype(float)

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Input the cleaned dataframe,
    OneHotEncode the categorical (non-ordinal) attributes,
    OrdinalEncode the ordinal attributes
    return the final dataframe
    """

    other = ['ua_ph', 'ua_spec_grav', 'age']
    ord = ['ua_blood', 'ua_glucose', 'ua_ketones', 'ua_leuk', 'ua_protein']
    onehot = ['chief_complaint', 'race', 'ethnicity',
              'maritalStatus', 'employStatus']
    label = [i for i in df.columns if i not in ord+other+onehot]

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), onehot),
            ('label', OrdinalEncoder(), label),
            ('ordinal', OrdinalEncoder(categories=[
             ['negative', 'small', 'moderate', 'large']]* len(ord)), ord)
        ])

    transformed = preprocessor.fit_transform(df)

    onehot_col_names = preprocessor.named_transformers_[
        'onehot'].get_feature_names_out(onehot)
    new_column_names = list(onehot_col_names) + label + ord
    # Preserve the original index
    df_transformed = pd.DataFrame(
        transformed, columns=new_column_names, index=df.index)  # type: ignore

    df_final = pd.concat([df[other], df_transformed], axis=1)

    return df_final, preprocessor


def process_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X_train, X_test, y_train, y_test
    """
    data_filename = os.path.join(os.getcwd(), 'data', 'S1File.csv')
    metadata_filename = os.path.join(os.getcwd(), 'data', 'metadata.csv')

    df = pd.read_csv(data_filename)
    metadata = pd.read_csv(metadata_filename)

    features = metadata.variable.to_list()
    label = 'UCX_abnormal'  # UCX test result
    diagnosis = 'UTI_diag'  # ED diagnosis

    # Map UCX and clinical diagnosis to int
    df[label] = df[label].map({'yes': 1, 'no': 0})
    df[diagnosis] = df[diagnosis].map({'Yes': 1, 'No': 0})

    # Reorder columns
    df = df[[label] + [diagnosis] + features]

    # Clean data
    df_cleaned = trim_missing(df)
    df_cleaned.head()

    X, encoder = encode_features(df_cleaned.iloc[:, 2:])
    Y = df_cleaned.iloc[:, :2]
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values,
                                                        test_size=0.2,
                                                        random_state=42)
    y_train, y_test = Y_train[:, 0], Y_test[:, 1]

    return X_train, X_test, y_train, y_test
