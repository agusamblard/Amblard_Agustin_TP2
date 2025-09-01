import pandas as pd
import numpy as np
from collections import Counter
from random import randint, uniform



def is_outlier(value, data_no_nan):
    Q1 = np.percentile(data_no_nan, 25)
    Q3 = np.percentile(data_no_nan, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return value < lower_bound or value > upper_bound

def replace_outliers_with_nan(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
        
    for column in df.columns:
        if column in exclude_columns:
            continue
        if df[column].dtype in ['float64', 'int64']:
            non_nan_values = df[column].dropna().values
            df[column] = df[column].apply(lambda x: np.nan if pd.notna(x) and is_outlier(x, non_nan_values) else x)
    return df



def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))



def knn_imputer(dataset, numeric_features, target_cols, k=5):
    """
    Imputa columnas numéricas y/o categóricas usando KNN basado en columnas numéricas.

    Parámetros:
        dataset: DataFrame original.
        numeric_features: lista de columnas numéricas para calcular distancias.
        target_cols: lista de columnas a imputar (pueden ser numéricas o categóricas).
        k: número de vecinos más cercanos.

    Retorna:
        DataFrame con las columnas imputadas.
    """
    df = dataset.copy()

    # Escalamos las columnas numéricas
    df_numeric = df[numeric_features]
    means = df_numeric.mean()
    stds = df_numeric.std()
    df_scaled = (df_numeric - means) / stds

    for target in target_cols:
        missing_idx = df[df[target].isna()].index

        for idx in missing_idx:
            row = df_scaled.loc[idx]

            # Filas que tienen el valor conocido en la columna objetivo
            known = df_scaled.loc[df[target].notna()]
            known_targets = df.loc[df[target].notna(), target]

            # Calcular distancias
            distances = known.apply(lambda x: euclidean_distance(x, row), axis=1)

            # Vecinos más cercanos
            nearest_idx = distances.nsmallest(k).index
            neighbor_values = known_targets.loc[nearest_idx]

            # Imputar
            if pd.api.types.is_numeric_dtype(df[target]):
                # Si es numérica, usamos promedio
                imputed_value = neighbor_values.mean()
            else:
                # Si es categórica, usamos la moda
                imputed_value = Counter(neighbor_values).most_common(1)[0][0]

            df.loc[idx, target] = imputed_value

    return df



def knn_imputer_auto(train_df, target_df, k=5):
    """
    Imputa automáticamente todas las columnas con NaNs en target_df,
    utilizando vecinos del conjunto de entrenamiento train_df.
    """
    target_df = target_df.copy()
    train_complete = train_df.dropna()

    cols_with_nans = target_df.columns[target_df.isna().any()].tolist()

    for target in cols_with_nans:
        print(f"Imputando columna: {target}")

        # Features numéricas (excluyendo la columna target)
        numeric_features = train_df.select_dtypes(include='number').columns.tolist()
        if target in numeric_features:
            numeric_features.remove(target)

        target_df = knn_impute_from_train(train_complete, target_df, numeric_features, target, k)

    return target_df


def smote_oversample(X, y, k=5):
    """
    Aplica SMOTE para balancear las clases en X e y.

    Parámetros:
        X: DataFrame o array de características numéricas.
        y: Serie o array con etiquetas binarias (0 y 1).
        k: Número de vecinos para interpolación.

    Retorna:
        df_res: datos balanceados con instancias sintéticas agregadas a la clase minoritaria.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Detectar clases
    class_counts = y.value_counts()
    maj_class = class_counts.idxmax()
    min_class = class_counts.idxmin()
    n_to_generate = class_counts[maj_class] - class_counts[min_class]

    X_min = X[y == min_class].reset_index(drop=True)
    new_samples = []

    for _ in range(n_to_generate):
        i = randint(0, len(X_min) - 1)
        xi = X_min.loc[i]

        # Calcular distancias con el resto
        distances = X_min.apply(lambda row: euclidean_distance(row, xi), axis=1)
        neighbors_idx = distances.nsmallest(k + 1).iloc[1:].index  # Excluir xi mismo
        neighbor = X_min.loc[neighbors_idx[randint(0, k - 1)]]

        # Interpolación
        gap = np.array([uniform(0, 1) for _ in range(len(xi))])
        synthetic = xi + gap * (neighbor - xi)
        new_samples.append(synthetic)

    X_new = pd.DataFrame(new_samples, columns=X.columns)
    y_new = pd.Series([min_class] * len(X_new))

    X_res = pd.concat([X, X_new], ignore_index=True)
    y_res = pd.concat([y, y_new], ignore_index=True)

    df_res = X_res.copy()
    # y_res debe llevar de nombre el de el y original
    df_res[y.name] = y_res
    return df_res

def knn_impute_from_train(train_df, target_df, numeric_features, target_col, k=5):
    """
    Imputa valores faltantes en target_df usando vecinos de train_df.
    Escala usando estadísticas del train.
    """
    from collections import Counter

    target_df = target_df.copy()
    
    # Escalar (con stats del train)
    train_scaled = (train_df[numeric_features] - train_df[numeric_features].mean()) / train_df[numeric_features].std()
    target_scaled = (target_df[numeric_features] - train_df[numeric_features].mean()) / train_df[numeric_features].std()

    for idx in target_df[target_df[target_col].isna()].index:
        row = target_scaled.loc[idx]

        # Solo usar filas conocidas del train
        known = train_scaled[train_df[target_col].notna()]
        known_targets = train_df.loc[train_df[target_col].notna(), target_col]

        distances = known.apply(lambda x: euclidean_distance(x, row), axis=1)
        nearest_idx = distances.nsmallest(k).index
        neighbor_values = known_targets.loc[nearest_idx]

        # Imputar
        if pd.api.types.is_numeric_dtype(known_targets):
            imputed_value = neighbor_values.mean()
        else:
            imputed_value = Counter(neighbor_values).most_common(1)[0][0]

        target_df.loc[idx, target_col] = imputed_value

    return target_df