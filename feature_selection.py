from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

def apply_selection(df, target, method):
    X = df.drop(columns=[target])
    y = df[target]

    if method == "variance":
        selector = VarianceThreshold(threshold=0.1)
        X = selector.fit_transform(X)
        return [], X, y

    elif method == "infogain":
        scores = mutual_info_classif(X, y)
        top_features = X.columns[scores.argsort()[-5:]]
        return list(top_features), X[top_features], y

    return list(X.columns), X, y