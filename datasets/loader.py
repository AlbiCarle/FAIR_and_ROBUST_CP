import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


@dataclass
class Dataset:

    X_train: torch.Tensor
    X_cal: torch.Tensor
    X_test: torch.Tensor

    y_train: torch.Tensor
    y_cal: torch.Tensor
    y_test: torch.Tensor

    s_train: torch.Tensor
    s_cal: torch.Tensor
    s_test: torch.Tensor

    s_index: int
    name: str

def flip_sensitive(X, s_index):
    X_cf = X.clone()
    X_cf[:, s_index] = 1 - X_cf[:, s_index]
    return X_cf

def load_adult():
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ["age","workclass","fnlwgt","education-num","marital-status",
            "occupation","relationship","race","sex","capital-gain","capital-loss",
            "hours-per-week","native-country","income"] #"education",

    data = pd.read_csv(url, names=cols, sep=",\s*", engine='python')

    # target: 0/1 -> -1/+1 for hinge loss
    y = ((data["income"] == ">50K").astype(int) * 2 - 1).values.astype(np.float32)

    # sensitive attribute: 0=Female, 1=Male
    sensitive_attr = (data["sex"] == "Male").astype(int).values.astype(np.float32)

    X = data.drop("income", axis=1)
    X = pd.get_dummies(X, drop_first=True)

    # split train / cal / test
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, sensitive_attr, test_size=0.4, random_state=42, stratify=y
    )

    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # optional
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    s_train = s_train[:2000]

    s_index = list(X.columns).index('sex_Male')

    scaler = StandardScaler()
    num_cols = list(range(X.shape[1]))
    num_cols.remove(s_index)

    X_train = X_train.values.astype(np.float32)
    X_cal = X_cal.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)

    X_train[:, num_cols] = scaler.fit_transform(X_train[:, num_cols])
    X_cal[:, num_cols] = scaler.transform(X_cal[:, num_cols])
    X_test[:, num_cols] = scaler.transform(X_test[:, num_cols])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    s_train_t = torch.tensor(s_train, dtype=torch.float32)
    s_cal_t = torch.tensor(s_cal, dtype=torch.float32)
    s_test_t = torch.tensor(s_test, dtype=torch.float32)

    return Dataset(
        X_train_t, X_cal_t, X_test_t,
        y_train_t, y_cal_t, y_test_t,
        s_train_t, s_cal_t, s_test_t,
        s_index,
        "adult"
    )


def load_compas():
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)

    # === TARGET ===
    y = ((df["two_year_recid"] == 1).astype(int) * 2 - 1).values.astype(np.float32)

    # === SENSITIVE ATTRIBUTE ===
    s = (df["race"] == "Caucasian").astype(int).values.astype(np.float32)
    
    # === NUMERIC FEATURES ===
    num_cols = ["age", "priors_count",  "days_b_screening_arrest"] #"decile_score",
    X_num = df[num_cols].fillna(0)

    # === CATEGORICAL FEATURES ===
    cat_cols = ["c_charge_degree", "race", 'sex', "age_cat"]  # "score_text"
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    # === COMBINA FEATURES ===
    X = pd.concat([X_num, X_cat], axis=1)

    # === TRAIN / CAL / TEST SPLIT ===
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.4, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    s_index = list(X.columns).index("race_Caucasian")
    
    # === SCALING NUMERIC ===
    scaler = StandardScaler()
    num_idx = [X.columns.get_loc(c) for c in num_cols]

    X_train_np = X_train.values.astype(np.float32)
    X_cal_np = X_cal.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    X_train_np[:, num_idx] = scaler.fit_transform(X_train_np[:, num_idx])
    X_cal_np[:, num_idx] = scaler.transform(X_cal_np[:, num_idx])
    X_test_np[:, num_idx] = scaler.transform(X_test_np[:, num_idx])

    # === CONVERSIONE A TORCH TENSOR ===
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    s_train_t = torch.tensor(s_train, dtype=torch.float32)
    s_cal_t = torch.tensor(s_cal, dtype=torch.float32)
    s_test_t = torch.tensor(s_test, dtype=torch.float32)

    return Dataset(
        X_train_t, X_cal_t, X_test_t,
        y_train_t, y_cal_t, y_test_t,
        s_train_t, s_cal_t, s_test_t,
        s_index, "compas"
    )

def load_german():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    cols = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment_since", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age",
        "other_installment_plans", "housing", "existing_credits",
        "job", "people_liable", "telephone", "foreign_worker", "class"
    ]

    df = pd.read_csv(url, sep=" ", header=None)
    df.columns = cols

    # === TARGET ===
    y = ((df["class"] == 1).astype(int) * 2 - 1).values.astype(np.float32)

    # === SENSITIVE ATTRIBUTE ===
    s = (df["foreign_worker"] == "A201").astype(int).values.astype(np.float32)

    # === NUMERIC FEATURES ===
    num_cols = [
        "duration", "credit_amount", "installment_rate",
        "residence_since", "age", "existing_credits", "people_liable"
    ]
    X_num = df[num_cols]

    # === CATEGORICAL FEATURES ===
    cat_cols = [c for c in df.columns if c not in num_cols + ["class"]]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    # === COMBINE FEATURES ===
    X = pd.concat([X_num, X_cat], axis=1)

    # === TRAIN / CAL / TEST SPLIT ===
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.7, random_state=42, stratify=y
    )
    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.3, random_state=42, stratify=y_temp
    )

    s_index = list(X.columns).index("foreign_worker_A202") if "foreign_worker_A202" in X.columns else None

    # === SCALING NUMERIC ===
    scaler = StandardScaler()
    num_idx = [X.columns.get_loc(c) for c in num_cols]

    X_train_np = X_train.values.astype(np.float32)
    X_cal_np = X_cal.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    X_train_np[:, num_idx] = scaler.fit_transform(X_train_np[:, num_idx])
    X_cal_np[:, num_idx] = scaler.transform(X_cal_np[:, num_idx])
    X_test_np[:, num_idx] = scaler.transform(X_test_np[:, num_idx])

    # === TORCH TENSORS ===
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    s_train_t = torch.tensor(s_train, dtype=torch.float32)
    s_cal_t = torch.tensor(s_cal, dtype=torch.float32)
    s_test_t = torch.tensor(s_test, dtype=torch.float32)

    return Dataset(
        X_train_t, X_cal_t, X_test_t,
        y_train_t, y_cal_t, y_test_t,
        s_train_t, s_cal_t, s_test_t,
        s_index, "german"
    )

def load_student():

    # === FETCH DATASET ===
    student = fetch_ucirepo(id=320)

    X = student.data.features.copy()
    y_raw = student.data.targets["G3"]

    # === TARGET ===
    y = ((y_raw >= 10).astype(int) * 2 - 1).values.astype(np.float32)

    # === SENSITIVE ATTRIBUTE ===
    s = (X["sex"] == "M").astype(int).values.astype(np.float32)

    # === REMOVE TARGET LEAKAGE ===
    X = X.drop(columns=["G1", "G2"], errors="ignore")

    # === NUMERIC FEATURES ===
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    X_num = X[num_cols]

    # === CATEGORICAL FEATURES ===
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_cat = pd.get_dummies(X[cat_cols], drop_first=True)

    # === COMBINE FEATURES ===
    X = pd.concat([X_num, X_cat], axis=1)

    # === TRAIN / CAL / TEST SPLIT ===
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.8, random_state=42, stratify=y
    )

    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.3, random_state=42, stratify=y_temp
    )

    # === INDEX SENSITIVE FEATURE ===
    s_index = list(X.columns).index("sex_M")

    # === SCALING NUMERIC ===
    scaler = StandardScaler()
    num_idx = [X.columns.get_loc(c) for c in num_cols]

    X_train_np = X_train.values.astype(np.float32)
    X_cal_np = X_cal.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    X_train_np[:, num_idx] = scaler.fit_transform(X_train_np[:, num_idx])
    X_cal_np[:, num_idx] = scaler.transform(X_cal_np[:, num_idx])
    X_test_np[:, num_idx] = scaler.transform(X_test_np[:, num_idx])

    # === TORCH TENSORS ===
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_cal_t = torch.tensor(X_cal_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_cal_t = torch.tensor(y_cal, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    s_train_t = torch.tensor(s_train, dtype=torch.float32)
    s_cal_t = torch.tensor(s_cal, dtype=torch.float32)
    s_test_t = torch.tensor(s_test, dtype=torch.float32)

    return Dataset(
        X_train_t, X_cal_t, X_test_t,
        y_train_t, y_cal_t, y_test_t,
        s_train_t, s_cal_t, s_test_t,
        s_index, "student"
    )

# +
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_arrhythmia():

    # === LOAD ===
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
    df = pd.read_csv(url, header=None)

    # === MISSING VALUES ===
    df = df.replace("?", np.nan).astype(float)

    # === TARGET ===
    # Classe 1 = normale, tutto il resto = aritmia
    y_raw = df.iloc[:, -1].values
    y = (y_raw != 1).astype(int) * 2 - 1
    y = y.astype(np.float32)

    # === SENSITIVE ATTRIBUTE ===
    # colonna 1 = sex (0 = female, 1 = male)
    s = df.iloc[:, 1].values.astype(np.float32)

    # === FEATURES ===
    X = df.iloc[:, :-1].copy()

    s_index = 1
    #X = X.drop(columns=[s_index])

    s_index_new = s_index#None  

    # === IMPUTATION ===
    imputer = SimpleImputer(strategy="mean")
    X_np = imputer.fit_transform(X)

    # === SPLIT ===
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X_np, y, s, test_size=0.4, random_state=42, stratify=y
    )

    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # === SCALING ===
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_cal   = scaler.transform(X_cal)
    X_test  = scaler.transform(X_test)

    # === TORCH ===
    def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

    return Dataset(
        to_tensor(X_train), to_tensor(X_cal), to_tensor(X_test),
        to_tensor(y_train), to_tensor(y_cal), to_tensor(y_test),
        to_tensor(s_train), to_tensor(s_cal), to_tensor(s_test),
        s_index_new,  
        "arrhythmia"
    )


# -

def load_bank():

    import numpy as np
    import pandas as pd
    import torch

    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # === FETCH ===
    bank = fetch_ucirepo(id=222)

    X = bank.data.features.copy()
    y_raw = bank.data.targets["y"]

    # === TARGET ===
    y = ((y_raw == "yes").astype(int) * 2 - 1).values.astype(np.float32)

    # === SENSITIVE ATTRIBUTE ===
    s = (X["marital"] == "married").astype(int).values.astype(np.float32)

    # === FEATURES ===
    X = pd.get_dummies(X, drop_first=True)

    # === SPLIT ===
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X, y, s, test_size=0.4, random_state=42, stratify=y
    )

    X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
        X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # === SENSITIVE INDEX ===
    s_index = list(X.columns).index("marital_married")

    # === SCALING ===
    scaler = StandardScaler()

    X_train_np = X_train.values.astype(np.float32)
    X_cal_np   = X_cal.values.astype(np.float32)
    X_test_np  = X_test.values.astype(np.float32)

    num_idx = list(range(X.shape[1]))
    num_idx.remove(s_index)

    X_train_np[:, num_idx] = scaler.fit_transform(X_train_np[:, num_idx])
    X_cal_np[:, num_idx]   = scaler.transform(X_cal_np[:, num_idx])
    X_test_np[:, num_idx]  = scaler.transform(X_test_np[:, num_idx])

    # === TORCH ===
    def to_tensor(x): return torch.tensor(x, dtype=torch.float32)

    return Dataset(
        to_tensor(X_train_np), to_tensor(X_cal_np), to_tensor(X_test_np),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_cal, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(s_train, dtype=torch.float32),
        torch.tensor(s_cal, dtype=torch.float32),
        torch.tensor(s_test, dtype=torch.float32),
        s_index,
        "bank"
    )


def load_dataset(name):
    if name == "adult":
        return load_adult()
    
    if name == "compas":
         return load_compas()
        
    if name == "german":
         return load_german()
        
    if name == "student":
         return load_student()
        
    if name == "arrhythmia":
        return load_arrhythmia()
        
    if name == "bank":
        return load_bank()
    
    raise ValueError(f"Dataset {name} unknown")
