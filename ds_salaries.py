import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)

df = pd.read_csv("datasets/ds_salaries.csv", index_col=0)
df.head()


# EDA
def get_stats(dataframe):
    return ("############### First 5 Line ###############", dataframe.head(),
            "############### Number of Values Owned ###############", dataframe.value_counts(),
            "############### Total Number of Observations ###############", dataframe.shape,
            "############### Variables Types ############### \n", dataframe.dtypes,
            "############### Total Number of Null Values ###############", dataframe.isnull().sum(),
            "############### Descriptive Statistics ###############", dataframe.describe().T
            )


get_stats(df)
df.shape


def get_value_counts(dataframe):
    for i in dataframe.columns:
        print(dataframe[i].value_counts())


get_value_counts(df)

df.groupby(["job_title", "experience_level"]).agg({"salary": ["mean"]})

df.groupby(["job_title"])["salary"].mean().sort_values(ascending=False)

df.groupby(["experience_level"])["salary"].mean().sort_values(ascending=False)

df.groupby("employee_residence")["salary"].mean().sort_values(ascending=False)


# Outliers
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "salary")


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "salary")


# Feature Engineering

# Converting object types to category

df.dtypes

objects = [col for col in df.columns if df[col].dtype == "O"]

df[objects]
df[objects] = df[objects].astype("category")

df.dtypes
df.head()


# Outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "salary")


# Editing abbreviated names
df["employment_type"].value_counts()

def convert_cat(dataframe, col1, col2, col3):
    mapping1 = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive-level"}
    mapping2 = {"FT": "Full Time", "PT": "Part Time", "CT": "Contract", "FL": "Freelance"}
    mapping3 = {"L": "Large", "M": "Medium", "S": "Small"}

    dataframe[col1] = dataframe[col1].map(mapping1).fillna(dataframe[col1])
    dataframe[col2] = dataframe[col2].map(mapping2).fillna(dataframe[col2])
    dataframe[col3] = dataframe[col3].map(mapping3).fillna(dataframe[col3])

    return dataframe


convert_cat(df, "experience_level", "employment_type", "company_size")

# Creating new features

# Salary
df["salary"].min(), df["salary"].max()
df["salary"].describe()

df["Salary_Category"] = pd.qcut(df["salary"], q=4,
                                labels=["Low", "Average", "High", "Very High"])

df[["salary", "Salary_Category"]].head(25)

# remote_ratio
df["remote_ratio"].value_counts()

def cat_remote_ratio(col):
    if col == 0:
        return "Not Remote"
    elif col == 50:
        return "Partially Remote"
    else:
        return "Fully Remote"

df["Remote_Category"] = df["remote_ratio"].apply(lambda x: cat_remote_ratio(x))

df[["remote_ratio", "Remote_Category"]].head(25)

df.head()
df.dtypes


# Encoding

#OHE
objects2 = [col for col in df.columns if df[col].dtype == "O"]
df[objects2] = df[objects2].astype("category")

cat_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]]
# I'm removing the job title variable from cat_cols because there are too many values in it
cat_cols = [x for x in cat_cols if x != "job_title"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

new_df = one_hot_encoder(df, cat_cols)
new_df.head()

new_df.dtypes

num_cols = [col for col in new_df.columns if new_df[col].dtypes not in ["category", "object"]]

# standartlaştırma
scaler = StandardScaler()
new_df[num_cols] = scaler.fit_transform(new_df[num_cols])

new_df[num_cols].head()

