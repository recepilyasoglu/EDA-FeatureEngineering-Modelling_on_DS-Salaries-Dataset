import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
df.isnull().sum()


def get_value_counts(dataframe):
    for i in dataframe.columns:
        print(dataframe[i].value_counts())

get_value_counts(df)

df.groupby(["job_title", "experience_level"]).agg({"salary": ["mean"]})

df.groupby(["job_title"])["salary"].mean().sort_values(ascending=False)

df.groupby(["experience_level"])["salary"].mean().sort_values(ascending=False)

df.groupby("employee_residence")["salary"].mean().sort_values(ascending=False)






















