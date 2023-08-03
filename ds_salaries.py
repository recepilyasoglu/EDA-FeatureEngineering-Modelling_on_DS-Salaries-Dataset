import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)

df = pd.read_csv("datasets/ds_salaries.csv", index_col=0)
df.head()


def get_stats(dataframe):
    return ("############### Frist 5 Line ###############", dataframe.head(),
            "############### Number of Values Owned ###############", dataframe.value_counts(),
            "############### Total Number of Observations ###############", dataframe.shape,
            "############### Variables Types ############### \n", dataframe.dtypes,
            "############### Total Number of Null Values ###############", dataframe.isnull().sum(),
            "############### Descriptive Statistics ###############", dataframe.describe().T
            )

get_stats(df)
