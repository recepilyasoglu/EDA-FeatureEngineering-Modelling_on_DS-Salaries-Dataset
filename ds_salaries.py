import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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

# Salary Distribution by Experience Level
plt.figure(figsize=(10, 6))
sns.set(font_scale=1)
sns.barplot(data=df, x="experience_level", y="salary")
plt.title('Salary Distribution by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Distribution of Job Titles
plt.figure(figsize=(14, 6))
sns.set(font_scale=1)
sns.countplot(data=df, x="job_title", order=df["job_title"].value_counts().index)
plt.title('Distribution of Job Titles')
plt.xlabel('Job Title')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Salary Distribution of Top 15 Job Titles
top_n = 15  # top first 15 job title

plt.figure(figsize=(14, 6))
sns.set(font_scale=1)
sns.barplot(data=df, x="job_title", y="salary", order=df.groupby("job_title")["salary"].median().sort_values().index[:top_n])
plt.title(f'Salary Distribution of Top {top_n} Job Titles')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Outliers

num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, num_cols)


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, num_cols))


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

num_cols.remove("work_year")

for col in num_cols:
    replace_with_thresholds(df, col)


# Editing abbreviated names
df["employment_type"].value_counts()
df["experience_level"].value_counts()

def convert_cat(dataframe, col1, col2, col3):
    mapping1 = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive-level"}
    mapping2 = {"FT": "Full Time", "PT": "Part Time", "CT": "Contract", "FL": "Freelance"}
    mapping3 = {"L": "Large", "M": "Medium", "S": "Small"}

    dataframe[col1] = dataframe[col1].map(mapping1).fillna(dataframe[col1])
    dataframe[col2] = dataframe[col2].map(mapping2).fillna(dataframe[col2])
    dataframe[col3] = dataframe[col3].map(mapping3).fillna(dataframe[col3])

    return dataframe


convert_cat(df, "experience_level", "employment_type", "company_size")
df.head()

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

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols = [x for x in ohe_cols if x != "work_year"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

new_df = one_hot_encoder(df, ohe_cols)
new_df.head()
new_df.dtypes

num_cols = [col for col in new_df.columns if new_df[col].dtypes != "category"]
num_cols = [col for col in num_cols if col != "work_year"]

# Standardization
scaler = StandardScaler()
new_df[num_cols] = scaler.fit_transform(new_df[num_cols])

robust = RobustScaler()
new_df[num_cols] = robust.fit_transform(new_df[num_cols])

new_df[num_cols].head()


# Modelling

y = new_df["salary"]  # bağımlı değişken
X = new_df.drop(["salary", "work_year", "salary_currency", "job_title", "employee_residence", "company_location"], axis=1)  # bağımsız değişkenler, ilgili sütunlar dışındaki değerler
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Random Forest
rf_model = RandomForestRegressor().fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)  # modeli test seti üzerinde tahmin et,


# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_y_pred = lr_model.predict(X_test)

def get_methods(test, pred, model_name):
    mae = mean_absolute_error(test, pred)
    mse = mean_squared_error(test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, pred)

    print(f"{model_name} MAE Score: {mae:.2f}")
    print(f"{model_name} MSE Score: {mse:.2f}")
    print(f"{model_name} RMSE Score: {rmse:.2f}")
    print(f"{model_name} R^2 Score: {r2:.2f}")


get_methods(y_test, lr_y_pred, "Linear Regression")
get_methods(y_test, rf_y_pred, "Random Forest Regression")


# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
