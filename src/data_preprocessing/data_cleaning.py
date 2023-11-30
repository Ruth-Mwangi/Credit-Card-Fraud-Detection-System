import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline


def load_data(spark, file_path) -> DataFrame:
    return spark.read.csv(file_path, header=True, inferSchema=True)


def handle_missing_values(df) -> DataFrame:
    # Handling Missing Values
    imputer = Imputer(inputCols=df.columns, outputCols=df.columns)
    df = imputer.setStrategy("mean").fit(df).transform(df)
    return df


def convert_time(df) -> DataFrame :
    df['Time_in_hours'] = df['Time'] / 3600
    return df.drop(['Time'], axis=1)


def save_preprocessed_data(df, output_path):
    df.write.csv(output_path, header=True, mode="overwrite")

