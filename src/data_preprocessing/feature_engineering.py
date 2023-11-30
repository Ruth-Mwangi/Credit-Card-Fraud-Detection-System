from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline


def feature_engineering(spark, df):
    # Feature Engineering
    feature_cols = df.columns
    feature_cols.remove('Class')  # Exclude the target variable

    # Drop existing 'features' column if it exists
    if 'features' in df.columns:
        df = df.drop('features')

    # Create a Vector Assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Normalizing Features
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

    # Creating a Pipeline for Feature Engineering
    feature_engineering_pipeline = Pipeline(stages=[assembler, scaler])

    # Fit and Transform the Data with Feature Engineering
    df = feature_engineering_pipeline.fit(df).transform(df)

    return df
