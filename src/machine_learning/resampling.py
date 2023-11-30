from imblearn.over_sampling import SMOTE
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
import sys

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType

sys.path.append('/home/wangui/Documents/projects/Credit-Card-Fraud-Detection-System/src')
from src.visualization.visualization import plot_data
import numpy as np
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array


def apply_smote(spark, x: DataFrame, y: DataFrame):
    x_pd = x.toPandas()
    y_pd = y.toPandas()
    # Define a UDF to convert a Spark Vector to a Python list
    vector_to_array_udf = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

    # Assuming 'features' is a Spark Vector column
    assembler = VectorAssembler(inputCols=["features"], outputCol="features_assembled")
    x_spark = assembler.transform(x).select("features_assembled")

    # Convert the Spark Vector to a Pandas DataFrame
    x_pd = x_spark.withColumn("features", vector_to_array_udf("features_assembled")).toPandas()

    # Convert the features to a NumPy array
    x_features = np.vstack(x_pd['features'].apply(lambda x: np.array(x)))

    # Combine the resampled features with the labels
    resampled_df = pd.concat(
        [pd.DataFrame(x_features, columns=[f'feature_{i}' for i in range(x_features.shape[1])]), y_pd], axis=1)

    # Define the resampling method
    method = SMOTE()

    # Apply SMOTE
    x_resampled, y_resampled = method.fit_resample(resampled_df.drop('Class', axis=1), resampled_df['Class'])

    # Convert back to PySpark DataFrames
    x_resampled_spark = spark.createDataFrame(pd.DataFrame(x_resampled), schema=["features"])
    y_resampled_spark = spark.createDataFrame(pd.DataFrame(y_resampled), schema=["Class"])

    # Plot the resampled data
    plot_data(x_resampled_spark, y_resampled_spark)
