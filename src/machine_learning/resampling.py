from imblearn.over_sampling import SMOTE
from pyspark.sql import DataFrame

import numpy as np
import pandas as pd
# import sys

# sys.path.append('/home/wangui/Documents/projects/Credit-Card-Fraud-Detection-System/src')
from src.visualization.visualization import prep_data




def apply_smote(spark, df: DataFrame) -> DataFrame:
    x, y = prep_data(df)
    print("Features")
    x.show()
    y.show()

    # Apply SMOTE
    # Convert PySpark DataFrame columns to NumPy arrays
    x_features = np.array([np.array(row['features']) for row in x.collect()])
    y_labels = np.array([row['Class'] for row in y.collect()])

    # Apply SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)
    x_resampled, y_resampled = smote.fit_resample(x_features, y_labels)

    resampled_df = pd.DataFrame(data=x_resampled, columns=[f'V{i}' for i in range(1, 29)])
    resampled_df['Class'] = y_resampled

    # Convert Pandas DataFrame back to PySpark DataFrame
    resampled_spark_df = spark.createDataFrame(resampled_df)

    # Show the resulting PySpark DataFrame
    resampled_spark_df.show()
    # plot_data(resampled_spark_df)
    return resampled_spark_df
