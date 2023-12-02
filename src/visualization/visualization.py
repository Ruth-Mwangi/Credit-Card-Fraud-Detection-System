import numpy as np
from matplotlib import pyplot as plt
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler


def prep_data(df: DataFrame) -> (DataFrame, DataFrame):
    """
    Convert the DataFrame into two variables:
    x: data columns (V1 - V28) as a PySpark DataFrame
    y: label column as a PySpark DataFrame
    """
    # Select the feature columns (V1 - V28)
    feature_columns = [f'V{i}' for i in range(1, 29)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    x = assembler.transform(df).select("features")
    # x.limit(10).show(10)

    # Select the label column
    y = assembler.transform(df).select("Class")
    # y.limit(10).show(10)

    return x, y


# Define a function to create a scatter plot of our data and labels
def plot_data(df: DataFrame):
    x, y = prep_data(df)
    x_pd = x.sample(False, 0.1).toPandas()  # Sample 10% of the data for visualization
    y_pd = y.sample(False, 0.1).toPandas()
    x_pd = x_pd.reset_index(drop=True)
    y_pd = y_pd.reset_index(drop=True)
    x.show()
    y.show()

    plt.scatter(x_pd[y_pd['Class'] == 0]['features'].apply(lambda x: x[0]),
                x_pd[y_pd['Class'] == 0]['features'].apply(lambda x: x[1]),
                label="Class #0", alpha=0.5, linewidth=0.15)

    plt.scatter(x_pd[y_pd['Class'] == 1]['features'].apply(lambda x: x[0]),
                x_pd[y_pd['Class'] == 1]['features'].apply(lambda x: x[1]),
                label="Class #1", alpha=0.5, linewidth=0.15, c='r')

    plt.legend()
    return plt.show()


def compare_plot(original_df: DataFrame, resampled: DataFrame):
    x, y = prep_data(original_df)
    x_pd = x.sample(False, 0.1).toPandas()  # Sample 10% of the data for visualization
    y_pd = y.sample(False, 0.1).toPandas()
    x_pd = x_pd.reset_index(drop=True)
    y_pd = y_pd.reset_index(drop=True)

    x_resampled, y_resampled = prep_data(resampled)
    x_resampled_pd = x_resampled.sample(False, 0.1).toPandas()  # Sample 10% of the data for visualization
    y_resampled_pd = y_resampled.sample(False, 0.1).toPandas()
    x_resampled_pd = x_resampled_pd.reset_index(drop=True)
    y_resampled_pd = y_resampled_pd.reset_index(drop=True)

    plt.subplot(1, 2, 1)
    plt.scatter(x_pd[y_pd['Class'] == 0]['features'].apply(lambda x: x[0]),
                x_pd[y_pd['Class'] == 0]['features'].apply(lambda x: x[1]),
                label="Class #0", alpha=0.5, linewidth=0.15)

    plt.scatter(x_pd[y_pd['Class'] == 1]['features'].apply(lambda x: x[0]),
                x_pd[y_pd['Class'] == 1]['features'].apply(lambda x: x[1]),
                label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('Original Set')
    plt.subplot(1, 2, 2)
    plt.scatter(x_resampled_pd[y_resampled_pd['Class'] == 0]['features'].apply(lambda x: x[0]),
                x_resampled_pd[y_resampled_pd['Class'] == 0]['features'].apply(lambda x: x[1]),
                label="Class #0", alpha=0.5, linewidth=0.15)

    plt.scatter(x_resampled_pd[y_resampled_pd['Class'] == 1]['features'].apply(lambda x: x[0]),
                x_resampled_pd[y_resampled_pd['Class'] == 1]['features'].apply(lambda x: x[1]),
                label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.title('SMOTE')
    plt.legend()
    plt.show()


