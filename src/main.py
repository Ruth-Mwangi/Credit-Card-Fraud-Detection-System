from pyspark.sql import SparkSession

from data_preprocessing.data_cleaning import load_data, handle_missing_values, save_preprocessed_data
from data_preprocessing.feature_engineering import feature_engineering
from common.common import group_data
from pyspark.sql.functions import col
from visualization.visualization import plot_data, prep_data
from machine_learning.resampling import apply_smote

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()
    # Load data
    input_path = '../data/raw/creditcard.csv'
    df = load_data(spark_session, input_path)

    # view features
    df.printSchema()
    # view top 5 rows
    df.show(5)
    # Group by the 'Class' column and count occurrences
    occurrences = group_data(df, ["Class"])
    # Calculate the total number of rows in the DataFrame
    total_rows = df.count()

    # Calculate the ratio of fraud cases
    fraud_ratio = occurrences.filter(col("Class") == 1).select("count").first()["count"] / total_rows

    # Calculate the ratio of non-fraudulent cases
    non_fraud_ratio = occurrences.filter(col("Class") == 0).select("count").first()["count"] / total_rows

    # Print the ratios
    print(f'Ratio of fraudulent cases: {fraud_ratio}\nRatio of non-fraudulent cases: {non_fraud_ratio}')

    # df = handle_missing_values(df)
    # output_path = '../data/processed/clean_data.csv'
    # save_preprocessed_data(df, output_path)
    # feature_engineered_data = feature_engineering(spark_session, df)
    # feature_engineered_data.select('scaled_features', 'Class').show(5, truncate=False)

    # visualization
    x, y = prep_data(df)
    apply_smote(spark_session, x, y)
    plot_data(x, y)
    # x.select("features").show(1,truncate=False)
    first_row_features = x.select("features").first()
    first_row_values = first_row_features['features'].toArray()
    print(f'X shape: ({x.count()}, {len(first_row_values)})')
    print(f'y shape: ({y.count()}, {len(y.columns)})')


