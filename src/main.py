from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from common.common import group_data
from data_preprocessing.data_cleaning import load_data
from machine_learning.ml import split_test_train

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("CreditCardFraudDetection").config("spark.driver.memory",
                                                                                    "8g").getOrCreate()

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

    # split train and test data
    split_test_train(spark_session, df)


