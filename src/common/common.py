from pyspark.sql import DataFrame


def group_data(df: DataFrame, groupings: []) -> DataFrame:
    # Group by the 'Class' column and count occurrences
    occurrences = df.groupBy(*groupings).count()
    # occurrences.show()
    return occurrences
