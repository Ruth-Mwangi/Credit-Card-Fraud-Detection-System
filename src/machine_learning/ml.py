import numpy as np
from matplotlib import pyplot as plt
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from src.machine_learning.resampling import apply_smote
from src.visualization.visualization import compare_plot


def split_test_train(spark_session,df: DataFrame):
    # Split the data into training and test sets
    (train_data, test_data) = df.randomSplit([0.8, 0.2])

    train_data_resampled = apply_smote(spark_session, train_data)
    compare_plot(train_data,train_data_resampled)
    # Create a feature vector by combining all the features
    feature_columns = [f'V{i}' for i in range(1, 29)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    # Transform the data to create the feature vector
    train_data_resampled = assembler.transform(train_data_resampled)
    test_data = assembler.transform(test_data)
    rf = RandomForestClassifier(labelCol="Class", featuresCol="features", numTrees=10, maxDepth=3)

    model = rf.fit(train_data_resampled)

    # Make predictions on the test data
    predictions = model.transform(test_data)

    # Show the predictions
    predictions.select("Class", "prediction", "probability").show(truncate=False)

    # Assuming "Class" is the actual class and "prediction" is the predicted class column
    evaluator_roc = BinaryClassificationEvaluator(labelCol="Class", rawPredictionCol="rawPrediction",
                                                  metricName="areaUnderROC")
    evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction", metricName="f1")

    f1_score = evaluator.evaluate(predictions)
    print(f"F1 Score: {f1_score:.4f}")

    # Calculate accuracy

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    print(f"Accuracy: {accuracy}")

    # Calculate f1


    # Calculate ROC AUC
    roc_auc = evaluator_roc.evaluate(predictions)
    print(f"Area Under ROC Curve: {roc_auc:.4f}")

    # Calculate confusion matrix (Note: This is a basic implementation)
    conf_matrix = predictions.groupBy("Class", "prediction").count().orderBy("Class", "prediction").toPandas()
    conf_matrix = conf_matrix.pivot(index="Class", columns="prediction", values="count").fillna(0).values
    print("Confusion Matrix:")
    print(conf_matrix)
    # You can also print other metrics if needed
    precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')


    # Plot ROC Curve (Note: This is a basic implementation)
    roc_curve = predictions.select("Class", "probability").toPandas()
    # Extracting probability of positive class
    roc_curve["prob_pos"] = roc_curve["probability"].apply(lambda x: x[1])

    # Convert Series to NumPy arrays
    prob_pos = np.array(roc_curve["prob_pos"])
    true_class = np.array(roc_curve["Class"])

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pos, true_class, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()