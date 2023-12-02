# Credit-Card-Fraud-Detection-System
This project is a credit card fraud detection system developed to identify and prevent fraudulent transactions in real-time. Leveraging big data technologies, machine learning models, and real-time processing, this project aims to enhance the security of financial transactions and protect users from unauthorized activities.

## Table of Contents

 [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Data Visualization](#data-visualization)
  - [Machine Learning](#machine-learning)
- [Model Architecture](#model-architecture)
- [Results](#results)


## Getting Started

### Prerequisites

- [Apache Spark](https://spark.apache.org/) installed and configured
- [Python](https://www.python.org/) (version >= 3.6)
- Python packages listed in the `requirements.txt` file

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ruth-Mwangi/Credit-Card-Fraud-Detection-System.git
   cd Credit-Card-Fraud-Detection-System
   ```
2. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create Data Directory**
   ```bash
   mkdir data/raw
   mkdir data/processed
   ```
5. **Download Data and Add To data/raw Directory**
   [Download data here](https://www.kaggle.com/code/patriciabrezeanu/credit-card-fraud-detection-with-tensorflow/input)


## Project-Structure

```bash
|-- data/                   # Contains raw and processed data
|-- src/                    # Source code for PySpark scripts
|-- requirements.txt        # List of Python packages required for the project
|-- .gitignore              # Specifies files and directories to be ignored by Git
|-- LICENSE                 # Project license information
|-- README.md               # Project README file
```
## Usage

### Data Preparation
Use the provided script to load and preprocess the dataset:

```bash
python src/data-preprocessing/data_preparation.py
```
Ensure that you have the dataset in CSV format at the specified input path (../data/raw/creditcard.csv). The script will load the data into a Spark DataFrame, display its schema and the top 5 rows, and calculate the ratios of fraudulent and non-fraudulent cases.

### Data Visualization
Use the provided script to visualize the data set:
```bash
python src/visualization/visualization.py
```
### Machine Learning

#### Split Test and Train Data

Before training the machine learning model, it's essential to split the dataset into training and testing sets. Execute the following script:

```bash

python ml.py
```
## Results

### Model Performance Metrics

After training and evaluating the credit card fraud detection model, the following performance metrics were obtained:

- **Accuracy:** 0.9969088917935492
- **Precision:** 0.9985605998517197
- **Recall (Sensitivity):** 0.9969088917935492
- **F1-Score:** 0.9975

These metrics provide an overview of how well the model is performing in detecting fraudulent transactions. The high accuracy indicates that the model is generally effective, while precision, recall, and F1-Score offer insights into the model's ability to correctly identify fraud cases.

