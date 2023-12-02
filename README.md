# Credit-Card-Fraud-Detection-System
This project is a credit card fraud detection system developed to identify and prevent fraudulent transactions in real-time. Leveraging big data technologies, machine learning models, and real-time processing, this project aims to enhance the security of financial transactions and protect users from unauthorized activities.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

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
