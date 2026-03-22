# Incident Prediction from Time-Series Data

## Project Overview

The goal of this project is to predict whether an incident will occur in the near future based on past time-series data.
More specifically, the model looks at the previous **W time steps** and predicts whether an incident will happen within the next **H time steps**.
This problem is formulated as a **binary classification task** using a sliding-window approach.

---

## Dataset

I used a real dataset from the **Numenta Anomaly Benchmark (NAB)**:

- File: `cpu_utilization_asg_misconfiguration.csv`
- Description: CPU utilization of a server cluster over time

The dataset represents a realistic monitoring scenario where CPU usage fluctuates under normal conditions and occasionally shows abnormal spikes.
Dataset sources:
- Kaggle: https://www.kaggle.com/datasets/boltzmannbrain/nab  
- GitHub: https://github.com/numenta/NAB/blob/master/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv  

---

## Incident Definition

The Kaggle version of the dataset did not include anomaly labels, so I defined incidents using a threshold:
- An incident is defined as:  **CPU utilization > 80**
This threshold was chosen because most values in the dataset are significantly lower, and values above 80 indicate unusually high CPU load that could realistically trigger an alert.

---

## Sliding-Window Formulation

To convert the time series into a supervised learning problem:
- **W = 20** → number of past time steps used as input  
- **H = 5** → prediction horizon  

Each sample is constructed as:
- Input: previous 20 CPU values  
- Target:  
  - `1` if an incident occurs in the next 5 time steps  
  - `0` otherwise  

This allows the model to learn patterns that precede incidents.

---

## Models

I trained and compared three models:
- Logistic Regression (baseline)
- Random Forest
- XGBoost
These were chosen to compare a simple linear model with more powerful tree-based methods.

---

## Evaluation Setup

- Data was split using a **chronological 70/30 split**
- No random shuffling was used (important for time-series data)

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score

These metrics are important because in alerting systems we care about both:
- avoiding false alarms (precision)
- not missing incidents (recall)
  
---

## Results

| Model               | Accuracy | Precision | Recall | F1-score |
|--------------------|---------|----------|--------|----------|
| Logistic Regression | 0.797   | 0.638    | 0.653  | 0.646    |
| Random Forest       | 0.768   | 0.590    | 0.585  | 0.587    |
| XGBoost             | 0.804   | 0.681    | 0.578  | 0.626    |

### Interpretation of Results

- Logistic Regression achieved the best **F1-score and recall**, meaning it provided the most balanced performance and detected more incidents.
- XGBoost achieved the highest **accuracy and precision**, meaning its predictions were more reliable when it raised an alert, but it missed more incidents.
- Random Forest performed slightly worse across all metrics in this setup.
Overall, Logistic Regression provided the best balance for this task, while XGBoost was more conservative.

---

## Possible Improvements

In a real-world system, I would:

- Use multiple metrics (CPU, memory, latency, etc.)
- Use real incident labels instead of thresholds
- Tune model hyperparameters
- Optimize alert thresholds based on business requirements
- Evaluate performance at the alert level, not only per sample

---

## How to Run

1. Download the dataset from one of the links above and place it in the project folder  
2. Install dependencies:

```bash
pip install -r requirements.txt
