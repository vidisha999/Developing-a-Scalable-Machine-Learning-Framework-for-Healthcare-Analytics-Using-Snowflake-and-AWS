# Healthcare Analytics to predict Patients' length of hospital stay integrating snowflake and AwS 
## Project Description
This project focuses on the analysis of healthcare data which is stored in the snowflake(a cloud-based data warehousing platform and database management system), to predict a patient's length of stay at hospital at the time of admission.AWS sagemaker is use to create a scheduled python notebook instance for model building and development and then real-time predictions are made using the trained model.Resulting predictions from the new data which are fetched at the scheduled time are fed back to the database system and stored in a seperate logging table for later use for model retraining and improvements. The intent of saving the predictions in the logging table is to analyze the  model preditions made over a period of time and  use that information for refining the model for improved predictive accuracy.

## Business Overview
 A critical area in the healthcare management is to improve the delivery of its services and  ensure patient outcomes. Analyzing the length of the stay (LOS) of patients at the time of admitting to the hospital would help healthcare providers to identify the areas to improve the delivery of care and make data driven decisions in the cost management.By analyzing the factors contribute the variation in LOS, business teams can derrive insights regarding the patients who have a risk of extending the LOS, in order to allocate available  resources in a timely manner for all patients, reducing unexpected outcomes and  costs due to facility limitations and operational defficiencies.In overall Analyzing the LOS is a critical aspect in the healthcare analytics which helps organizations gain a deeper understanding about the inpatient care and cost management. 
 
 ## Project Architecture 
 ![reference diagram](Images/Flowcharts.png)
As shown in the reference architecture, 
1. the  Exploratory Data Analysis(EDA) and the Feature Engineering steps are performed in snowflake
2. Data Preprocessing, Feature Selection and Model building and training steps are performed in AWS sagemaker
3. The logging table for storing predictions in the scoring process is stored in the snowflake
4. The Snowflake and AWS SageMaker environments are integrated using the Snowflake Python Connector.

## Core Objectives 
 1. Build a machine learning model to predict the LOS of patients.
 2. Simulate live data scoring and insert predictions to the logging table.
 3. Send status email at each step of model bilding pipeline using SMTP mail.

## Data 
The training data loaded in snowflake are collected from 230k patients across various regions and hospitals. There are total 19 features available in the data.
The simulation data is available for 71K patients for prediction purpose.

-------
## Steps in the retraining pipeline 
1. Create the Data drift detector object using the training data and save the detector as a pickle file for use during retraining.
2. Create a schedule which runs periodically to check a data drift from the data in the logging table created during the scoring process and to raise a trigger if there's a drift.
3. If the retrain trigger is activated, retrieve all data from the logging table up to the when periodic time window begins and retrain the model on new data, then save the retrained model on a seperate folder called " Retrain Artifacts".
5. Old trained model and newly retrained model should be tested  on the remaining data in the logging table, which is only within the priodic time window (testing data) to evalaute the model performance by comparing their performance metrics.
6. Once the final model is selected, the other model should be pushed to "Archives" folder and the selcted model should be in the currect directory along with its features and mappings.

### 1. Buiding Data Drift Detector 

Alibi python library is used  to build the data drift detector due to its simplicity and versality in  data detection. 
- It uses the **TabularDrift** method which performs **feature-wise test** to comapre the statistical properties of the features in the reference dataset(trainng dataset) against the new dataset to identify a data drift. The **feature-wise- test** uses **Chi-square** test for categorical features and **two sample Kolmogorov- Sirmonov** test for continuous numerical data.
- To reduce the complexity and improve the computational efficiency ,each categorical features and their categories are converted to numeric values before passing the features to the data drift detector.(Knowing the categorical data allows detector to seperately identify categorical and numerical data to apply appropriate statistial test)

```python
import alibi
from alibi_detect.cd import TabularDrift
# A list containing  categorical columns' indices derrived from the X_train dataframe and a dictionary with None values passed for each index are created to allow detector to infer the numerical values from the reference data.
catgeories_per_feature= { f: None for f in cat_indices}
# Initialize the data drift detector
cd=TabularDrift(x_train.values,p_val=0.05, categories_per_feature=categories_per_feature)
# saving the data detector as a pickle file
with open ("Trained_Drift_Detector.pkl", "wb") as F:
      pickle.dump(cd,F)
```
### 2. schedule a periodic data drift detection during scoring 
Due to the large volume of data, detection is performed in batches periodically, collecting data within a specified time window.

- **data_monitoring_batch_query(a)** function is created to query the real time data from the logging table periodically. It use the "ADMISSION_DATE" column to specify the time window.
  ```python
  def data_monitoring_batch_query(a):
  from sqlalchemy import text
      query= f''' SELECT ..., ....,....                # select all column names from logging table
                  FROM HEALTHDB.HEALTH_SCHEMA.LOGGING_TABLE
                  WHERE ADMISSION_DATE >= CURRENT_DATE +N -(a*7) AND ADMISSION_DATE < CURRENT_DATE +N - {(a+1)*7}'''
       # N= difference of days from current date to the begining of data, a= batch id
       return text(query)
  ```

















