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

