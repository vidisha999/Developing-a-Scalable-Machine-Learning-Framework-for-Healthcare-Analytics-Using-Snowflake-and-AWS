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
3. Build a model drift detector, and a model monitoring function that uses "LOS" and "predicted_LOS" columns from logging table data queried in batches to calculate the current performance metric and compare it against the  reference performance metrics calculated during initial training to detect a drift.
4. If the retrain trigger is activated, retrieve all data from the logging table up to the when periodic time window begins and retrain the model on new data, then save the retrained model on a seperate folder called " Retrain Artifacts".
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
Due to the large volume of data, detection is performed in batches of 7 days periodically, collecting data within a specified time window.

- **data_monitoring_batch_query(a)** function is created to query the real time data from the logging table periodically. It use the "ADMISSION_DATE" column to specify the time window. This function only querys necessary feature columns and eliminates target variable 'LOS' or 'PREDICTED_LOS' during the quering process.
  ```python
  def data_monitoring_batch_query(a):
  from sqlalchemy import text
      query= f''' SELECT ..., ....,....                # select all necessary column names from logging table
                  FROM HEALTHDB.HEALTH_SCHEMA.LOGGING_TABLE
                  WHERE ADMISSION_DATE >= CURRENT_DATE +N -(a*7) AND ADMISSION_DATE < CURRENT_DATE +N - {(a+1)*7}'''
       # N= difference of days from current date to the begining of data, a= batch id
       return text(query)
  ```

- **data_monitoring(batch_id)** functions is created to load the training data until the specified time window, prepare the final data frame which is ready to check the data drift, by applying the trained data drift detector.

```python
def data_monitoring(batch_id):
    with engine.connect() as conn:
         batch_df=pd.DataFrame(pd.read_sql(data_monitoring_batch_query(batch_id),conn)) #query the data from SQL using snowflake-python connector 

   batch_df.columns=[col.upper() for col in batch_df.columns.tolist()] # preprocess column names for consistency
   cat_cols=[..,...,...]
   num_cols= [..,...,...]# define categorical columns and numerical columns from the df
   batch_final=batch_df[cat_cols+num_cols] # prepare the final dataframe ready for data drift detection

   with open(" Trained_Drift_Detector",'rb') as F:     # load the trained detector saved as a pkl file
       trained_drift_detector_model=pickle.load(F)
  fpreds=trained_drift_detector.predict(batch_final.values, drift_type=feature) # detect the data drift on the new data feature wise

  log_df=pd.DataFrame() # Add results of the fpreds to this dataframe
  log_df['Time period']= ([str(batch_df['ADMISSION_DATE'].max()) + 'to' + str( batch_df['ADMISSION_DATE].min())]*   # define the time period batch is processed
                                            len(batch_final.columns.tolist()])
  log_df['Features'] = final_df.columns.tolist()
  log-df['Is Drift'] = fpreds['data']['is_drift']
  log_df['Stat test'] = log_df['Features'].apply( lambda x: 'chi2' if x in cat_cols else 'KS')
  log_df['stat value'] = np.round(fpreds['data']['distance'])
  log_df['p-value'] = np.round(fpreds['data']['p_val'])

  return log_df
```
### 3. Build Model Drift detector
    
- Create **model_drift_check** function that detects drift in the model based on performance metrics of each model, based on the type of the predictive model
  ```python
  def check_model_drift(ref_metrics_dict,current_metrics_dict,type='classification', tol=0.1)
      if type == 'classification'; # following performance metrics are used if a classification model
          precision_change=abs((cur_metric_dict['Precision']-ref_metric_dict['Precision'])/ref_metric_dict['Precision'])
          recall_change=...... # calculate recall change
          roc_auc_change=..... # calculate roc_auc change

          counter=0
          for i in [precision_change, recall_change,roc_auc_change]:
                if i > tol:
                    conter +=1
          if counter > 0:
              print(" There is a model drift")
              return 1
          else:
              print('There is no model drift')
              return 0, precision_change, recall_change, roc_auc_change
  
      elif type =='regression':
           rmse_change=abs((cur_metric_dict['RMSE']-ref_metric_dict['RMSE'])/ref_metric_dict['RMSE'])
           mae_change=abs((cur_metric_dict['MAE']-ref_metric_dict['MAE'])/ref_metric_dict['MAE'])
          
          counter=0
          for i in [rmse_change, mae_change]:
                if i > tol:
                    conter +=1
          if counter > 0:
              print(" There is a model drift")
              return 1
          else:
              print('There is no model drift')
              return 0, rmse_change, mae_change # return the changes in metrics if there is a model drift
             
       else:
            print("There is no model drift.")
            rmse_change = 'NONE'
            mae_change = 'NONE'
            return 0, rmse_change, mae_change
  ```
### 3. Schedule periodic model drift detection during scoring

- Build **model_monitoring_batch_query(a)** function to query all data from the logging table in batches for the specified time window
  ```python
   def model_monitoring_batch_query(a):
   from sqlalchemy import text
         query_sim = f'''
                     SELECT *   # select all columns from the logging table
                     FROM HEALTHDB.HEALTH_SCHEMA.LOGGING_TABLE
                     WHERE ADMISSION_DATE >= CURRENT_DATE-N+{a*7} AND ADMISSION_DATE < CURRENT_DATE-N+{(a+1)*7}
        return text(query_sim)
  ```
- Build **model_monitoring(batch_id)** to periodically detect the model drift from the scoring data
  ```python
  def model_monitoring_batch(batch_id):
        with engine.connect() as conn:
            batch_df=pd.DataFrame(pd.read_sql(model_monitoring_batch_query(batch_id),conn)) #query the data from SQL using snowflake-python connector 

       # create the current performance metrics using the scoring data
       actual=batch_df['LOS_X']
       predicted = batch_df['PREDICTED_LOS']

       rmse = np.sqrt(metrics.mean_squared_error(actual,predicted))
       mae= np.sqrt(metrics.mean_absolute_error(actual,predcited))

       scoring_ref_metrics={}
       scoring_ref_metrics['rmse']=rmse
       scoring_ref_metrics['mae']=mae

       # load reference performance metrics dictionary which was saved during initial training
       
        with open('MODEL_XGB_PERFM_METRICS.pkl', 'rb') as F:
                model_ref_metrics=pickle.load(F)

  
      # detect the model drift and log (the treshold for performance metrics is 0.1 and model type is regression model).
      model_drift, rmse_change,mae_change = check_model_drift(model_ref_metric,scoring_ref_metrics,type='regression',tol=0.1)
    
      # Log values
    log = {}
    log['Time Period'] = str(batch_df['ADMISSION_DATE'].min()) + ' to ' + str(batch_df['ADMISSION_DATE'].max())
    log['Scoring Metrics'] = scoring_ref_metrics
    log['Training Metrics'] = model_ref_metric
    log['Model Drift IND'] = model_drift
    log['RMSE Change'] = RMSE_CHANGE
    log['MAE Change'] = MAE_CHANGE
    ```





