# Laptop_price_prediction

**Goal** - Making a model that can predict price of various kind of laptop.

**URL of Deployed of ML model** - 

https://prediction-laptop-price.herokuapp.com/predict


**Responsibilities** -

* Collected data from Kaggle.
* Pre-processed data and did Feature Engineering - 
  * Removed text attached with numerical value like Weight - from '1.3kg'to 1.3, RAM - from '4GB' to 4.
  * Getting more features from a feature by text extracting. Like for example -
   * From feature 'Memory' with value '128GB SSD' , two features extracted Memory_size(128) and memory_type("SSD')
   * For some records, memory with value '1TB SSD +  1TB HDD' , So, total 4 features extracted for all, Memory1_size, Memory1_type, Memory2_size, Memory2_type  
   * From feature 'Cpu' with value 'Intel Core i5 2.3GHz' , features Cpu_Company,Cpu_Type,Cpu_Type_intensity,Cpu_Ghz extracted with value Intel,Core,i5,2.3 respectively 
   * From feature 'GPU' with value 'Intel Iris Plus Graphics 640' , features Gpu_company,Gpu_Type extracted with values Intel,Iris respectively
   * From feature 'ScreenResolution' with value 'IPS Panel Retina Display 2560x1600', features ScreenType,SRPxlH,SRPxlV extracted with values IPS,2560,1600 respectively.
  * For categorical fetaures, very less frequent values are replaced with a new category. Like for fetaure 'Cpu_type' - value 'Cpu_type_other'
  * Did some visualization
  * Encoded categorical data by **LabelEncoder** and scaled by **StandardScaler**.  
* Built and trained several ML regressor models (**Random Forest Regressor, Support Vector Regressor, Ridge Regressor **).
* Evaluated model’s performance by **R2 score, Adjacent R2 score, rmse, mae**)
* Deployed the model performing best on **Heroku** platform using **Flask, Pickle**.
