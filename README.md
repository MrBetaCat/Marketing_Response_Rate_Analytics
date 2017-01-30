## Marketing Response Rate Analytics
This project is using xgboost to predict customer response rate when given a direct mail offer and further doing some business intelligence analytics.  
The data is from a previous Kaggle competition: Springleaf Marketing Response.  
There are 145231 records as training data and 145232 records as tesing data. The total number of features are 1933. The targe variable is binary (0/1).  
In that competition, attendees are challenged to construct new meta-variables and employ feature-selection methods to approach the daunting wide dataset.  
In our project, except prediction part, we'll further analysis to evaluate the performance of different xgboost versions and maximize the profitability by using the expected value framework when considering different profit&cost or given fixed budget. 

The programming language is R.

I. Because xgboost usually takes longer time than traditional machine learning models, we organized our successful configuration steps in file use_openMP_in_R.txt. You can find how to succeed in enabling openMP when compiling openMP supported R packages especially the “xgboost” package.

II. Codes with detailed comments are included in file response_rate.R.   
1. Employed xgboost to determine which customers will respond to a direct mail offer.   
2. Enabled parallel-version xgboost to accelerate the training and testing process.  
3. Performed xgboost parameters tuning by grid/random search to achieve better accuracy. 


### This repository will keep updating!
