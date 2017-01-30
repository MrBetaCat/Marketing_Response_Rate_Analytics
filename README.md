## Marketing Response Rate Analytics
This project is using xgboost to predict customer response rate when given a direct mail offer and further doing some business intelligence analytics.  
The data is from a previous Kaggle competition: Springleaf Marketing Response.
In that competition, attendees are challenged to construct new meta-variables and employ feature-selection methods to approach the daunting wide dataset.  
In our project, except prediction part, we did further analysis to evaluate the performance of different xgboost versions and maximize the profitability 
by using the expected value framework when considering different profit&cost or given fixed budget.  

There are three parts:
1. how to install openMP supported xgboost in R.  
2. Project code written by R language.  
3. Final analytics.  


1. Because xgboost usually takes longer time than traditional machine learning models, we organized our successful configuration steps in file use_openMP_in_R.txt.
You can find how to succeed in enabling openMP when compiling openMP supported R packages especially the “xgboost” package.

2. 
