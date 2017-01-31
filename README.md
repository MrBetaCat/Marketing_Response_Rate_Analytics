## Marketing Response Rate Analytics
This project is using XGBoost to predict customer response rate when given a direct mail offer and further doing some business intelligence analytics.  

In our project, except prediction part, we'll further analysis to evaluate the performance of different XGBoost versions and maximize the profitability by using the expected value framework when considering different profit&cost or given fixed budget. 

The programming languages in this project is R.

### Summary:
1) Employed XGBoost to determine which customers will respond to a direct mail offer.  
2) Enabled parallel-version XGBoost to accelerate the training (140k+ data) and testing process by 10X+.  
3) Performed XGBoost parameters tuning by grid/random search to achieve better AUC.  

### Preparation: 
1) Because XGBoost usually takes longer time than traditional machine learning models, we organized our successful configuration steps in file use_openMP_in_R.txt. You can find how to succeed in enabling openMP when compiling openMP supported R packages especially the “XGBoost” package.

### Project Structure:  
/-------File List
&ensp;|&ensp;&ensp;&ensp;&ensp;|--------use_openMP_in_R.txt 
&ensp;|-------R Code  
&ensp;|&ensp;&ensp;&ensp;&ensp;|--------response_rate.R  

### Code Notes:
1) response_rate.R : predict cunstomer response by XGBoost with parament tuning.

### Sample Data Notes: 
1) The data is from a previous Kaggle competition: Springleaf Marketing Response. In that competition, attendees are challenged to construct new meta-variables and employ feature-selection methods to approach the daunting wide dataset.  
2) There are 145231 records as training data and 145232 records as tesing data. 
3) The total number of features are 1933. The targe variable is binary (0/1). 

### This repository will keep updating!
