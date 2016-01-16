#Churn Problem

Users that end subscription based services are considers users that churn. Subscription based businesses have the problem of successfully predicting which users will and will not churn. The Churn_Problem notebook contains a thorough study and solution to the churn problem. 

The dataset used is a record of user behavior for a telephone compnay. The feature set includes day time minutes, international calls, and number of voice mails of every user. 

The thorough study of the Churn Problem address several common problems that Data Scientist confront:
  - The Unbalanced Class problem
  - Determining Feature Importance
  - Selecting and justifying appropriate metrics 
  - Selecting and validating Machine Learning models

#Churn Solution

First, a compnay must be able to identify which users are at risk of churning. The probability of churning. As well as the False Positives (non-churns predicted to churn) and the False Negatives (churns prediced to not churn). Typically, FNs are much more costly to a compnay than FTs -- this is taken into account. 

This study of the Churn Problem ends with the creation of a Profit Curve. 

A Profit Curve balances the trade off between the cost of reaching users that are at risk of churning and the profit of keeping users that are at risk of churning. 

###Churn_Problem_Star_Cluster Notebook

StarCluster is an open source cluster-computing toolkit for Amazon’s Elastic Compute Cloud (EC2) [http://star.mit.edu/cluster/] 

This data engineering technology is used for two reasons:
  - An efficent way to load data store on disk to RAM
  - To parallize the grid search processes in searching for optimized machine learning hyperparameters

By parallizing the grid search, the time taken to search the parameter space for the optimum combination of parameters
shortens from hours to minutes. 


