# Dashboard: Scalable Machine Learning Pipeline 

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/pipeline_img.png)

### This is a complete pipeline: 
1. Runs Classification and Regression data through an ETL pipeline
2. Parallelizable model building tool trains, scores agaisnt several metrics, and stores results
3. Model's metrics socres are displayed on learning curves. 
4. The interactive dashboard allows the user to select among several different models for both Classiciation and Regression. 

-----

### Scalability and Performance Boost

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/runtime_reg.png)

This pipeline was tested on a 8 core laptop. The chart shows that speed increases are achieved as the number of cores increases. 
The limiting factor for the performance boost being, of course, the run time of a single model's trian time and the number of cores. 

----

### Technology

- The front end was built using a Python library called [**Dash**](https://plot.ly/products/dash/)
- The Scalable model building tool was built by me and can be found [**here**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/ml_pipeline.py)
- The machine learning models are taken from [**Scikit-Learn**](http://scikit-learn.org/stable/)
- The pipeline is being deployed on [**AWS EC2**](https://aws.amazon.com/ec2/) 

------

Check out the [**Live Dashboard Here**](http://54.215.234.117/)

Check out the [**Dash Script**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/ml_pipeline.py)

