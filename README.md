### Welcome
My name is Alexander. I am a Data Scientist based out of Berkeley, CA. My industry experience includes Wine.com, NASA Ames Research Center, teaching Data Science at General Assembly, and consulting. What excites me most about Data Science is leveraging machine learning and big data tech to build deployable, data-driven products that bring tangible value to people's lives. 

Below are some personal projects that I've worked on in the past. More to come soon!

**NOTE:** These projects are in chronological order, so newer projects appear first and older projects appear towards the bottom. 

# Projects

## Dashboard: Scalable Machine Learning Pipeline 

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/pipeline_img.png)

#### This is a complete pipeline: 
1. Runs Classification and Regression data through an ETL pipeline
2. Parallelizable model building tool trains, scores agaisnt several metrics, and stores results
3. Model's metrics socres are displayed on learning curves. 
4. The interactive dashboard allows the user to select among several different models for both Classiciation and Regression. 

-----

#### Scalability and Performance Boost

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/runtime_reg.png)

This pipeline was tested on a 8 core laptop. The chart shows that speed increases are achieved as the number of cores increases. 
The limiting factor for thing performance boost being, of course, the run time of a single model's trian time. 

----

#### Technology

- The front end was built using a Python library called [**Dash**](https://plot.ly/products/dash/)
- The Scalable model building tool was built by me and can be found [**here**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/ml_pipeline.py)
- The machine learning models are taken from [**Scikit-Learn**](http://scikit-learn.org/stable/)
- The pipeline is being deployed on [**AWS EC2**](https://aws.amazon.com/ec2/) 

------

Check out the [**Live Dashboard Here**](http://54.215.234.117/)

Check out the [**Dash Script**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_MLTool/ml_pipeline.py)


------

## Analytical Dashboard 

![](http://www.kwhanalytics.com/wp-content/uploads/2018/01/kWh_share_logo.jpg)

![](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/dashboard_screenshot.png)

This is a prototype analytical dashboard for solar energy consumers. 

This is our scenario: imagine that one of Google's locations (there are many in the USA) has 4 buildings, each with solar panel installations. They want to keep track of 3 very importannt trends: 

1. Energy Consumption by each building
2. Energy Production by each building
3. Energy cost/profit by each building

The numbers will be tracked monthly. The cost is the energy bill for each building, so that means that the building has consumed more energy than its solar panels produced. The profit is the money made by selling excess energy back to the energy grid. In the end, we will have one years worth of data for each building. 

Check out the [**LIVE DASHBOARD HERE**](http://54.153.32.166/)

Check out the [**DASH SCRIPT**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/kwh_analytics.py)

Check out the [**JUPYTER NOTEBOOK**](https://github.com/DataBeast03/DataBeast/blob/master/Dash_Dashboard/Dashboard.ipynb) where the models were built. 


------

### Content Based Recommender for New York Times Articles 

![](https://blog.gdeltproject.org/wp-content/uploads/2014-new-york-times-logo.png)

In this notebook, I create a content based recommender for New York Times articles. This recommender is an example of a very simple data product. I follow the same proceedure outlined in this [Medium article](https://medium.com/data-lab/how-we-used-data-to-suggest-tags-for-your-story-a120076d0bb6#.4vu7uby9z).

![](https://cdn-images-1.medium.com/max/1600/1*3BP9i12zmh99F4fyjUdi3w.png)


However, we will not be recommending tags. Instead we'll be recommending new articles that a user should read based on the article that they are currently reading.

Check out the [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/NYT_Recommender/Content_Based_Recommendations.ipynb)


-----


### Machine Learning Tool

The focus of this tool is to make the machine learning model building and validation workflow very fast and easy. 

This is done by abstracting away all the cross validation and plotting functionality with a reusable class. 

This class also allows us to train and score these models in parallel. 

It also has built in learning curve plotting functionality to assess model performance.   

As a case study, we use a Cellular Service Provider data set where we are tasked with building a model that can identify users 
who are predicted to churn. Naturally in subscription based services, these data sets are unbalanced since most users 
don't cancel their subscription on any given month. 

Let's see how this tool can help us achieve our goal!

Check out the [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/ML_Tool/ML_Tool.ipynb)

```python

# create model
rfc = RandomForestClassifier(n_estimators=100, 
                             criterion='entropy', 
                             n_jobs=-1)
# initialize ml tool 
cv_rfc = cross_validation(rfc, 
                      X_churn, 
                      Y_churn, 
                      average='binary',
                      init_chunk_size=100, 
                      chunk_spacings=100,
                      n_splits=3)

# call method for model training
cv_rfc.train_for_learning_curve()

# call method for ploting model results
cv_rfc.plot_learning_curve(image_name="Learning_Curve_Plot_RF", save_image=True)

```


![](https://github.com/DataBeast03/DataBeast/blob/master/ML_Tool/Learning_Curve_Plot_RF.png)


----
### Classify Physical Activities with CNN Deep Learning Models 
<img src="https://github.com/DataBeast03/DataBeast/blob/master/DeepLearning/CNN_Activity_Classification/sport_watch_logos.png" width="400"><img src="http://adventuresinmachinelearning.com/wp-content/uploads/2017/04/CNN-example-block-diagram-1024x340.jpg" width="400">



Based on the research of the authors of this [whitepaper](https://arxiv.org/pdf/1610.07031.pdf), I trained a Convolutional Neural Network to classify the physical activities of users wearing wrist devices that contain sensors like an accelerometer and gyroscope. In order words, the CNN was trained on time-series data and not images and performed quite well. 

Check out the code in this [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/DeepLearning/CNN_Activity_Classification/CNN_Activity_Classification.ipynb). 

-----



### Entity Search Engine
<img src="http://www.listchallenges.com/f/lists/d7aacdae-74bd-42ff-b397-b73905b5867b.jpg" width="400"><img src="https://github.com/DataBeast03/DataBeast/blob/master/NYT_Articles/ScreenShot_dataViz.png" width="400">

I engineered a data product that allows the user to search for unassuming relationships bewteen entities in New York Times articles. The articles were scraped from the NYT api. I used Sklearn's implementation of Latent Dirichlet Allocation for Topic Modeling and the NLTK library for Entity Recognition. This data product is an excellent example of how Machine Learning and Natural Language Processing can be used to build an application to serve the needs of an end user. 

I wrote three object oriented classes for this project:

**topic_model_distributions** 
has methods to fit Latent Dirichlet Allocation (LDA) for topic modeling and methods to get certain distributions that are necessary to visualize the LDA results using the pyLDAvis data viz tool

**named_entity_extraction**
has methods to identify and extract named entities, the like that we observed in the police shooting article. It also has methods that count the number of entity appearances in each topic and the number of entity apperances in each article.

**entity_topic_occurances**
has methods to identify co-occurances of entities within the same topic and within the same document. 

Check out the code in this [Jupyter Notebook](https://github.com/DataBeast03/DataBeast/blob/master/NYT_Articles/NYT_Articles_2016_EDA_Presentation_Version.ipynb). 

------


### Deep Learning 
![Particle Swarm Optimization](http://www.itm.uni-stuttgart.de/research/pso_opt/bilder/pso.gif)

I designed and coded deep learning networks using the Theano neural-network library in Python. I've used deep learning to build [image classifiers](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/ImageRecognition/ImageRecogniton_CNN.ipynb), [predict the appearnce of sunspots 6 months into the future](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/TimeSeries/Long_Short_Term_Memory_(LSTM).ipynb), and to better understand how convex and non-convex optimization algorithms work, including [gradient descent](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/High_Performance_Gradient_Descent.ipynb) and [particle swarm](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/Global_Optimization.ipynb). 

----


### Big Data Engineering
![](https://s3-us-west-2.amazonaws.com/dsci6007/assets/fig2-1.png)

I've used Spark, SparkSQL, SparkStreaming, Kafka, HBase, and Lambda Architecture to engineer an [ETL pipeline](https://github.com/AlexanderPhysics/DataBeast/blob/master/DataEngineering/Batch_and_Serving_Layers.ipynb). This pipeline accepts unstructured data, then Kafka feeds identical copies of the data into a batch and speed layer. This ensure that the user can have real-time, up-to-the second data for their SQL queries. This project brought together so many different big data technologies and gave me a valuable understanding of how to design a robust data infrastructure. 

----

### Predicting Daily Activities (IoT)
![](http://www.lucas-blake.com/uploads/1250/internet-of-things-landscape__large.jpg)

The goal of [this notebook](https://github.com/DataBeast03/DataBeast/blob/master/MachineLearning/Data_Scienec_Case_Study_IoT.ipynb) is to train a classifier to predict which activities users are engaging in based on sensor data collected from devices attached to all four limbs and the torso. This will be accomplished by feature engineering the sensor data and training machine learning classifiers, SVM and a Deep Learning Network. 

----

### Sentiment Analysis 
![](http://www.clarabridge.com/wp-content/uploads/2014/04/Sentiment.jpg)

Using Sklearn's machine learning library and NLTK's NLP library, I trained several models to classify and predict [user sentiment](https://github.com/AlexanderPhysics/DataBeast/blob/master/NaturalLanguageProcessing_NLP/Sentiment_Analysis_Feature_Engineering.ipynb) on IMDB movie reviews. After exploring models and text vectorizations, I created a machine learning ensemble using Logistic Regression, SVM, and Naive Bayes, and vectorized the text into a bag-of-words representation. I then experimented on how to improve on this work by using Word2Vec neural network. 

-----

### Twitter Web Scraper
![](http://www.computerworld.dk/fil/143802/540?scale_up)

For this project, I build a [web scraper for Twitter data](https://github.com/AlexanderPhysics/DataBeast/blob/master/Twitter_Project/Twitter_Scrape_and_Analyze.ipynb) and scrape all the tweets that Galvanize has ever sent out to its followers. The unstructured data is stored in a local MongoDB database. The data is then inspected, cleaned, and structured for analysis.

-----

### Customer Churn
![](https://www.optimove.com/wp-content/uploads/2014/02/Customer-Churn-Prediction-Prevention.png)

I explore the metrics of accuracy, precision, and recall in rich detail in order to understand how unbalanced classes affect machine learning prediction of customer churn. I explore 3 ways of balancing classes, discussing the pros and cons of each appraoch. I then use the best performing model's predictive probabilities to [identify customers that are most at risk for churn](https://github.com/AlexanderPhysics/DataBeast/blob/master/Business/Churn_Problem.ipynb). Finally, I design an experiment that maximizes profits and minimizes cost for a marketing campiagn of reducing churn. 

------


## Contact
Please feel free to contact me about my work or any exciting opportunities. My email is alexanderbarriga03@gmail.com
