## Welcome
My name is Alexander. I am a Data Scientist based out of Berkeley, CA. My industry experience includes Wine.com and NASA Ames Research Center. What excites me most about Data Science is leveraging machine learning and big data tech to build deployable, data-driven products that bring tangible value to people's lives. This page briefly explores some of the projects that I store in the DataBeast, WineRecommender, and NASA repos. 


## Industry 

### Recommendation System
![](https://img.grouponcdn.com/coupons/mK4w3Pv4cen2UWmZ8bH76/wine_comHIRES-500x500/v1/t200x200.png)

For my master's theis, I built an automated [wine recommendation system](https://github.com/AlexanderPhysics/Wine_Recommender/blob/master/scripts/Spark_Recommender_Prototype.ipynb) using Spark. The recommender provides users with personalized wines, presented by varietals. This project brought together a range of skills and technologies, include Exploratory Data Analysis, machine learning, and distributed computing. My favorite part of this project was overcoming the challenge of sparse ratings. Only 2.5% of Wine.com's users rate wines, so I feature engineered the purchase data to reflect user sentiment by creating a novel transformation that behaves like a statistical Z-score. 

I wrote an [article](https://github.com/DataBeast03/Wine_Recommender/blob/master/Capstone_Paper_Alexander.pdf) that explains my methodology, experimentation, and results for this project. 



### Image Feature Extraction
![](http://www.cleantechinstitute.org/Images/NASA%20Ames-Cleantech%20Institute.jpg)

I engineered a [feature extract pipeline](https://github.com/AlexanderPhysics/NASA/blob/master/image_scripts/Display_Notebook.ipynb) for satellite images: images go in, data tables come out. These images had valuable information about sunspots, including their changing positions over time and magnetic properties. The project leveraged Skimage for object detection and NASA's Pleaides supercomputer for cloud cluster computing. This data will empower Heliophysicist to design more sophisticated models of the solar cycle. The solar cycle has a direct impact on the health and stability of satellites as well as Earth's climate. 




## Projects
The following projects, and more, can be found in this repo. 


### Entity Search Engine


<img src="http://www.listchallenges.com/f/lists/d7aacdae-74bd-42ff-b397-b73905b5867b.jpg" width="450"><img src="https://github.com/DataBeast03/DataBeast/blob/master/NYT_Articles/ScreenShot_dataViz.png" width="450">






### Deep Learning 
![Particle Swarm Optimization](http://www.itm.uni-stuttgart.de/research/pso_opt/bilder/pso.gif)

I designed and coded deep learning networks using the Theano neural-network library in Python. I've used deep learning to build [image classifiers](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/ImageRecognition/ImageRecogniton_CNN.ipynb), [predict the appearnce of sunspots 6 months into the future](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/TimeSeries/Long_Short_Term_Memory_(LSTM).ipynb), and to better understand how convex and non-convex optimization algorithms work, including [gradient descent](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/High_Performance_Gradient_Descent.ipynb) and [particle swarm](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/Global_Optimization.ipynb). 

### Big Data Engineering
![](https://s3-us-west-2.amazonaws.com/dsci6007/assets/fig2-1.png)

I've used Spark, SparkSQL, SparkStreaming, Kafka, HBase, and Lambda Architecture to engineer an [ETL pipeline](https://github.com/AlexanderPhysics/DataBeast/blob/master/DataEngineering/Batch_and_Serving_Layers.ipynb). This pipeline accepts unstructured data, then Kafka feeds identical copies of the data into a batch and speed layer. This ensure that the user can have real-time, up-to-the second data for their SQL queries. This project brought together so many different big data technologies and gave me a valuable understanding of how to design a robust data infrastructure. 

### Predicting Daily Activities (IoT)
![](http://www.lucas-blake.com/uploads/1250/internet-of-things-landscape__large.jpg)

The goal of [this notebook](https://github.com/DataBeast03/DataBeast/blob/master/MachineLearning/Predicting_Daily_Activities_IoT.ipynb) is to train a classifier to predict which activities users are engaging in based on sensor data collected from devices attached to all four limbs and the torso. This will be accomplished by feature engineering the sensor data and training machine learning classifiers, SVM and a Deep Learning Network. 


### Sentiment Analysis 
![](http://www.clarabridge.com/wp-content/uploads/2014/04/Sentiment.jpg)

Using Sklearn's machine learning library and NLTK's NLP library, I trained several models to classify and predict [user sentiment](https://github.com/AlexanderPhysics/DataBeast/blob/master/NaturalLanguageProcessing_NLP/Sentiment_Analysis_Feature_Engineering.ipynb) on IMDB movie reviews. After exploring models and text vectorizations, I created a machine learning ensemble using Logistic Regression, SVM, and Naive Bayes, and vectorized the text into a bag-of-words representation. I then experimented on how to improve on this work by using Word2Vec neural network. 


### Twitter Web Scraper
![](http://www.computerworld.dk/fil/143802/540?scale_up)

For this project, I build a [web scraper for Twitter data](https://github.com/AlexanderPhysics/DataBeast/blob/master/Twitter_Project/Twitter_Scrape_and_Analyze.ipynb) and scrape all the tweets that Galvanize has ever sent out to its followers. The unstructured data is stored in a local MongoDB database. The data is then inspected, cleaned, and structured for analysis.

### Customer Churn
![](http://blog.clientheartbeat.com/wp-content/uploads/2013/09/customer-churn.jpg)

I explore the metrics of accuracy, precision, and recall in rich detail in order to understand how unbalanced classes affect machine learning prediction of customer churn. I explore 3 ways of balancing classes, discussing the pros and cons of each appraoch. I then use the best performing model's predictive probabilities to [identify customers that are most at risk for churn](https://github.com/AlexanderPhysics/DataBeast/blob/master/Business/Churn_Problem.ipynb). Finally, I design an experiment that maximizes profits and minimizes cost for a marketing campiagn of reducing churn. 


## Contact
Please feel free to contact me about my work or any exciting opportunities. My email is alexanderbarriga03@gmail.com
