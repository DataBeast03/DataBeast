## Welcome
My name is Alexander. I am a Data Scientist based out of Berkeley, CA. My industry experience includes Wine.com and NASA Ames Research Center. What excites me most about Data Science is leveraging machine learning and big data tech to build deployable, data-driven products that bring tangible value to people's lives. This page briefly explores some of the projects that I store in the DataBeast, WineRecommender, and NASA repos. 


## Industry 

![](https://img.grouponcdn.com/coupons/mK4w3Pv4cen2UWmZ8bH76/wine_comHIRES-500x500/v1/t200x200.png)

I built a scalable, automated [wine recommendation system](https://github.com/AlexanderPhysics/Wine_Recommender/blob/master/scripts/Spark_Recommender_Prototype.ipynb) using Spark. The recommender provides users with personalized wines, presented by varietals. This project brought together a range of skills and technologies, include Exploratory Data Analysis, machine learning, and distributed computing. My favorite part of this project was overcoming the challenge of sparse ratings. Only 2.5% of Wine.com's users rate wines, so I feature engineered the purchase data to reflect user sentiment by creating a novel transformation that behaves like a statistical Z-score.  




![](http://www.cleantechinstitute.org/Images/NASA%20Ames-Cleantech%20Institute.jpg)

I engineered a [feature extract pipeline](https://github.com/AlexanderPhysics/NASA/blob/master/image_scripts/Display_Notebook.ipynb) for satellite images: images go in, data tables come out. These images had valuable information about sunspots, including their changing positions over time and magnetic properties. The project leveraged Skimage for object detection and NASA's Pleaides supercomputer for cloud cluster computing. This data will empower Heliophysicist to design more sophisticated models of the solar cycle. The solar cycle has a direct impact on the health and stability of satellites as well as Earth's climate. 




## Projects
The following projects, and more, can be found in this repo. 

### Deep Learning 
![Particle Swarm Optimization](http://www.itm.uni-stuttgart.de/research/pso_opt/bilder/pso.gif)

I designed and coded deep learning networks using the Theano neural-network library in Python. I've used deep learning to build [image classifiers](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/ImageRecognition/ImageRecogniton_CNN.ipynb), [predict the appearnce of sunspots 6 months into the future](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/TimeSeries/Long_Short_Term_Memory_(LSTM).ipynb), and to better understand how convex and non-convex optimization algorithms work, including [gradient descent](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/High_Performance_Gradient_Descent.ipynb) and [particle swarm](https://github.com/AlexanderPhysics/DataBeast/blob/master/DeepLearning/Optimization/Global_Optimization.ipynb). 

### Big Data Engineering
![](https://s3-us-west-2.amazonaws.com/dsci6007/assets/fig2-1.png)

I've used Spark, SparkSQL, SparkStreaming, Kafka, HBase, and Lambda Architecture to engineer an [ETL pipeline](https://github.com/AlexanderPhysics/DataBeast/blob/master/DataEngineering/Batch_and_Serving_Layers.ipynb). This pipeline accepts unstructured data, then Kafka feeds identical copies of the data into a batch and speed layer. This ensure that the user can have real-time, up-to-the second data for their SQL queries. This project brought together so many different big data technologies and gave me a valuable understanding of how to design a robust data infrastructure. 

### Sentiment Analysis 
Using Sklearn's machine learning library and NLTK's NLP library, I trained several models to classify and predict [user sentiment](https://github.com/AlexanderPhysics/DataBeast/blob/master/NaturalLanguageProcessing_NLP/Sentiment_Analysis_Feature_Engineering.ipynb) on IMDB movie reviews. After exploring models and text vectorizations, I created a machine learning ensemble using Logistic Regression, SVM, and Naive Bayes, and vectorized the text into a bag-of-words representation. I then experimented on how to improve on this work by using Word2Vec neural network. 

## Contact
Please feel free to contact me about my work or any exciting opportunities. My email is alexanderbarriga03@gmail.com
