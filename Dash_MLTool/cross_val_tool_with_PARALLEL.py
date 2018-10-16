from sklearn.utils import shuffle
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sb
from ipyparallel import Client
import os
from time import sleep
import multiprocessing as mp


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)




def score_model_classification(model, n_points, X_train, X_test, Y_train, Y_test, averageType='binary'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

    
    model.fit(X_train, Y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # score models 
    f1_train = f1_score(Y_train, y_pred_train, average=averageType)
    f1_test = f1_score(Y_test, y_pred_test, average=averageType)

    acc_train = accuracy_score(Y_train, y_pred_train)
    acc_test = accuracy_score(Y_test, y_pred_test)

    pre_train = precision_score(Y_train, y_pred_train, average=averageType)
    pre_test = precision_score(Y_test, y_pred_test, average=averageType)

    rec_train = recall_score(Y_train, y_pred_train, average=averageType) 
    rec_test = recall_score(Y_test, y_pred_test,average=averageType)

    return f1_train, f1_test, acc_train, acc_test, pre_train, \
           pre_test, rec_train, rec_test, n_points


def score_model_regression(model, n_points, X_train, X_test, Y_train, Y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from numpy import sqrt

    
    model.fit(X_train, Y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # score models 
    rmse_train = sqrt(mean_squared_error(Y_train, y_pred_train))
    rmse_test = sqrt(mean_squared_error(Y_test, y_pred_test))

    mse_train = mean_squared_error(Y_train, y_pred_train)
    mse_test = mean_squared_error(Y_test, y_pred_test)

    mae_train = mean_absolute_error(Y_train, y_pred_train)
    mae_test = mean_absolute_error(Y_test, y_pred_test)

    r2_train = r2_score(Y_train, y_pred_train) 
    r2_test = r2_score(Y_test, y_pred_test)

    return rmse_train, rmse_test, mse_train, mse_test, mae_train, \
           mae_test, r2_train, r2_test, n_points




class cross_validation_with_PARALLEL(object):
    '''This class provides cross validation of any classification data set by incrementally 
       increasing the number of samples in the training and test set and performing KFold 
       splits at every iteration. 
       
       During cross validation the metrics accuracy, recall, precision, and f1-score are recored. 
       The results of the cross validation are display on four learning curves. 
       
       This class can now be implement in parallel. '''
    
    def __init__(self, model, X_data, Y_data, X_test=None, Y_test=None, 
                 n_splits=3, init_chunk_size = 100, chunk_spacings = 10, average = "binary",
                learning_type="Classification"):

        self.X, self.Y =  shuffle(X_data, Y_data, random_state=1234)
        
        self.model = model
        self.learning_type = learning_type
        self.n_splits = n_splits
        self.chunk_size = init_chunk_size
        self.chunk_spacings = chunk_spacings   
        
        
        # parallel metrics
        self.m1_mean_train = []
        self.m1_mean_test = []
        self.m2_mean_train = []
        self.m2_mean_test = []
        self.m3_mean_train = []
        self.m3_mean_test = []
        self.m4_mean_train = []
        self.m4_mean_test = []
        
        
        # classification metrics
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []
        
        self.f1_mean_train = []
        self.f1_mean_test = []
        self.acc_mean_train = []
        self.acc_mean_test = []
        self.pre_mean_train = []
        self.pre_mean_test = []
        self.rec_mean_train = []
        self.rec_mean_test = []
        
        # regression metrics 
        self.rmse_train = []
        self.rmse_test = []
        self.mse_train = []
        self.mse_test = []
        self.mae_train = []
        self.mae_test = []
        self.r2_train = []
        self.r2_test = []
        
        self.rmse_mean_train = []
        self.rmse_mean_test = []
        self.mse_mean_train = []
        self.mse_mean_test = []
        self.mae_mean_train = []
        self.mae_mean_test = []
        self.r2_mean_train = []
        self.r2_mean_test = []  
        
        self.training_size = []
        self.training_size_holdout_set = []
        self.averageType = average
        
        self.tasks_list = []

    def make_chunks(self):
        '''Partitions data into chunks for incremental cross validation'''
        
        # get total number of points
        self.N_total = self.X.shape[0] 
        # partition data into chunks for learning
        self.chunks = list(np.arange(self.chunk_size, self.N_total, self.chunk_spacings ))
        self.remainder = self.X.shape[0] - self.chunks[-1]
        self.chunks.append( self.chunks[-1] + self.remainder )



    def train_for_learning_curve(self):
        '''KFold cross validates model and records metric scores for learning curves. 
           Metrics scored are f1-score, precision, recall, and accuracy'''

        # partiton data into chunks 
        self.make_chunks()
        # for each iteration, allow the model to use 10 more samples in the training set 
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1234)
        
        
        # iterate through the first n samples
        for n_points in self.chunks: 
            
        
            # split the first n samples in k folds 
            for train_index, test_index in self.skf.split(self.X[:n_points], self.Y[:n_points]):
                
                self.train_index, self.test_index = train_index, test_index
                
                self.X_train = self.X[self.train_index]
                self.X_test = self.X[self.test_index]
                self.Y_train = self.Y[self.train_index]
                self.Y_test = self.Y[self.test_index]
                
                self.model.fit(self.X_train, self.Y_train)
                self.y_pred_train = self.model.predict(self.X_train)
                self.y_pred_test = self.model.predict(self.X_test)
                
                
                if self.learning_type == "Classification":
                    self.log_metric_scores_()
                else:
                    self.log_metric_scores_regression_()
             
            if self.learning_type == "Classification":
                self.log_metric_score_means_()
            else:
                self.log_metric_score_means_regression_()
            
            self.training_size.append(n_points)      
        
    def train_for_learning_curve_PARALLEL(self, n_processes):
        '''KFold cross validates model and records metric scores for learning curves. '''

    
        # create mp pool object
        pool = mp.Pool(processes=n_processes)

        # partiton data into chunks 
        self.make_chunks()
        
        # for each iteration, allow the model to use n_points more samples in the training set 
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1234)


        # iterate through the first n samples
        for n_points in self.chunks: 


            # split the first n samples in k folds 
            for train_index, test_index in self.skf.split(self.X[:n_points], self.Y[:n_points]):

                self.train_index, self.test_index = train_index, test_index
                
                self.X_train = self.X[self.train_index]
                self.X_test = self.X[self.test_index]
                self.Y_train = self.Y[self.train_index]
                self.Y_test = self.Y[self.test_index]
                
                # run async tasks
                if self.learning_type=="Classification":
                      t = pool.apply_async(score_model_classification, args=(self.model, 
                                                                           n_points, 
                                                                           self.X_train,
                                                                           self.X_test,
                                                                           self.Y_train, 
                                                                           self.Y_test,
                                                                           self.averageType) )
                else:
                      t = pool.apply_async(score_model_regression, args=(self.model, 
                                                                           n_points, 
                                                                           self.X_train,
                                                                           self.X_test,
                                                                           self.Y_train, 
                                                                           self.Y_test) )
        
                
                # store jobs/results
                self.tasks_list.append(t) 
                
        # shut down cluster       
        pool.close()
                
        # sort asyncronous results for plotting
        self.sort_results_PARALLEL_()  
        
        
    def sort_results_PARALLEL_(self):
        '''Groups results from each worker node into a dictionary.
           Then move and sorts results into lists for ease scoring'''

        self.resutls_dict = defaultdict(list)
        # move results from tasks objects to defaultdict
        # this also groups by key, since there is no guarantee 
        # that jobs will be distributed in order with an async client     
        for t in self.tasks_list:
            results = t.get()
            n_train_samples = results[-1]
            train_test_scores = results[:-1]
            self.resutls_dict[n_train_samples].append(train_test_scores)

        # next move train_scores, test_scores, n_samples to lists for plotting
        for n_samples, scores in self.resutls_dict.items():

            # split train and test scores into separate lists 
            m1_train, m1_test, m2_train, m2_test, m3_train,\
            m3_test, m4_train, m4_test = np.array(scores).T            

            # store values in lists
            self.m1_mean_train.append(np.mean(m1_train))
            self.m1_mean_test.append(np.mean(m1_test))

            self.m2_mean_train.append(np.mean(m2_train))
            self.m2_mean_test.append(np.mean(m2_test))

            self.m3_mean_train.append(np.mean(m3_train))
            self.m3_mean_test.append(np.mean(m3_test))

            self.m4_mean_train.append(np.mean(m4_train))
            self.m4_mean_test.append(np.mean(m4_test))

            self.training_size.append(n_samples)

        # convert lists into arrays
        self.m1_mean_train = np.array(self.m1_mean_train)
        self.m1_mean_test = np.array(self.m1_mean_test)
        self.m2_mean_train = np.array(self.m2_mean_train)
        self.m2_mean_test = np.array(self.m2_mean_test)
        self.m3_mean_train = np.array(self.m3_mean_train)
        self.m3_mean_test = np.array(self.m3_mean_test)
        self.m4_mean_train = np.array(self.m4_mean_train)
        self.m4_mean_test = np.array(self.m4_mean_test)
        self.training_size = np.array(self.training_size)

        # sort values based on increasing training set size (for plotting)
        sorted_inx = np.argsort(self.training_size)
        self.training_size = self.training_size[sorted_inx]
        
        if self.learning_type == "Classification":
            self.f1_mean_train = self.m1_mean_train[sorted_inx]
            self.f1_mean_test = self.m1_mean_test[sorted_inx]
            self.acc_mean_train = self.m2_mean_train[sorted_inx]
            self.acc_mean_test = self.m2_mean_test[sorted_inx]
            self.pre_mean_train = self.m3_mean_train[sorted_inx]
            self.pre_mean_test = self.m3_mean_test[sorted_inx]
            self.rec_mean_train = self.m4_mean_train[sorted_inx]
            self.rec_mean_test = self.m4_mean_test[sorted_inx]
        else:
            self.rmse_mean_train = self.m1_mean_train[sorted_inx]
            self.rmse_mean_test = self.m1_mean_test[sorted_inx]
            self.mse_mean_train = self.m2_mean_train[sorted_inx]
            self.mse_mean_test = self.m2_mean_test[sorted_inx]
            self.mae_mean_train = self.m3_mean_train[sorted_inx]
            self.mae_mean_test = self.m3_mean_test[sorted_inx]
            self.r2_mean_train = self.m4_mean_train[sorted_inx]
            self.r2_mean_test = self.m4_mean_test[sorted_inx]           

        
    def validate_for_holdout_set(self, X_holdout, Y_holdout):
        '''Performs cross validation on the holdout set without use of KFolds.'''
        
        
        self.X_test = X_holdout
        self.Y_test = Y_holdout
        
        # partiton data into chunks 
        self.make_chunks()
        
        # clear lists of training results 
        self.reinitialize_metric_lists_()
        self.reinitialize_mean_metric_lists_()
        
        # clear results list
        self.training_size_holdout_set = []
        
        for n_points in self.chunks:
            
            self.X_train = self.X[:n_points]
            self.Y_train = self.Y[:n_points]

            self.model.fit(self.X_train, self.Y_train)
            self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = self.model.predict(self.X_test)
            
            if self.learning_type == "Classification":
                self.log_metric_scores_()
            else:
                self.log_metric_scores_regression_()

            if self.learning_type == "Classification":
                self.log_metric_score_means_()
            else:
                self.log_metric_score_means_regression_()

            self.training_size.append(n_points)
            
 
    def log_metric_score_means_regression_(self):
        '''Recrods the mean of the four regression metrics recording during training'''
        self.rmse_mean_train.append(np.sum(self.rmse_train)/len(self.rmse_train))
        self.rmse_mean_test.append(np.sum(self.rmse_test)/len(self.rmse_test))
        
        self.mse_mean_train.append(np.sum(self.mse_train)/len(self.mse_train))
        self.mse_mean_test.append(np.sum(self.mse_test)/len(self.mse_test))
        
        self.mae_mean_train.append(np.sum(self.mae_train)/len(self.mae_train))
        self.mae_mean_test.append(np.sum(self.mae_test)/len(self.mae_test))
        
        self.r2_mean_train.append(np.sum(self.r2_train)/len(self.r2_train))
        self.r2_mean_test.append(np.sum(self.r2_test)/len(self.r2_test)) 
        
        self.reinitialize_metric_lists_regression_()



                            
    def log_metric_score_means_(self):
        '''Recrods the mean of the four classification metrics recording during training'''
        self.f1_mean_train.append(np.sum(self.f1_train)/len(self.f1_train))
        self.f1_mean_test.append(np.sum(self.f1_test)/len(self.f1_test))
        
        self.acc_mean_train.append(np.sum(self.acc_train)/len(self.acc_train))
        self.acc_mean_test.append(np.sum(self.acc_test)/len(self.acc_test))
        
        self.pre_mean_train.append(np.sum(self.pre_train)/len(self.pre_train))
        self.pre_mean_test.append(np.sum(self.pre_test)/len(self.pre_test))
        
        self.rec_mean_train.append(np.sum(self.rec_train)/len(self.rec_train))
        self.rec_mean_test.append(np.sum(self.rec_test)/len(self.rec_test))
        
        self.reinitialize_metric_lists_()
            
  
    def reinitialize_metric_lists_regression_(self):
        '''Reinitializes regression metrics lists for training'''
        self.rmse_train = []
        self.rmse_test = []
        
        self.mse_train = []
        self.mse_test = []
        
        self.mae_train = []
        self.mae_test = []
        
        self.r2_train = []
        self.r2_test = []  

    def reinitialize_mean_metric_lists_regression_(self):
        '''Reinitializes regression mean metrics lists for training'''
        self.rmse_mean_train = []
        self.rmse_mean_test = []
  
        self.mse_mean_train = []
        self.mse_mean_test = []
        
        self.mae_mean_train = []
        self.mae_mean_test = []
        
        self.r2_mean_train = []
        self.r2_mean_test = []


    def reinitialize_metric_lists_(self):
        '''Reinitializes classification metrics lists for training'''
        self.f1_train = []
        self.f1_test = []
        
        self.acc_train = []
        self.acc_test = []
        
        self.pre_train = []
        self.pre_test = []
        
        self.rec_train = []
        self.rec_test = []
        
    def reinitialize_mean_metric_lists_(self):
        '''Reinitializes classification mean metrics lists for training'''
        self.f1_mean_train = []
        self.f1_mean_test = []
  
        self.acc_mean_train = []
        self.acc_mean_test = []
        
        self.pre_mean_train = []
        self.pre_mean_test = []
        
        self.rec_mean_train = []
        self.rec_mean_test = []

    def log_metric_scores_regression_(self):
        '''Records the regression metric scores during each training iteration'''
        
#         self.Y_train = self.Y_train.astype(np.float32)
#         self.y_pred_train = self.y_pred_train.astype(np.float32)
        
#         self.Y_train = self.Y_train.astype(np.float32)
#         self.y_pred_train = self.y_pred_train.astype(np.float32)
        
#         self.Y_train = self.Y_train.astype(np.float32)
#         self.y_pred_train = self.y_pred_train.astype(np.float32)
        
#         self.Y_train = self.Y_train.astype(np.float32)
#         self.y_pred_train = self.y_pred_train.astype(np.float32)
        
        self.rmse_train.append(np.sqrt(mean_squared_error(self.Y_train, self.y_pred_train)))
        self.mse_train.append(mean_squared_error( self.Y_train, self.y_pred_train) )

        self.mae_train.append(mean_absolute_error(self.Y_train, self.y_pred_train))
        self.r2_train.append(r2_score( self.Y_train, self.y_pred_train) )

        self.rmse_test.append(np.sqrt(mean_squared_error(self.Y_test, self.y_pred_test)))
        self.mse_test.append(mean_squared_error(self.Y_test, self.y_pred_test))

        self.mae_test.append(mean_absolute_error(self.Y_test, self.y_pred_test))
        self.r2_test.append(r2_score(self.Y_test, self.y_pred_test))        
            
    def log_metric_scores_(self):
        '''Records the classification metric scores during each training iteration'''
        self.f1_train.append(f1_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.acc_train.append(accuracy_score( self.Y_train, self.y_pred_train) )

        self.pre_train.append(precision_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.rec_train.append(recall_score( self.Y_train, self.y_pred_train, average=self.averageType) )

        self.f1_test.append(f1_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.acc_test.append(accuracy_score(self.Y_test, self.y_pred_test))

        self.pre_test.append(precision_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.rec_test.append(recall_score(self.Y_test, self.y_pred_test,average=self.averageType))
            

    def plot_learning_curve(self,  image_name="Learning_Curve_Plot", save_image=False):
        '''Plots f1 and accuracy learning curves for a given model and data set'''
        
        if self.learning_type == "Classification":
            fig = plt.figure(figsize = (17,12))
            # plot f1 score learning curve
            fig.add_subplot(221)   # left
            plt.title("F1-Score vs. Number of Training Samples")
            plt.plot(self.training_size, self.f1_mean_train, label="Train")
            plt.plot(self.training_size, self.f1_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("F1-Score")
            plt.grid()
            plt.legend(loc=4);

            # plot accuracy learning curve
            fig.add_subplot(222)   # right 
            plt.title("Accuracy vs. Number of Training Samples")
            plt.plot(self.training_size, self.acc_mean_train, label="Train")
            plt.plot(self.training_size, self.acc_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.legend(loc=4);

            # plot precision learning curve
            fig.add_subplot(223)   # left
            plt.title("Precision Score vs. Number of Training Samples")
            plt.plot(self.training_size, self.pre_mean_train, label="Train")
            plt.plot(self.training_size, self.pre_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("Precision")
            plt.grid()
            #plt.ylim(min(self.pre_mean_test), max(self.pre_mean_train) + 0.05)
            plt.legend(loc=4);

            # plot accuracy learning curve
            fig.add_subplot(224)   # right 
            plt.title("Recall vs. Number of Training Samples")
            plt.plot(self.training_size, self.rec_mean_train, label="Train")
            plt.plot(self.training_size, self.rec_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("Recall")
            plt.grid()
            plt.legend(loc=4);
            
        else:
            fig = plt.figure(figsize = (17,12))
            # plot rmse score learning curve
            fig.add_subplot(221)   # left
            plt.title("Root Mean Square Error vs. Number of Training Samples")
            plt.plot(self.training_size, self.rmse_mean_train, label="Train")
            plt.plot(self.training_size, self.rmse_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("RMSE")
            plt.grid()
            plt.legend(loc=1);

            # plot mse learning curve
            fig.add_subplot(222)   # right 
            plt.title("Mean Square Error vs. Number of Training Samples")
            plt.plot(self.training_size, self.mse_mean_train, label="Train")
            plt.plot(self.training_size, self.mse_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("MSE")
            plt.grid()
            plt.legend(loc=1);

            # plot mae learning curve
            fig.add_subplot(223)   # left
            plt.title("Mean Absolute Error Score vs. Number of Training Samples")
            plt.plot(self.training_size, self.mae_mean_train, label="Train")
            plt.plot(self.training_size, self.mae_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("MAE")
            plt.grid()
            #plt.ylim(min(self.pre_mean_test), max(self.pre_mean_train) + 0.05)
            plt.legend(loc=1);

            # plot r2 learning curve
            fig.add_subplot(224)   # right 
            plt.title("R-Square vs. Number of Training Samples")
            plt.plot(self.training_size, self.r2_mean_train, label="Train")
            plt.plot(self.training_size, self.r2_mean_test, label="Test");
            plt.xlabel("Number of Training Samples")
            plt.ylabel("R2")
            plt.grid()
            plt.legend(loc=1);          
        
        if  save_image==True:
            fig.savefig('./{}.png'.format(image_name))
