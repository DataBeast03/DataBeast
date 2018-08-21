from sklearn.utils import shuffle
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score 
import matplotlib.pyplot as plt
import seaborn as sb
from ipyparallel import Client
import os
from time import sleep
import multiprocessing


import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)




def score_model(model, n_points, X_train, X_test, Y_train, Y_test, averageType='binary'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score 

    
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



class cross_validation_with_PARALLEL(object):
    '''This class provides cross validation of any classification data set by incrementally 
       increasing the number of samples in the training and test set and performing KFold 
       splits at every iteration. 
       
       During cross validation the metrics accuracy, recall, precision, and f1-score are recored. 
       The results of the cross validation are display on four learning curves. 
       
       This class can now be implement in parallel. '''
    
    def __init__(self, model, X_data, Y_data, X_test=None, Y_test=None,\
                 n_splits=3, init_chunk_size = 100, chunk_spacings = 10, average = "binary"):

        self.X, self.Y =  shuffle(X_data, Y_data, random_state=1234)
        
        self.model = model
        self.n_splits = n_splits
        self.chunk_size = init_chunk_size
        self.chunk_spacings = chunk_spacings        
        
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.X_holdout = []
        self.Y_holdout = []
        
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
                self.log_metric_scores_()   
                
            self.log_metric_score_means_()
            self.training_size.append(n_points)
            
        # clear lists for future runs
        #self.reinitialize_mean_metric_lists_()        

# -----------
    # NOTE: Can't send score_model as method into worker nodes
    # TypeError: can't pickle _thread.lock objects
    # But passing in as a score_model fuction works 
    # INVESTIGATE THIS AT SOME POINT
#     def score_model(self, model, n_points, X_train, X_test, Y_train,Y_test, averageType):
        
#         from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
#         # fit model
#         model.fit(X_train, Y_train)

#         # get train and test predictions
#         y_pred_train = model.predict(X_train)
#         y_pred_test = model.predict(X_test)

#         # score models 
#         f1_train = f1_score(Y_train, y_pred_train, average=averageType)
#         f1_test = f1_score(Y_test, y_pred_test, average=averageType)

#         acc_train = accuracy_score(Y_train, y_pred_train)
#         acc_test = accuracy_score(Y_test, y_pred_test)

#         pre_train = precision_score(Y_train, y_pred_train, average=averageType)
#         pre_test = precision_score(Y_test, y_pred_test, average=averageType)

#         rec_train = recall_score(Y_train, y_pred_train, average=averageType) 
#         rec_test = recall_score(Y_test, y_pred_test,average=averageType)

#         return f1_train, f1_test, acc_train, acc_test, pre_train, \
#                pre_test, rec_train, rec_test, n_points       
# ------------        
        
    #def train_for_learning_curve_PARALLEL(self, lb_view, score_model):
    def train_for_learning_curve_PARALLEL(self, n_cpus):
        '''KFold cross validates model and records metric scores for learning curves. 
           Metrics scored are f1-score, precision, recall, and accuracy
           
           This function enables parallel processing.
           
           lb_view - ipyparellel load balance client instance
           score_model - function for scoring classification predictions'''
        
        
        # make sure user passed in n - 1 for cpus to be used for this job
        cpu_count =  multiprocessing.cpu_count()
        if n_cpus > cpu_count:
            n_cpus = cpu_count - 1
        
        # spin up cluster
        # start up a local cluster of n cpus
        os.system("ipcluster start -n={} --daemon".format(n_cpus))

        # wait for clientn to come online
        sleep(1.0)
        client = Client()
        while len(client) != n_cpus:
            sleep(0.5)
        lb_view = client.load_balanced_view()
    
        

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
                t = lb_view.apply_async(score_model, 
                                   self.model,
                                   n_points, 
                                   self.X_train,
                                   self.X_test,
                                   self.Y_train, 
                                   self.Y_test,
                                   self.averageType)
            
                
                # store jobs/results
                self.tasks_list.append(t) 

        # sort asyncronous results for plotting
        self.sort_results_PARALLEL_()  
        
        
        
        # stop cluster
        os.system("ipcluster stop")
        
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
            f1_train, f1_test, acc_train, acc_test, pre_train,\
            pre_test, rec_train, rec_test = np.array(scores).T

            # store values in lists
            self.f1_mean_train.append(np.mean(f1_train))
            self.f1_mean_test.append(np.mean(f1_test))

            self.acc_mean_train.append(np.mean(acc_train))
            self.acc_mean_test.append(np.mean(acc_test))

            self.pre_mean_train.append(np.mean(pre_train))
            self.pre_mean_test.append(np.mean(pre_test))

            self.rec_mean_train.append(np.mean(rec_train))
            self.rec_mean_test.append(np.mean(rec_test))

            self.training_size.append(n_samples)

        # convert lists into arrays
        self.f1_mean_train = np.array(self.f1_mean_train)
        self.f1_mean_test = np.array(self.f1_mean_test)
        self.acc_mean_train = np.array(self.acc_mean_train)
        self.acc_mean_test = np.array(self.acc_mean_test)
        self.pre_mean_train = np.array(self.pre_mean_train)
        self.pre_mean_test = np.array(self.pre_mean_test)
        self.rec_mean_train = np.array(self.rec_mean_train)
        self.rec_mean_test = np.array(self.rec_mean_test)
        self.training_size = np.array(self.training_size)

        # sort values based on increasing training set size (for plotting)
        sorted_inx = np.argsort(self.training_size)

        self.training_size = self.training_size[sorted_inx]
        self.f1_mean_train = self.f1_mean_train[sorted_inx]
        self.f1_mean_test = self.f1_mean_test[sorted_inx]
        self.acc_mean_train = self.acc_mean_train[sorted_inx]
        self.acc_mean_test = self.acc_mean_test[sorted_inx]
        self.pre_mean_train = self.pre_mean_train[sorted_inx]
        self.pre_mean_test = self.pre_mean_test[sorted_inx]
        self.rec_mean_train = self.rec_mean_train[sorted_inx]
        self.rec_mean_test = self.rec_mean_test[sorted_inx]

        
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
            
            self.log_metric_scores_()   
            self.log_metric_score_means_()
            self.training_size.append(n_points)
            
    
                            
    def log_metric_score_means_(self):
        '''Recrods the mean of the four metrics recording during training'''
        self.f1_mean_train.append(np.sum(self.f1_train)/len(self.f1_train))
        self.f1_mean_test.append(np.sum(self.f1_test)/len(self.f1_test))
        
        self.acc_mean_train.append(np.sum(self.acc_train)/len(self.acc_train))
        self.acc_mean_test.append(np.sum(self.acc_test)/len(self.acc_test))
        
        self.pre_mean_train.append(np.sum(self.pre_train)/len(self.pre_train))
        self.pre_mean_test.append(np.sum(self.pre_test)/len(self.pre_test))
        
        self.rec_mean_train.append(np.sum(self.rec_train)/len(self.rec_train))
        self.rec_mean_test.append(np.sum(self.rec_test)/len(self.rec_test))
        
        self.reinitialize_metric_lists_()
            
            
    def reinitialize_metric_lists_(self):
        '''Reinitializes metrics lists for training'''
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []
        
    def reinitialize_mean_metric_lists_(self):
        '''Reinitializes mean metrics lists for training'''
        self.f1_mean_train = []
        self.f1_mean_test = []
  
        self.acc_mean_train = []
        self.acc_mean_test = []
        
        self.pre_mean_train = []
        self.pre_mean_test = []
        
        self.rec_mean_train = []
        self.rec_mean_test = []

            
    def log_metric_scores_(self):
        '''Records the metric scores during each training iteration'''
        self.f1_train.append(f1_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.acc_train.append(accuracy_score( self.Y_train, self.y_pred_train) )

        self.pre_train.append(precision_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.rec_train.append(recall_score( self.Y_train, self.y_pred_train, average=self.averageType) )

        self.f1_test.append(f1_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.acc_test.append(accuracy_score(self.Y_test, self.y_pred_test))

        self.pre_test.append(precision_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.rec_test.append(recall_score(self.Y_test, self.y_pred_test,average=self.averageType))
            

    def plot_learning_curve(self,  train_point_type_color = 'o-', test_point_type_color='o-', image_name="Learning_Curve_Plot", save_image=False):
        '''Plots f1 and accuracy learning curves for a given model and data set'''
        
        fig = plt.figure(figsize = (17,12))
        # plot f1 score learning curve
        fig.add_subplot(221)   # left
        plt.title("F1-Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.f1_mean_train, train_point_type_color, label="Train")
        plt.plot(self.training_size, self.f1_mean_test, test_point_type_color, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1-Score")
        plt.grid()
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(222)   # right 
        plt.title("Accuracy vs. Number of Training Samples")
        plt.plot(self.training_size, self.acc_mean_train, train_point_type_color, label="Train")
        plt.plot(self.training_size, self.acc_mean_test, test_point_type_color, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend(loc=4);
        
        # plot precision learning curve
        fig.add_subplot(223)   # left
        plt.title("Precision Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.pre_mean_train, train_point_type_color, label="Train")
        plt.plot(self.training_size, self.pre_mean_test, test_point_type_color, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Precision")
        plt.ylim(min(self.pre_mean_test), max(self.pre_mean_train) + 0.05)
        plt.grid()
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(224)   # right 
        plt.title("Recall vs. Number of Training Samples")
        plt.plot(self.training_size, self.rec_mean_train, train_point_type_color, label="Train")
        plt.plot(self.training_size, self.rec_mean_test, test_point_type_color, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Recall")
        plt.grid()
        plt.legend(loc=4);
        
        if  save_image==True:
            fig.savefig('./{}.png'.format(image_name))
