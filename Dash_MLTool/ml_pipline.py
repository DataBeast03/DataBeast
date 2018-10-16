# -*- coding: utf-8 -*-
import pickle
import os
import os.path
import copy
import time
import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from dash.dependencies import Input, Output
from flask_caching import Cache

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from  cross_val_tool_with_PARALLEL import cross_validation_with_PARALLEL as cross_validation

from ETL_functions import * 


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, 
    external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions']=True

# dropdown menue options 
all_options = {
        'Classification': ["Logistic Regression", "Random Forest"],
        'Regression': ["Linear Regression", "Gradient Boosted Trees"]
    }


# assigne possible models for each task type 
model_options = {
        'Classification': 
            {"Logistic Regression": LogisticRegression, 
            "Random Forest": RandomForestClassifier},
        'Regression': 
            {"Linear Regression": LinearRegression, 
            "Gradient Boosted Trees": GradientBoostingRegressor},
    }

# dir where cv results are stored for caching
cv_results_path = "/Users/databeast03/DataBeast/Dash_MLTool/cv_results/"


#### ETL Pipeline ####
# execute ETL pipeline 
regression_data = "raw_data/googleplaystore.csv"
classification_data = "raw_data/churn.csv"
print("loadinging data...")
load_process_save_regression(regression_data)
load_process_save_classification(classification_data)


# To Do create function for Grid Search pipline

# To Do call cross_validation to get metric results for plotting  


#### app layout ######

app.layout = html.Div([
    # dcc.RadioItems(
    #     id='task-type-dropdown',
    #     options=[{'label': k, 'value': k} for k in all_options.keys()],
    #     value='Classification'
    # ),

    # display task type
    html.Div([
        dcc.Dropdown(
            id='task-type-dropdown',
            options=[{'label': "Classification", 'value': "Classification"},
                     {'label': "Regression", 'value': "Regression"}],
            value='Regression'
        )


    ],
    style={'width': '75%', 'display': 'inline-block', 'fontSize': 20}),



    # dispaly model type
    html.Div([
        dcc.RadioItems(id='models-dropdown',
                style={'width': '50%', 'display': 'inline-block', 'fontSize': 20}
    )]),

    html.Hr(),
        # display selections
    #html.Div(id='display-selected-values'), 


    dcc.Markdown('''## Scalable Machine Learning Pipeline'''),


    html.Div([
    html.Div("""About the Backend""", 
        style={'color': 'black', 'fontSize': 25}),
    html.P('''
        The backend is feeding data into an ETL pipeline that pre-processes the data and saves it to file. 
        A custom built data tool handels all of the cross validation: spliting the data, fitting the model with K-fold, and scoring the model agaisnt several metrics. 
        This tool can train and score both classifcation and regression models. 
        The tool is also built to be scalble, simple increase the number of cpus. 
        Finally, this entire pipeline is being served on an AWS EC2 instance. 
        ''', 
        className='my-class', 
        id='p-backend', 
        style={'color': 'black', 'fontSize': 18}),

    html.Hr(),

    html.Div("""About the Frontend""", 
        style={'color': 'black', 'fontSize': 25}),
    html.P('''
        The frontend is displaying the model's results on an interactive dashboard. 
        The devloper can select between Regression and Classification tasks.  
        They can then select which models to build.
        Then they can compare the performance of those models across several metrics. 
        ''', 
        className='my-class', 
        id='p-frontend', 
        style={'color': 'black', 'fontSize': 18}),

    html.Hr(),

    html.Div("""About the Data""", 
        style={'color': 'black', 'fontSize': 25}),

    html.P('''
        The Regression data is from the Google Play Store. Here we are are predicting app ratings, which vary from 1 to 5. 
        The Classification data is a generic churn data set. The labels are binary and unbalanced, with non-churners appearing about 5x more than churners. 
        ''', 
        className='my-class', 
        id='p-data', 
        style={'color': 'black', 'fontSize': 18}),


    # html.P('''The dropdown menu allows you to select which of Google's building's energy behavior to analyze. 
    #     '''

    #     , 
    #     className='my-class', 
    #     id='my-p-element-2', 
    #     style={'color': 'black', 'fontSize': 18}),

    ]),

    #dcc.Markdown('''![](http://www.casinonewsdaily.com/wp-content/uploads/2015/10/google-play-logo-300x190.jpg)'''),

    # html.Hr(),

    # html.Div(children='Dash: A web application framework for Python.', 
    #     style={'textAlign': 'center'}),



    #### Regression #### 
    # display two 
    html.Div([
        dcc.Graph(id='curve-one'),
    ], style={'display': 'inline-block', 'width': '100%', 'height':'50%'}),

    # display two 
    html.Div([
        dcc.Graph(id='curve-two'),
    ], style={'display': 'inline-block', 'width': '100%', 'height':'50%'}),

        # display two 
    html.Div([
        dcc.Graph(id='curve-three'),
    ], style={'display': 'inline-block', 'width': '100%', 'height':'50%'}),

    # display two 
    html.Div([
        dcc.Graph(id='curve-four'),
    ], style={'display': 'inline-block', 'width': '100%', 'height':'50%'}),


    #### Classification ####



], style={'columnCount': 3, 'marginBottom': 10, 'marginTop': 10})



@app.callback(
    dash.dependencies.Output('models-dropdown', 'options'),
    [dash.dependencies.Input('task-type-dropdown', 'value')])
def set_model_options(selected_task_type):
    '''Assoicates which models are options when a specific task type 
       is selected. '''
    return [{'label': i, 'value': i} for i in all_options[selected_task_type]]


@app.callback(
    dash.dependencies.Output('models-dropdown', 'value'),
    [dash.dependencies.Input('models-dropdown', 'options')])
def set_model_value(available_options):
    ''''returns the selected model type'''
    return available_options[0]['value']


@app.callback(
    dash.dependencies.Output('display-selected-values', 'children'),
    [dash.dependencies.Input('task-type-dropdown', 'value'),
     dash.dependencies.Input('models-dropdown', 'value')])
def set_display_children(selected_task_type, selected_model):
    '''Dispalys both the selected task type and model type'''
    return '''{} task has been assigned {}'''.format(
        selected_task_type, selected_model
    )


#### Data Processing + Modeling #####

def cross_validate_model(task_type, model_type):

    results = "_".join((cv_results_path , task_type, model_type, ".pkl"))

    if os.path.exists(results):
        return pickle.load(open(results, "rb"))

    else:
        model = model_options[task_type][model_type]
        model = model()

        if task_type=="Classification":
            data_path = "/Users/databeast03/DataBeast/Dash_MLTool/ETL_data/churn.pkl"
            X_data, Y_data = pickle.load(open(data_path, 'rb'))
            init_chunk_size=100
        else:
            data_path = "/Users/databeast03/DataBeast/Dash_MLTool/ETL_data/googleAppStore_data.pkl"
            X_data, Y_data = pickle.load(open(data_path, 'rb'))
            init_chunk_size=1000

        cv=\
        cross_validation(model, 
                         X_data, 
                         Y_data,
                         n_splits=3, 
                         init_chunk_size=init_chunk_size, 
                         chunk_spacings=100, 
                         learning_type=task_type)
        cv.train_for_learning_curve()

        pickle.dump(cv, open(results, "wb"))

        return cv



#### learning curves #####

@app.callback(
dash.dependencies.Output('curve-one', 'figure'),
[dash.dependencies.Input('task-type-dropdown', 'value'),
 dash.dependencies.Input('models-dropdown', 'value')])
def create_curve_one(task_type, model_type):

    cv = cross_validate_model(task_type, model_type)

    if task_type=="Classification":
        y_train = cv.f1_mean_train
        y_test = cv.f1_mean_test
        label="f1"
        title = "F1 Score"
    else:
        y_train = cv.rmse_mean_train
        y_test=cv.rmse_mean_test
        label="RMSE"
        title = "Root Mean Square Error"

    return {
        'data': [

        go.Scatter(
            x=cv.training_size,
            y=y_train,
            mode='lines+markers',
            name="{} Train".format(label)

        ),
        go.Scatter(
            x=cv.training_size,
            y=y_test,
            mode='lines+markers',
            name="{} Test".format(label)

        )
        ],


        'layout': {
            'title':"{} vs. Training Set Size".format(title),
            'height': 525,
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Training Set Size', 'showgrid': True},
            "yaxis":{'title': label},
            }
        }



@app.callback(
dash.dependencies.Output('curve-two', 'figure'),
[dash.dependencies.Input('task-type-dropdown', 'value'),
 dash.dependencies.Input('models-dropdown', 'value')])
def create_curve_two(task_type, model_type):

    cv = cross_validate_model(task_type, model_type)

    if task_type=="Classification":
        y_train=cv.pre_mean_train
        y_test=cv.pre_mean_test
        label="f1"
        title="Precsion Score"
    else:
        y_train=cv.mse_mean_train
        y_test=cv.mse_mean_test
        label="RMSE"
        title="Mean Square Error"

    return {
        'data': [

        go.Scatter(
            x=cv.training_size,
            y=y_train,
            mode='lines+markers',
            name="{} Train".format(label)

        ),
        go.Scatter(
            x=cv.training_size,
            y=y_test,
            mode='lines+markers',
            name="{} Test".format(label)

        )
        ],


        'layout': {
            'title':"{} vs. Training Set Size".format(title),
            'height': 525,
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Training Set Size', 'showgrid': True},
            "yaxis":{'title': label},
            }
        }



@app.callback(
dash.dependencies.Output('curve-three', 'figure'),
[dash.dependencies.Input('task-type-dropdown', 'value'),
 dash.dependencies.Input('models-dropdown', 'value')])
def create_curve_three(task_type, model_type):

    cv = cross_validate_model(task_type, model_type)

    if task_type=="Classification":
        y_train=cv.acc_mean_train
        y_test=cv.acc_mean_test
        label="f1"
        title="Accuracy Score"
    else:
        y_train=cv.mae_mean_train
        y_test=cv.mae_mean_test
        label="MAE"
        title="Mean Absolute Error"

    return {
        'data': [

        go.Scatter(
            x=cv.training_size,
            y=y_train,
            mode='lines+markers',
            name="{} Train".format(label)

        ),
        go.Scatter(
            x=cv.training_size,
            y=y_test,
            mode='lines+markers',
            name="{} Test".format(label)

        )
        ],


        'layout': {
            'title':"{} vs. Training Set Size".format(title),
            'height': 525,
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Training Set Size', 'showgrid': True},
            "yaxis":{'title': label},
            }
        }


@app.callback(
dash.dependencies.Output('curve-four', 'figure'),
[dash.dependencies.Input('task-type-dropdown', 'value'),
 dash.dependencies.Input('models-dropdown', 'value')])
def create_curve_four(task_type, model_type):

    cv = cross_validate_model(task_type, model_type)

    if task_type=="Classification":
        y_train=cv.rec_mean_train
        y_test=cv.rec_mean_test
        label="Recall"
        title="Recall Score"
    else:
        y_train=cv.r2_mean_train
        y_test=cv.r2_mean_test
        label="R2"
        title="R2 Score"

    return {
        'data': [

        go.Scatter(
            x=cv.training_size,
            y=y_train,
            mode='lines+markers',
            name="{} Train".format(label)

        ),
        go.Scatter(
            x=cv.training_size,
            y=y_test,
            mode='lines+markers',
            name="{} Test".format(label)

        )
        ],


        'layout': {
            'title':"{} vs. Training Set Size".format(title),
            'height': 525,
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Training Set Size', 'showgrid': True},
            "yaxis":{'title': label},
            }
        }


if __name__ == '__main__':
    app.run_server(debug=True)