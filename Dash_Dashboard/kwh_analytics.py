# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from flask import Flask


# load data
def load_df(dash_ready_data_path):
    return pd.read_csv(dash_ready_data_path)

def get_unique_buildings():
    uni_buildings = []
    for col in df.columns:
        if "building" in col:
            building = "_".join(col.split("_")[:2])
            uni_buildings.append(building)
    return np.unique(uni_buildings)    

def load_forecast_model( model_path):
    return pickle.load(open(model_path, 'rb'))


def drop_underscore(building):
    return " ".join((building.split("_")[0], building.split("_")[1]))

dash_ready_data_path = "dash_ready_data.csv"
consumption_model_filename = "trained_model_consumption_forecast.pkl"
production_model_filename = "trained_model_production_forecast.pkl"

# data manipulations
df = load_df(dash_ready_data_path)
unique_buldings = get_unique_buildings()

# load models
production_model = load_forecast_model(production_model_filename)
consumption_model = load_forecast_model(consumption_model_filename)

month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']


### ------


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#server = Flask(__name__)
app = dash.Dash(__name__,
    sharing=True, 
    #server=server,
	external_stylesheets=external_stylesheets)

application = app.server

#available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([

    # all diplay 
    html.Div([

    	dcc.Markdown(''' ![](https://www.energy.gov/sites/prod/files/styles/borealis_photo_gallery_large_respondmedium/public/kwh%20logo.png?itok=YlDw0inB)'''),


        html.Div(children='''
            Clean Energy Production and Consumption accross all Google buildings. 
        ''', style={'fontsize':30}),


        # display one
        html.Div([
            dcc.Dropdown(
                id='crossfilter-column',
                options=[{'label': i, 'value': i} for i in unique_buldings],
                value='building_1'
            )


        ],
        style={'width': '49%', 'display': 'inline-block'})    

    ]),


    # display two 
    html.Div([
        dcc.Graph(id='time-series-one'),
        dcc.Graph(id='time-series-two'),
    ], style={'display': 'inline-block', 'width': '100%', 'height':'50%'}),

    dcc.Markdown('''![](https://qph.fs.quoracdn.net/main-qimg-1ea9111b0b681a15e0713ca7a6896985)'''),

    # summery table with aggs
    html.Div(
        html.Table(id='pro-con-summery-table'),
        className='two columns',
        style={'display': 'inline-block', 'width': '100%', 'height':'100%', 'fontSize': 20}
    ),



    html.Div(id='pro-con-text-summery',
        style={'color': 'black', 'fontSize': 20}),



    html.Div(
        html.Table(id='cost-profit-summery-table'),
        className='two columns',
        style={'display': 'inline-block', 'width': '100%', 'height':'100%', 'fontSize': 20}
    ),

    #dcc.Markdown(id='cost-profit-text-summery'), 
    html.Div(id='cost-profit-text-summery',
        style={'display': 'inline-block', 'color': 'black', 'fontSize': 20}),

	html.Div([
	    html.Div("""ABOUT THIS DASHBOARD""", 
	    	style={'color': 'black', 'fontSize': 30}),
	    html.P('''
	    	This is a simulation in how a dashboard and machine learning 
	    	can be combined to provide a valuable service for clients. 
	    	Using the corrected_kwh data from the data challenge and a 
	    	single household's electrical energy consumption over the course of 4 years (UCI ML Database).
	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''The dropdown menu allows you to select which of Google's building's energy behavior to analyze. 
	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-2', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''
	    	Energy Production vs Consumption chart shows how much electrical energy 
	    	was produced and consumed monthly. 
	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-3', 
	    	style={'color': 'black', 'fontSize': 18}),

	    html.P('''
	    	Cost Cut vs Profit Made From Energy Production chart shows one of two things: 
	    	negative values show how much each build's energy bill has been reduced to, 
	    	this is the final monthly bill after using up all solar energy produced. 
	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-4', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''
	    	Positive values indicate a profit that the build made by selling excess energy 
	    	into the grid.  

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-5', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''
	    	Each chart provides a forecasted value(s). 
	    	These values are generated by a ML model that is trained on historical data from the client. 
	    	We provide 3 forecasted values: next month's solar energy production, next month's energy consumption, 
	    	and next month's bill (forecasted cost cut).  

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-6', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''
	    	Technology:  

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-7', 
	    	style={'color': 'black', 'fontSize': 26}),

	    html.P('''

	    	This dashboard was build using Dash by Plot.ly
 

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-8', 
	    	style={'color': 'black', 'fontSize': 18}),

	    html.P('''
	    	The ML model is from Sci-kit Learn's library. 

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-9', 
	    	style={'color': 'black', 'fontSize': 18}),


	    html.P('''
	    	This app is being served on an AWS EC2 instance.

	    	'''

	    	, 
	    	className='my-class', 
	    	id='my-p-element-10', 
	    	style={'color': 'black', 'fontSize': 18}),	    


	], style={'marginBottom': 50, 'marginTop': 60, }),


    ],style={'columnCount': 2, 'marginBottom': 50, 'marginTop': 25})





def create_time_series_one(data_pro, pro_forecast, data_con, con_forecast, title):
    return {
        'data': [

        go.Scatter(
            x=month_labels,
            y=data_pro,
            mode='lines+markers',
            name="Energy Produced"
        ),

        go.Scatter(
            x=month_labels,
            y=data_con,
            mode='lines+markers', 
            name="Energy Consumed",
        ),

        go.Scatter(
            x=month_labels[-2:],
            y=[data_pro[-1], pro_forecast[0]],
            mode='lines+markers', 
            name="Production Forecast",
            line = dict(
                #color = ('rgb(205, 12, 24)'),
                width = 2,
                dash = 'dash') 

        ),

        go.Scatter(
            x=month_labels[-2:],
            y=[data_con[-1], con_forecast[0]],
            mode='lines+markers', 
            name="Consumption Forecast",
            line = dict(
                #color = ('rgb(205, 12, 24)'),
                width = 2,
                dash = 'dash') 


        )],


        'layout': {
            'title':title,
            'height': 525,
            #'margin': {'l': 40, 'b': 20, 't': 50, 'r': 10},
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Months', 'showgrid': True},
            "yaxis":{'title': 'Energy (kWh)'},
        }
    }


def create_time_series_two(cost_profit, cost_profit_forecast, title):
    return {
        'data': [

        go.Scatter(
            x=month_labels,
            y=cost_profit,
            mode='lines+markers',
            name="Cost or Profit"
        ),



        go.Scatter(
            x=month_labels,
            y=[0 for i in range(1,13)],
            mode='lines+markers',
            name="Zero Cost or Profit",
            line = dict(
                color = ('red'),
                width = 2,
                dash = 'dash') 


        ),
        

        go.Scatter(
            x=month_labels[-2:],
            y=[cost_profit[-1], cost_profit_forecast[0]],
            mode='lines+markers', 
            name="Cost or Profit Forecast",
            line = dict(
                #color = ('rgb(205, 12, 24)'),
                width = 2,
                dash = 'dash') 


        )],


        'layout': {
            'title':title,
            'height': 525,
            #'margin': {'l': 40, 'b': 20, 't': 50, 'r': 10},
            "legend":{'x': 1, 'y': 0},
            "xaxis":{'title': 'Months', 'showgrid': True},
            "yaxis":{'title': 'Dollars ($USD)'},


            # 'annotations': [{
            #     'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
            #     'xref': 'paper', 'yref': 'paper', 'showarrow': False,
            #     'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
            #     'text': title, "font":14
            # }]
        }
    }








  
def get_data(column_name):
    pro_col = "".join((column_name, "_production" ))
    con_col = "".join((column_name, "_consumption" ))

    data_pro = df[pro_col].values[:-1]
    data_con = df[con_col].values[:-1]

    con_to_forecast = np.array(data_con[-2])
    con_to_forecast.reshape((-1,1))

    pro_forecast = production_model.predict([data_pro])
    con_forecast = consumption_model.predict(con_to_forecast)

    return data_pro, pro_forecast, data_con, con_forecast

def calc_cost_profit_margin(energy_pro, energy_con):

    kwh_col = "kwh_cost"
    # in principle, we don't know the 12th monthly average cost for kwh
    # sicne we are forecasting for 12th month
    # we wil assume that average monthly cost will the same from 11th to 12th month
    kwh_cost = df[kwh_col].values[:-1]

    # frist 11  months: cost of consumption
    cons_cost = energy_con*kwh_cost
    # first 11 months: potential profit generated from panels 
    production_value_generated = energy_pro*kwh_cost
    # first 11 months: difference between cost of con and value gen form panels 
    cost_profit = production_value_generated - cons_cost


    return cost_profit



@app.callback(
    dash.dependencies.Output('pro-con-summery-table', component_property='children'),
    [dash.dependencies.Input('crossfilter-column', 'value')])    
def generate_table(column_name):

    columns = ["Average Monthly Production.", "Average Monthly Consumption"]
    data_pro, pro_forecast, data_con, con_forecast = get_data(column_name)

    ave_monthly_pro = np.mean(data_pro)
    ave_monthly_con = np.mean(data_con)


    table_values = [ave_monthly_pro,
                    ave_monthly_con]


    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in columns])] +

        # Body
        [html.Tr([html.Td("{:.3} kwh".format(val) ) for val in table_values])]
    )



@app.callback(
    dash.dependencies.Output('cost-profit-summery-table', component_property='children'),
    [dash.dependencies.Input('crossfilter-column', 'value')])    
def generate_table(column_name):

    columns = ["Cost Cut", "Forecast Cost Cut"]
    data_pro, pro_forecast, data_con, con_forecast = get_data(column_name)
    cost_profit = calc_cost_profit_margin(data_pro, data_con)

    ave_monthly_pro = np.mean(data_pro)
    ave_monthly_con = np.mean(data_con)
    total_cost_profit = np.abs(np.sum(cost_profit))

    kwh_col = "kwh_cost"
    # in principle, we don't know the 12th monthly average cost for kwh
    # sicne we are forecasting for 12th month
    # we wil assume that average monthly cost will the same from 11th to 12th month
    kwh_cost = df[kwh_col].values[:-1]

    # repeat calculation for forecast 
    cons_cost_forecast = con_forecast*kwh_cost[-1]
    production_value_forecast = pro_forecast*kwh_cost[-1]
    cost_profit_forecast = production_value_forecast - cons_cost_forecast
    forecast_cost_profit = np.abs(np.sum(cost_profit_forecast))


    table_values = [total_cost_profit,
                    forecast_cost_profit]


    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in columns])] +

        # Body
        [html.Tr([html.Td("${:.3}".format(val) ) for val in table_values])]
    )


@app.callback(
    dash.dependencies.Output('pro-con-text-summery', component_property="children"),
    [dash.dependencies.Input('crossfilter-column', 'value')])
def update_pro_con_text_summery(column_name):

    text = \
    '''Average Energy Production and Consumption for {}'''.format(column_name)

    return text


@app.callback(
    dash.dependencies.Output('cost-profit-text-summery', component_property="children"),
    [dash.dependencies.Input('crossfilter-column', 'value')])
def update_cost_profit_text_summery(column_name):

    text = \
    '''Total Cost Cutting this year for {}

    By using solar, Google has reduced its electricity bill
    by the amount under COST CUT so for this year.
    Based on the company's previous levels of energy consumption 
    and production from the installed solar panels, our AI models forecast
    that your going to cut cost by the amount under FORECASTED COST CUT.
    '''.format(column_name)

    return text


@app.callback(
    dash.dependencies.Output('time-series-one', 'figure'),
    [dash.dependencies.Input('crossfilter-column', 'value')])
def update_timeseries_one(column_name):
    data_pro, pro_forecast, data_con, con_forecast = get_data( column_name)

    title_text = "Energy Production vs Consumption"
    title = '<b>{}</b><br>{}'.format(title_text, column_name)
    return create_time_series_one(data_pro, pro_forecast, data_con, con_forecast, title)


@app.callback(
    dash.dependencies.Output('time-series-two', 'figure'),
    [dash.dependencies.Input('crossfilter-column', 'value')])
def update_timeseries_two(column_name):
    data_pro, pro_forecast, data_con, con_forecast = get_data(column_name)

    cost_profit = calc_cost_profit_margin(data_pro, data_con)


    kwh_col = "kwh_cost"
    # in principle, we don't know the 12th monthly average cost for kwh
    # sicne we are forecasting for 12th month
    # we wil assume that average monthly cost will the same from 11th to 12th month
    kwh_cost = df[kwh_col].values[:-1]

    # repeat calculation for forecast 
    con_cost_forecast = con_forecast*kwh_cost[-1]
    production_value_forecast = pro_forecast*kwh_cost
    cost_profit_forecast = production_value_forecast - con_cost_forecast



    title_text = "Cost Cut or Profit Made from Energy Production "
    title = '<b>{}</b><br>{}'.format(title_text, column_name)
    return create_time_series_two(cost_profit, cost_profit_forecast, title)



if __name__ == '__main__':
    application.run(debug=True,  port=8080)