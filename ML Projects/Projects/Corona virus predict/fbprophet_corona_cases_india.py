# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:35:27 2020

@author: windows 10
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from fbprophet import Prophet

init_notebook_mode()

data = pd.read_csv('corona_india.csv')
data = data.iloc[15:,0:2]


end = datetime.datetime.now() - datetime.timedelta(2)
date_index = pd.date_range('2020-03-01', end)

fig = px.area(data, x=date_index, y='Total cases' )
#plot(fig)


df_prophet = data.rename(columns={"Date": "ds", "Total cases": "y"})
df_prophet.tail()

from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot

m = Prophet(
    changepoint_prior_scale=0.22, # increasing it will make the trend more flexible
    changepoint_range=0.99, # place potential changepoints in the first 95% of the time series
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
)

m.fit(df_prophet)

# predict for next 31 days
future = m.make_future_dataframe(periods=31)
coronavirus = m.predict(future)


coronavirus[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(35)
