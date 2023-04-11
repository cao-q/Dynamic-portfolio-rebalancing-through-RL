# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import pandas as pd
from bokeh.io import curdoc, output_notebook, show
from bokeh.plotting import figure, output_file, save

#%%
# Change portfolio info accordingly
lagged = False

if not lagged:
    df = pd.read_csv('data/rl/index/daily_nav.csv', parse_dates=['Date'])
    passive_df = pd.read_csv('data/rl/index/passive_daily_nav.csv', parse_dates=['Date'])
    df_list = [df, passive_df]
else:
    df = pd.read_csv('data/rl/index/lagged/daily_nav.csv', parse_dates=['Date'])
    passive_df = pd.read_csv('data/rl/index/passive_daily_nav.csv', parse_dates=['Date'])
    df_list = [df, passive_df]

def plot_daily_nav(df_list: list, x_col='Date'):
    """**First dataframe must be the Portfolio with switching.** 
    """
    p = figure(title="Portfolio Net Asset Comparison", x_axis_type='datetime',
                background_fill_color="#fafafa")

    p.line(df_list[0][x_col], df_list[0]['Net'].values.tolist(), legend="RL rebalanced",
        line_color="black")

    p.line(df_list[1][x_col], df_list[1]["^BVSP"].values.tolist(), legend="^BVSP",
        line_color="red")

    p.line(df_list[1][x_col], df_list[1]["^TWII"].values.tolist(), legend="^TWII", 
        line_color="orange")

    p.line(df_list[1][x_col], df_list[1]["^IXIC"].values.tolist(), legend="^IXIC",
        line_color="olivedrab")

    p.legend.location = "top_left"
    output_file('data/rl/index/daily_nav_comp.html')
    save(p)
    show(p)

plot_daily_nav(df_list)
