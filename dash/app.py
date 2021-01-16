#!/usr/bin/env python
# coding: utf-8

# In[4]:
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate

# In[2]:

# In[9]:
# Read Data
merge_df_tsla = pd.read_csv("TSLA_price_merge_df.csv")
merge_df = pd.read_csv("AAPL_price_merge_df.csv")
aapl_strat = pd.read_csv("aapl_strat.csv")
tsla_strat = pd.read_csv("tsla_strat.csv")


# Functions
# days = 253-1 since day 0 has no returns
def sharpe(ema, ticker, rf=0, days=252):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}'].pct_change()
        sharpe_ratio = np.sqrt(days) * (df.mean() - rf) / df.std()
        return '{:.2f}'.format(sharpe_ratio)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}'].pct_change()
        sharpe_ratio = np.sqrt(days) * (df.mean() - rf) / df.std()
        return '{:.2f}'.format(sharpe_ratio)


def total_returns(ema, ticker):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}']
        tot_returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(tot_returns * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}']
        tot_returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(tot_returns * 100)


def max_drawdown(ema, ticker):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}']
        max_dd = (df.min() - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(max_dd * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}']
        max_dd = (df.min() - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(max_dd * 100)


def num_buy_trades(ema, ticker):
    if ticker == '$AAPL':
        return len(aapl_strat.loc[aapl_strat[f'Action EMA {ema}'].str.contains('BUY')])
    elif ticker == '$TSLA':
        return len(tsla_strat.loc[tsla_strat[f'Action EMA {ema}'].str.contains('BUY')])


def num_sell_trades(ema, ticker):
    if ticker == '$AAPL':
        return len(aapl_strat.loc[aapl_strat[f'Action EMA {ema}'].str.contains('SELL')])
    elif ticker == '$TSLA':
        return len(tsla_strat.loc[tsla_strat[f'Action EMA {ema}'].str.contains('SELL')])


def volatility(ema, ticker, days=252):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}'].pct_change()
        daily_vol = np.log(1 + df).std()
        annualised_vol = daily_vol * np.sqrt(days)
        return '{:.2f}%'.format(annualised_vol * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}'].pct_change()
        daily_vol = np.log(1 + df).std()
        annualised_vol = daily_vol * np.sqrt(days)
        return '{:.2f}%'.format(annualised_vol * 100)


def get_table(df, ema, ticker):
    df.columns = ['Date', 'Portfolio Value ($USD)', 'Net Shares', 'Net Cash', 'Trade Signal',
                  'Pre-market P/L ($USD)', 'Trade Session P/L ($USD)', 'Total Day P/L ($USD)',
                  'Cumulative P/L ($USD)']
    df.drop(['Pre-market P/L ($USD)', 'Trade Session P/L ($USD)'], axis=1)
    # df["Date"] = df["Date"].apply(lambda x: x.date())
    data = df.to_dict('rows')

    columns = [{
        'id': df.columns[0],
        'name': 'Date',
        'type': 'datetime'
    }, {
        'id': df.columns[4],
        'name': 'Trade Signal',
        'type': 'text'
    }, {
        'id': df.columns[1],
        'name': 'Portfolio Value ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[2],
        'name': 'Net Shares',
        'type': 'numeric',
    }, {
        'id': df.columns[3],
        'name': 'Net Cash',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[7],
        'name': 'Daily P/L ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[8],
        'name': 'Cumulative P/L ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }]

    package = html.Div([
                        dbc.Col(html.H4(f'EMA {ema} Trade Log ({ticker})')),
                        dbc.Col(dt.DataTable(data=data, columns=columns, id='table',
                                             fixed_rows={'headers': True},
                                             style_table={'height': 350},
                                             style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                           'color': 'white'},

                                             style_cell_conditional=[
                                                 {
                                                     'if': {'column_id': c},
                                                     'textAlign': 'left'
                                                 } for c in ['Date', 'Region']
                                             ],

                                             style_data_conditional=[
                                                 {
                                                     'if': {'row_index': 'odd'},
                                                     'backgroundColor': 'rgb(248, 248, 248)'
                                                 }
                                             ],
                                             style_cell={
                                                 'whiteSpace': 'normal', 'textAlign': 'left',
                                                 'height': 'auto'}))
                        ])

    return package


external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']

navbar = dbc.Nav(className="navbar-nav mr-auto", children=[
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-apple"), "   $AAPL Stocktwits"],
                             href="https://stocktwits.com/symbol/AAPL", target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-car"), "  $TSLA Stocktwits"],
                             href="https://stocktwits.com/symbol/TSLA", target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  Linkedin"],
                             href="https://www.linkedin.com/in/jay-lin-jiele-02a5a114a/", target="_blank"),
    ])
])

# Dash App
# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([html.H2(children="StockTwits Sentiments Momentum Trading Dashboard",
                         style={'textAlign': 'centre',
                                'color': 'white'})], md=10),
        dbc.Col(md=2, children=[navbar])
    ], className='navbar navbar-expand-lg navbar-dark bg-primary'),
    html.Br(), html.Br(),
    dbc.Row([
        dbc.Col(xs=12, sm=12, md=12, lg=3, xl=3, children=[
            dbc.Col(html.H3(children='Select Bull/Bear Ratio EMA to view different portfolio performance: ▼'),
                    style={'textAlign': 'left'}),
            dbc.Col(dcc.Slider(
                id='slider',
                min=5,
                max=20,
                value=5,
                marks={i: str(i) for i in [5, 6, 7, 8, 9, 10, 15, 20]},
                step=None, )),
            html.Br(),
            dbc.Col(html.Div(id="output-panel")),
            html.Br(), html.Br(),
            dbc.Col(dbc.Card(body=True, className="card bg-light mb-3", children=[
                html.Div("About This Dashboard", className="card-header"),
                html.Div(className="card-body", children=[
                    html.P(children=["""This experiment aims to mine $TSLA and $AAPL sentiments 
                    on StockTwits and derive trading signals based on their pre-market sentiments moving 
                    averages. StockTwits is a social media platform for retail traders to share their 
                    speculations and sentiments regarding any stock.""",
                                     html.Br(),
                                     html.Br(),
                                     """This dashboard was built on Plotly Dash to make it interactive. You can
                                     tweak the EMAs from the slider bar above to view the different backtest
                                      performances.""",
                                     html.Br(), html.Br(),
                                     """Thanks for viewing, you may find out more about the project through my 
                                     article on """,
                                     html.A("Medium.",
                                            href='https://jayljl.medium.com/mining-stocktwits-retail-sentiments-'
                                                 'for-momentum-trading-4594a91833b4',
                                            target="_blank",
                                            style={'color': 'blue'})

                                     ], className="card-text")
                ])
            ])
                    )
        ]),

        dbc.Col(lg=9, xl=9, children=[
            dbc.Col(html.H3("Portfolio Backtest Performance and Signals"), width={"size": 7, "offset": 3},
                    style={'textAlign': 'center'}),
            dbc.Tabs(className="nav nav-pills", id='yaxis-column',
                     children=[
                         dbc.Tab(label='$TSLA', tab_id='$TSLA'),
                         dbc.Tab(label='$AAPL', tab_id='$AAPL')
                     ],
                     active_tab="$TSLA"),
            dcc.Graph(id='Portfolio-chart'),
            dcc.Graph(id='buy-sell-chart'),
            dcc.Graph(id='bull-bear-chart'),
            dbc.Col(html.Div(id="data-table"))
        ])
    ])
])


# Function to render Portfolio Chart
@app.callback(
    Output('Portfolio-chart', 'figure'),
    Input('yaxis-column', 'active_tab'),
    Input('slider', 'value'))  # Slider value will be EMA 5,6,7,8,9,10,15,20
def update_graph(yaxis_column_name, ema):
    # TSLA / Portfolio

    if yaxis_column_name == "$TSLA":
        data = [
            go.Scatter(x=tsla_strat[f'Date EMA {ema}'], y=(tsla_strat[f"Portfolio Value EMA {ema}"]),
                       mode='lines', name="Portfolio Value",
                       line=dict(color='red', width=0.5), fill='tozeroy')
        ]
        fig = go.Figure(data=data)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(
            title=f"<b>Portfolio Performance ($TSLA) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="red", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig

    # AAPL / Porfolio
    elif yaxis_column_name == "$AAPL":
        data = [
            go.Scatter(x=aapl_strat[f'Date EMA {ema}'], y=(aapl_strat[f"Portfolio Value EMA {ema}"]),
                       mode='lines', name="Portfolio Value",
                       line=dict(color='black', width=0.5), fill='tozeroy')
        ]
        fig = go.Figure(data=data)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(
            title=f"<b>Portfolio Performance ($AAPL) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                ticksuffix=' USD',
                fixedrange=True,
                range=[6000, 18000]
            ))
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig


# Function to render Buy/ Sell Chart Callback
@app.callback(
    Output('buy-sell-chart', 'figure'),
    Input('yaxis-column', 'active_tab'),
    Input('slider', 'value'))  # Slider value will be EMA 5,6,7,8,9,10,15,20
def update_graph(yaxis_column_name, ema):
    # TSLA / Signal
    if yaxis_column_name == "$TSLA":
        data = [
            go.Scatter(x=tsla_strat[f'Date EMA {ema}'], y=(tsla_strat[f"Adjusted Close EMA {ema}"]),
                       mode='lines', name="TSLA Closing Price",
                       line=dict(color='#86d3e3', width=2)),
            go.Scatter(x=tsla_strat.loc[tsla_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Date EMA {ema}"],
                       y=(tsla_strat.loc[
                           tsla_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Buy",
                       marker=dict(symbol='triangle-up', color="green", size=7)),
            go.Scatter(x=tsla_strat.loc[tsla_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Date EMA {ema}"],
                       y=(tsla_strat.loc[
                           tsla_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Sell",
                       marker=dict(symbol='triangle-down', color="red", size=7))
        ]
        fig = go.Figure(data=data)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(
            title=f"<b>Long/ Short Signal ($TSLA) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="red", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig

        # AAPL / Signal
    elif yaxis_column_name == "$AAPL":
        data = [
            go.Scatter(x=aapl_strat[f'Date EMA {ema}'], y=(aapl_strat[f"Adjusted Close EMA {ema}"]),
                       mode='lines', name="AAPL Closing Price",
                       line=dict(color='#86d3e3', width=2)),
            go.Scatter(x=aapl_strat.loc[aapl_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Date EMA {ema}"],
                       y=(aapl_strat.loc[
                           aapl_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Buy",
                       marker=dict(symbol='triangle-up', color="green", size=7)),
            go.Scatter(x=aapl_strat.loc[aapl_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Date EMA {ema}"],
                       y=(aapl_strat.loc[
                           aapl_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Sell",
                       marker=dict(symbol='triangle-down', color="red", size=7))
        ]
        fig = go.Figure(data=data)
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(
            title=f"<b>Long/ Short Signal ($AAPL) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig


# Function to render stats
@app.callback(
    Output("output-panel", "children"),
    Input("slider", "value"),  # ema value
    Input('yaxis-column', 'active_tab'))  # tsla aapl
def render_output_panel(ema, ticker):
    panel = html.Div([
        html.H4(f'EMA {ema} Strategy'),
        dbc.Card(body=True, className="text-white bg-primary", children=[
            #             html.H3(f'EMA {ema} Strategy', className='card-title'),

            html.H6("Total Returns:", style={"color": "white"}),
            html.H3(total_returns(ema, ticker), style={"color": "white"}),

            html.H6("Sharpe Ratio:", className="text-success"),
            html.H3(sharpe(ema, ticker), className="text-success"),

            html.H6("Volatility:", style={"color": "white"}),
            html.H3(volatility(ema, ticker), style={"color": "white"}),

            html.H6("Max Drawdown:", className="text-danger"),
            html.H3(max_drawdown(ema, ticker), className="text-danger"),

            html.H6("Number of Long / Short Trades:", style={"color": "white"}),
            html.H3(f"Long: {num_buy_trades(ema, ticker)}", style={"color": "white"}),
            html.H3(f"Short: {num_sell_trades(ema, ticker)}", style={"color": "white"}),

        ])
    ])
    return panel


# Function for bull/bear graph
@app.callback(
    Output('bull-bear-chart', 'figure'),
    Input('yaxis-column', 'active_tab'))
def update_graph_2(yaxis_column_name):
    # TSLA / Signal
    if yaxis_column_name == "$TSLA":

        fig = make_subplots(specs=[[{'secondary_y': True}]])

        # TSLA Bullish Area (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df_tsla["Date"], y=merge_df_tsla["% of Bullish"],
            mode='lines',
            name="Bullish",
            line=dict(width=1, color='rgba(0,102,0,0.3)'),
            stackgroup='one',
            groupnorm='percent'), secondary_y=False)

        # TSLA Bearish Area (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df_tsla["Date"], y=merge_df_tsla["% of Bearish"],
            mode='lines',
            name="Bearish",
            line=dict(width=1, color='rgba(190,23,23,0.3)'),
            stackgroup='one', fill='tonexty'), secondary_y=False)

        # TSLA Chart (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df_tsla["Date"], y=merge_df_tsla["Adjusted Close"],
            mode='lines',
            name='TSLA Closing Price',
            line=dict(width=2, color='black', dash='solid')),
            secondary_y=True)

        # 50% median line (Always visible)
        fig.add_trace(go.Scatter(
            x=merge_df["Date"], y=merge_df["Middle line"],
            mode='lines',
            name="50%",
            line=dict(width=0.5, color='black', dash='dot'),
            showlegend=False,
            hoverinfo='x'), secondary_y=False)

        # Add the top few buttons
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # Configure x & y axis & hovermode
        fig.update_layout(
            title=dict(
                text="<b>Stocktwits Pre-Market Sentiments vs $TSLA Performance</b>",
            ),
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
            xaxis=dict(
                showgrid=False,
                showline=True,
                tickmode='auto',
                nticks=7,
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                range=[0, 100],
                showgrid=False,
                ticksuffix='%',
                fixedrange=True,
                tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            # Converted into the dropdown menu option
            yaxis2=dict(
                type='linear',
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD')
            #         range=[0, 200])
        )

        # Configure spikes
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikedash='dot', spikemode="across",
                         tickformat='%d %b %y', spikethickness=1)
        fig.update_yaxes(title_text="<b>% of Bulls vs Bears</b>", showspikes=False, spikecolor="grey",
                         spikethickness=0.25)
        fig.update_yaxes(title_text="<b>Closing Price</b>", secondary_y=True)

        return fig

        # AAPL / Signal
    elif yaxis_column_name == "$AAPL":
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        # AAPL
        # AAPL Bullish Area (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df["Date"], y=merge_df["% of Bullish"],
            mode='lines',
            name="Bullish",
            line=dict(width=1, color='rgba(0,102,0,0.3)'),
            stackgroup='one',
            groupnorm='percent'), secondary_y=False)

        # AAPL Bearish Area (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df["Date"], y=merge_df["% of Bearish"],
            mode='lines',
            name="Bearish",
            line=dict(width=1, color='rgba(190,23,23,0.3)'),
            stackgroup='one', fill='tonexty'), secondary_y=False)

        # AAPL Chart (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=merge_df["Date"], y=merge_df["Adjusted Close"],
            mode='lines',
            name='AAPL Closing Price',
            line=dict(width=2, color='black', dash='solid')),
            secondary_y=True)

        # 50% median line (Always visible)
        fig.add_trace(go.Scatter(
            x=merge_df["Date"], y=merge_df["Middle line"],
            mode='lines',
            name="50%",
            line=dict(width=0.5, color='black', dash='dot'),
            showlegend=False,
            hoverinfo='x'), secondary_y=False)

        # Add the top few buttons
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # Configure x & y axis & hovermode
        fig.update_layout(
            title="<b>Stocktwits Pre-Market Sentiments vs $AAPL Performance </b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
            xaxis=dict(
                showgrid=False,
                showline=True,
                tickmode='auto',
                nticks=7,
                fixedrange=True,
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                range=[0, 100],
                showgrid=False,
                ticksuffix='%',
                fixedrange=True,
                tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            # Converted into the dropdown menu option
            yaxis2=dict(
                type='linear',
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD')
            #         range=[0, 200])
        )

        # Configure spikes
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikedash='dot', spikemode="across",
                         tickformat='%d %b %y', spikethickness=1)
        fig.update_yaxes(title_text="<b>% of Bulls vs Bears</b>", showspikes=False, spikecolor="grey",
                         spikethickness=0.25)
        fig.update_yaxes(title_text="<b>Closing Price</b>", secondary_y=True)

        return fig


@app.callback(
    Output('data-table', 'children'),
    Input('yaxis-column', 'active_tab'),
    Input("slider", "value"))
def update_graph(yaxis_column_name, ema):
    # TSLA / Signal
    if (yaxis_column_name == "$TSLA") & (ema == 5):
        df = tsla_strat.iloc[:, 1:10]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 6):
        df = tsla_strat.iloc[:, 11:20]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 7):
        df = tsla_strat.iloc[:, 21:30]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 8):
        df = tsla_strat.iloc[:, 31:40]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 9):
        df = tsla_strat.iloc[:, 41:50]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 10):
        df = tsla_strat.iloc[:, 51:60]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 15):
        df = tsla_strat.iloc[:, 61:70]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 20):
        df = tsla_strat.iloc[:, 71:80]
        return get_table(df, ema, ticker=yaxis_column_name)

    # AAPL / Signal
    elif (yaxis_column_name == "$AAPL") & (ema == 5):
        df = aapl_strat.iloc[:, 1:10]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 6):
        df = aapl_strat.iloc[:, 11:20]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 7):
        df = aapl_strat.iloc[:, 21:30]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 8):
        df = aapl_strat.iloc[:, 31:40]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 9):
        df = aapl_strat.iloc[:, 41:50]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 10):
        df = aapl_strat.iloc[:, 51:60]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 15):
        df = aapl_strat.iloc[:, 61:70]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 20):
        df = aapl_strat.iloc[:, 71:80]
        return get_table(df, ema, ticker=yaxis_column_name)


# this is needed for the procfile to deploy to heroku
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=3334)
