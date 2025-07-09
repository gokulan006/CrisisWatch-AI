import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import datetime
import dash_bootstrap_components as dbc
import folium
from folium.plugins import HeatMap
from io import BytesIO
import base64
from wordcloud import WordCloud
from sqlalchemy import create_engine

import plotly.io as pio
pio.templates.default = "plotly"
 
engine=create_engine('sqlite:///posts.db')
def create_dashboard(flask_app):
    app = dash.Dash(
        server=flask_app,
        name='Dashboard',
        url_base_pathname='/dashboard/',
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    def fetch_posts(start_date=pd.to_datetime('2025-06-29'),end_date=pd.to_datetime(datetime.date.today())):
        df=pd.read_sql_table('reddit_posts',con=engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if start_date:
            df=df[df['timestamp'] >= start_date]
        
        if end_date:
            df=df[df['timestamp'] <= end_date]
        
        return df
        

    def fetch_locations(start_date=pd.to_datetime('2025-06-29'),end_date=pd.to_datetime(datetime.date.today())):
        df=pd.read_sql_table('locations',con=engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if start_date:
            df=df[df['timestamp'] >= start_date]
        
        if end_date:
            df=df[df['timestamp'] <= end_date]
        
        return df

    def fetch_user_behavior(start_date=pd.to_datetime('2025-06-29'),end_date=pd.to_datetime(datetime.date.today())):
        df=pd.read_sql_table('Users',con=engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if start_date:
            df=df[df['timestamp'] >= start_date]
        
        if end_date:
            df=df[df['timestamp'] <= end_date]
        
        return df
    
    posts_df=fetch_posts()

    def loading_wrapper(graph_component, id):
        return dcc.Loading(
            id=f"loading-{id}",
            type="circle",
            children=[graph_component],
            fullscreen=False
        )

    tab1=html.Div([
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Risk Level Distribution",className='card-title'),
                        loading_wrapper(dcc.Graph(id='risk-distribution'), "risk-distribution")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Upvotes Distribution by Risk Level",className='card-title'),
                        loading_wrapper(dcc.Graph(id='risk-upvotes'), "risk-upvotes")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Risk Level Trends Over Time",className='card-title'),
                        dcc.Dropdown(
                            id='risk-filter',
                            options=[{'label':risk_level,'value':risk_level} for risk_level in posts_df['risk_level'].unique()],
                            placeholder='Select a Risk Level',
                            value='High Risk',
                            className='dropdown'
                        ),
                        loading_wrapper(dcc.Graph(id='risk-trend'), "risk-trend")
                         
                    ],className='card-body')
                ],className='card'),
                width=6
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Posts Over Time",className='card-title'),
                        loading_wrapper(dcc.Graph(id='post-trend'), "post-trend")
                    ],className='card-body')
                ],className='card pb-8'),
                width=6,
            )
        ])
    ],className='p-3')

    tab2=html.Div([
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Sentiment Distribution",className='card-title'),
                        loading_wrapper(dcc.Graph(id='sentiment-distribution'), "sentiment-distribution")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Sentiment vs Engagement by risk',className='card-title'),
                        loading_wrapper(dcc.Graph(id='sentiment-risk'), "sentiment-risk")
                    ],className='card-body')
                ],className='card'),
                width=6
            )
            
        ]),
        
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Sentiment Trend Over Time',className='card-title'),
                        dcc.Dropdown(
                            id='sentiment-filter',
                            options=[{'label':sentiment , "value":sentiment} for sentiment in posts_df['sentiment'].unique()],
                            placeholder='Select a Sentiment',
                            value='Negative',
                            className='dropdown'
                        ),
                        loading_wrapper(dcc.Graph(id='sentiment-trend'), "sentiment-trend")
                    ],className='card-body')
                ],className='card pb-8'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Sentiment Word Cloud',className='card-title'),
                        html.Img(src='./assets/wordcloud.png', style={'width': '100%'},className='word-cloud')
                    ],className='card-body')
                ],className='card'),
                width=6,
                
            )
        ])
        
    ],className='p-3')


    tab3=html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Top 10 High-Risk Locations',className='card-title'),
                        loading_wrapper(dcc.Graph(id='locations-10'), "locations-10")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Geographic Risk Over Time',className='card-title'),
                        loading_wrapper(dcc.Graph(id='location-trend'), "location-trend")
                    ],className='card-body')
                ],className='card'),
                width=6
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Geospatial Heatmap of High-Risk Locations',className='card-title'),
                        html.Iframe(
                            srcDoc=open('./assets/geospatial_heatmap.html','r').read(),
                            width='100%',
                            height='500',
                            className='heatmap'
                        )
                    ],className='card-body')
                ])
            )
        ],class_name='px-20')
        

    ],className='p-3')

    tab4=html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Top 10 High-Risk Users',className='card-title'),
                        loading_wrapper(dcc.Graph(id='user-top-10'), "user-top-10")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('User Post Distribution',className='card-title'),
                        loading_wrapper(dcc.Graph(id='user-post-distribution'), "user-post-distribution")  
                    ],className='card-body')
                ],className='card'),
                width=6
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('User Behavior Patterns',className='card-title'),
                        loading_wrapper(dcc.Graph(id='user-pattern'), "user-pattern")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('User Sentiment Trend Distribution',className='card-title'),
                        loading_wrapper(dcc.Graph(id='user-sentiment'), "user-sentiment")
                    ],className='card-body')
                ],className='card'),
                width=6
            )
        ])
    ],className='p-3')

    tab5=html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Average Engagement Over Time',className='card-title'),
                        loading_wrapper(dcc.Graph(id='engagement-time'), "engagement-time")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Engagement Distribution by Risk Level',className='card-title'),
                        loading_wrapper(dcc.Graph(id='engagement-risk'), "engagement-risk")    
                    ],className='card-body')
                ],className='card'),
                width=6
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Engagement Distribution by Sentiment',className='card-title'),
                        loading_wrapper(dcc.Graph(id='engagement-sentiment'), "engagement-sentiment")
                    ],className='card-body')
                ],className='card'),
                width=6
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Top 10 Engaging Users',className='card-title'),
                        loading_wrapper(dcc.Graph(id='engagement-users'), "engagement-users")
                    ],className='card-body')
                ],className='card'),
                width=6
            )
        ])
    ],className='p-3')

    app.layout=html.Div([
        
            html.Header([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-bar text-2xl text-green-600 animate-pulse-slow hover-wiggle"),
                        html.Span('CrisisWatch AI Dashboard',className="text-xl font-bold text-gray-900 hover-text-gradient transition-all duration-300")],
                        className="flex items-center space-x-2 animate-slide-in-left"),
                    
                    html.Div([
                        html.Label('Select Date Range',className='date-label'),
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=datetime.date(2025,1,1),
                            max_date_allowed=datetime.date(2025,7,9),
                            start_date=datetime.date(2025,6,29),
                            end_date=datetime.date(2025,7,9),
                            display_format='YYYY-MM-DD',
                            className='date-picker',
                            style={
                                'fontSize': '18px',
                                'padding-right': '15px',
                                'padding-bottom': '15px'
                            }
                        )
                    ],className='flex-col items-center space-x-4 animate-slide-in-right border bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 transform hover-scale hover-glow button-pulse font-medium mt-2')
                ],className='flex justify-between items-center h-16 px-8')
            ],className='bg-white/95 backdrop-blur-sm border-b border-gray-200 transition-all duration-300 p-3'),

            html.Div([
                dbc.Tabs([
                    dbc.Tab(tab1, label="Risk Analysis",className='tab',label_style={'background-color':"#4aaeff",'color':'white'},active_label_style={'transform': 'scale(1.1)','background-color':'#2563eb'}),
                    dbc.Tab(tab2, label='Sentiment Analysis',className='tab',label_style={'background-color':"#4aaeff",'color':'white'},active_label_style={'transform': 'scale(1.1)','background-color':'#2563eb'}),
                    dbc.Tab(tab3, label='Geographic Analysis',className='tab',label_style={'background-color':"#4aaeff",'color':'white'},active_label_style={'transform': 'scale(1.1)','background-color':'#2563eb'}),
                    dbc.Tab(tab4,label='User Behavior Analysis',className='tab',label_style={'background-color':"#4aaeff",'color':'white'},active_label_style={'transform': 'scale(1.1)','background-color':'#2563eb'}),
                    dbc.Tab(tab5,label='Engagement Analysis',className='tab',label_style={'background-color':"#4aaeff",'color':'white'},active_label_style={'transform': 'scale(1.1)','background-color':'#2563eb'})
                ],className='dashboard-tabs'),
            ])
    ])

    @app.callback(
        Output('risk-distribution','figure'),
        Output('risk-upvotes','figure'),
        Output('risk-trend','figure'),
        Output('post-trend','figure'),
        Input('date-range','start_date'),
        Input('date-range','end_date'),
        Input('risk-filter','value'),
    )
    def update_risk_analysis(start_date,end_date,risk_value):
        
        posts_df=fetch_posts(start_date,end_date)
        risk_pie=px.pie(
            posts_df,
            names='risk_level',
            color_discrete_map={
                'High Risk': '#ef4444',
                'Moderate Risk': '#f59e0b', 
                'Low Risk': '#10b981'
            }
        )
        risk_pie.update_layout(
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        risk_upvotes=px.violin(
            posts_df,
            x='risk_level',
            y='upvotes',
            box=True,
            color='risk_level',
            color_discrete_map={
                'High Risk': '#ef4444',
                'Moderate Risk': '#f59e0b', 
                'Low Risk': '#10b981'
            }
        )
        risk_upvotes.update_layout(
            margin=dict(l=20, r=20, t=40, b=20)
        )

        risk_over_time= posts_df.groupby([posts_df['timestamp'].dt.date,posts_df['risk_level']==risk_value]).size().unstack().fillna(0)
        risk_over_time=risk_over_time.get(True, pd.Series()).rename("Count")

        risk_trend=px.line(
            risk_over_time,
            x=risk_over_time.index,
            y="Count"
        )
        

        post_over_time=posts_df.groupby(posts_df['timestamp'].dt.date).size()
        post_trend=px.area(
            post_over_time
        )
        post_trend.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        return (risk_pie,risk_upvotes,risk_trend,post_trend)
    @app.callback(
        Output('sentiment-distribution','figure'),
        Output('sentiment-risk','figure'),
        Output('sentiment-trend','figure'),
        Input('date-range','start_date'),
        Input('date-range','end_date'),
        Input('sentiment-filter','value'),
    )
    def update_sentiment_analysis(start_date,end_date,sentiment_value):
        
        posts_df=fetch_posts(start_date,end_date)
        sentiment_fig=px.pie(
            posts_df,
            names='sentiment',
            color_discrete_map={
                'Positive': '#10b981',
                'Negative': '#ef4444',
                'Neutral': '#6b7280'
            }
        )
        sentiment_fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20)
        )

        sentiment_risk=px.violin(
            posts_df,
            x='sentiment',
            y='upvotes',
            box=True,
            color='sentiment',
            color_discrete_map={
                'Positive': '#10b981',
                'Negative': '#ef4444',
                'Neutral': '#6b7280'
            }
        )
        sentiment_risk.update_layout(
            margin=dict(l=20, r=20, t=40, b=20)
        )

        sentiment_over_time=posts_df.groupby([posts_df['timestamp'].dt.date, posts_df['sentiment']==sentiment_value]).size().unstack().fillna(0)
        sentiment_over_time=sentiment_over_time.get(True, pd.Series()).rename("Count")
        sentiment_trend=px.line(
            sentiment_over_time
        )
        sentiment_trend.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap='Blues'
        ).generate(' '.join(posts_df['content'].astype(str)))
        wordcloud.to_file("./assets/wordcloud.png")
        return (sentiment_fig,sentiment_risk,sentiment_trend)

    @app.callback(
        Output('locations-10','figure'),
        Output('location-trend','figure'),
        Input('date-range','start_date'),
        Input('date-range','end_date')
    )
    def update_risk_analysis(start_date,end_date):
        locations_df=fetch_locations(start_date,end_date)
        m = folium.Map(
            location=[locations_df['latitude'].mean(), locations_df['longitude'].mean()],
            zoom_start=2,
            tiles='cartodbpositron'
        )
        
        heat_data = [[row['latitude'], row['longitude']] for index, row in locations_df.iterrows()]
        HeatMap(heat_data).add_to(m)
        m.save('./assets/geospatial_heatmap.html')

        location_counts=locations_df['location'].value_counts().head(10)
        locations_10=px.bar(
            location_counts
        )
        locations_10.update_layout(
            xaxis_title="Location",
            yaxis_title="Count",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        locations_df['date']=locations_df['timestamp'].dt.date

        location_trend = px.scatter_geo(
            locations_df,
            lat='latitude',
            lon='longitude',
            color='risk_level',
            animation_frame='date',
            color_discrete_map={
                'High Risk': '#ef4444',
                'Moderate Risk': '#f59e0b', 
                'Low Risk': '#10b981'
            }
        )
        location_trend.update_layout(
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return (locations_10,location_trend)

    @app.callback(
        Output('user-top-10','figure'),
        Output('user-post-distribution','figure'),
        Output('user-sentiment','figure'),
        Output('user-pattern','figure'),
        Input('date-range','start_date'),
        Input('date-range','end_date')
    )
    def update_user_behavior(start_date,end_date):
        user_df=fetch_user_behavior(start_date,end_date)

        user_df=user_df[~user_df['username'].isin(['nan','None','[deleted]'])]
        top_high_risk_user=user_df.groupby('username')['high_risk_count'].sum().nlargest(10)
        user_top_10=px.bar(
            top_high_risk_user,
            labels={'value': 'High-Risk Posts', 'username': 'User'}, 
            template='plotly'
        )
        user_top_10.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )

        user_post_distribution=px.histogram(
            user_df,
            x='post_count'
        )
        user_post_distribution.update_layout(
            xaxis_title="Post Count",
            yaxis_title="Number of Users",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        user_sentiment=px.box(
            user_df,
            y='sentiment_trend'
        )
        user_sentiment.update_layout(
            margin=dict(l=20, r=20, t=40, b=20)
        )

        user_pattern=px.scatter(
            user_df,
            x='post_count',
            y='high_risk_count'
        )
        user_pattern.update_layout(
            xaxis_title="Total Posts",
            yaxis_title="High Risk Posts",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return (user_top_10,user_post_distribution,user_sentiment,user_pattern)

    @app.callback(
        Output('engagement-time','figure'),
        Output('engagement-risk','figure'),
        Output('engagement-sentiment','figure'),
        Output('engagement-users','figure'),
        Input('date-range','start_date'),
        Input('date-range','end_date')
    )
    def update_engagement_analysis(start_date,end_date):
        posts_df=fetch_posts(start_date,end_date)
        posts_df['engagement']=posts_df['upvotes'] + posts_df['comments']
        engagement=posts_df.groupby(posts_df['timestamp'].dt.date)['engagement'].mean()
        engagement_time=px.line(
            engagement,
        )
        engagement_time.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Average Engagement",
            showlegend=False
        )

        engagement_risk=px.violin(
            posts_df,
            x='risk_level',
            y='engagement',
            color='risk_level',
            box=True,
            color_discrete_map={
                'High Risk': '#ef4444',
                'Moderate Risk': '#f59e0b', 
                'Low Risk': '#10b981'
            }
        )
        engagement_risk.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Risk Level",
            yaxis_title="Engagement Score"
        )

        engagement_sentiment=px.violin(
            posts_df,
            x='sentiment',
            y='engagement',
            box=True,
            color='sentiment',
            color_discrete_map={
                'Positive': '#10b981',
                'Negative': '#ef4444',
                'Neutral': '#6b7280'
            }
        )
        engagement_sentiment.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Sentiment",
            yaxis_title="Engagement Score"
        )
        
        posts_df=posts_df[posts_df['username'] != 'None']
        top_users=posts_df.groupby('username')['engagement'].sum().nlargest(10)
        engagement_user= px.bar(
            top_users
        )
        engagement_user.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Username",
            yaxis_title="Total Engagement",
            showlegend=False
        )

        return (engagement_time,engagement_risk,engagement_sentiment,engagement_user)
    
    return app
 
