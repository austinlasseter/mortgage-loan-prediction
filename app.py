import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pickle
import json
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State


########### Define your variables ######
myheading1='Predicting Mortgage Loan Approval'
image1='ames_welcome.jpeg'
tabtitle = 'Mortgage Loans'
sourceurl = 'https://www.kaggle.com/burak3ergun/loan-data-set'
githublink = 'https://github.com/austinlasseter/mortgage-loan-prediction'

########### open the pickle files ######
# dataframes for visualization
approved=pd.read_csv('model_components/approved_loans.csv')
denied=pd.read_csv('model_components/denied_loans.csv')
# random forest model
filename = open('model_components/loan_approval_rf_model.pkl', 'rb')
rf = pickle.load(filename)
filename.close()
# encoder1
filename = open('model_components/loan_approval_onehot_encoder.pkl', 'rb')
encoder1 = pickle.load(filename)
filename.close()
# ss_scaler1: monthly_return
filename = open('model_components/loan_approval_ss_scaler1.pkl', 'rb')
ss_scaler1 = pickle.load(filename)
filename.close()
# ss_scaler2: ln_total_income
filename = open('model_components/loan_approval_total_income.pkl', 'rb')
ss_scaler2 = pickle.load(filename)
filename.close()
# ss_scaler3: loan_amount
filename = open('model_components/loan_approval_loan_amount.pkl', 'rb')
ss_scaler3 = pickle.load(filename)
filename.close()

########### Write a function to preprocess & predict
features = ['Credit_History',
'LoanAmount',
'Loan_Amount_Term',
'ApplicantIncome',
'CoapplicantIncome',
 'Property_Area',
 'Gender',
 'Education',
  'Self_Employed'
 ]

# Create a function that can take any 8 valid inputs & make a prediction
def make_predictions(listofargs, Threshold):
    try:
        df='hello'
        # the order of the arguments must match the order of the features
        df = pd.DataFrame(columns=features)
        df.loc[0] = listofargs

        # convert arguments from integers to floats:
        for var in ['Credit_History', 'LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']:
            df[var]=int(df[var])

        # recode a few columns using the same steps we employed on the training data
        df['Gender'].replace({'Male': 1, 'Female': 0}, inplace = True)
        df['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace = True)
        df['Self_Employed'].replace({'Yes': 1, 'No': 0}, inplace = True)
        df['LoanAmount'] = df['LoanAmount']*1000

        # transform the categorical variable using the same encoder we trained previously
        ohe=pd.DataFrame(encoder1.transform(df[['Property_Area']]).toarray())
        col_list = ['Property_Area_{}'.format(item) for item in ['Semiurban', 'Urban', 'Rural']]
        ohe.columns=col_list
        df = pd.concat([df, ohe],axis=1)

        # create new features using the scalers we trained earlier
        ln_monthly_return_raw  = np.log(df['LoanAmount']/df['Loan_Amount_Term']).values
        ln_total_income_raw = np.log(int(df['ApplicantIncome']) + int(df['CoapplicantIncome']))
        ln_LoanAmount_raw = np.log(1000*df['LoanAmount'])
        df['ln_monthly_return'] = ss_scaler1.transform(np.array(ln_monthly_return_raw).reshape(-1, 1))
        df['ln_total_income'] = ss_scaler2.transform(np.array(ln_total_income_raw).reshape(-1, 1))
        df['ln_LoanAmount'] = ss_scaler3.transform(np.array(ln_LoanAmount_raw).reshape(-1, 1))

        # drop & rearrange the columns in the order expected by your trained model!
        df=df[['Gender', 'Education', 'Self_Employed', 'Credit_History',
           'Property_Area_Semiurban', 'Property_Area_Urban', 'ln_monthly_return',
           'ln_total_income', 'ln_LoanAmount']]

        prob = rf.predict_proba(df)
        raw_approval_prob=prob[0][1]
        Threshold=Threshold*.01
        approval_func = lambda y: 'Approved' if raw_approval_prob>Threshold else 'Denied'
        formatted_denial_prob = "{:,.1f}%".format(100*prob[0][0])
        formatted_approval_prob = "{:,.1f}%".format(100*prob[0][1])
        return approval_func(raw_approval_prob), formatted_approval_prob, formatted_denial_prob        # return list(df.columns), list(df.columns), str(df.head().values)
    except:
        return 'Invalid inputs','Invalid inputs','Invalid inputs'


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle

########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),

    html.Div([
        html.Div(
            [dcc.Graph(id='fig1',style={'width': '90vh', 'height': '90vh'}),
            ], className='eight columns'),
        html.Div([
                html.H3("Features"),
                html.Div('Credit History'),
                dcc.Input(id='Credit_History', value=1, type='number', min=0, max=1, step=1),
                html.Div('Loan Amount'),
                dcc.Input(id='LoanAmount', value=150, type='number', min=10, max=800, step=10),
                html.Div('Term (in months)'),
                dcc.Input(id='Loan_Amount_Term', value=360, type='number', min=120, max=480, step=10),
                html.Div('Applicant Income'),
                dcc.Input(id='ApplicantIncome', value=5500, type='number', min=0, max=80000, step=100),
                html.Div('Co-Applicant Income'),
                dcc.Input(id='CoapplicantIncome', value=2500, type='number', min=0, max=80000, step=100),
                html.Div('Property Area'),
                dcc.Dropdown(id='Property_Area',
                    options=[{'label': i, 'value': i} for i in ['Semiurban','Urban','Rural']],
                    value='Urban'),
                html.Div('Gender'),
                dcc.Dropdown(id='Gender',
                    options=[{'label': i, 'value': i} for i in ['Male', 'Female']],
                    value='Female'),
                html.Div('Education'),
                dcc.Dropdown(id='Education',
                    options=[{'label': i, 'value': i} for i in ['Graduate', 'Not Graduate']],
                    value='Graduate'),
                html.Div('Self Employed'),
                dcc.Dropdown(id='Self_Employed',
                    options=[{'label': i, 'value': i} for i in ['No','Yes']],
                    value='No'),
                html.Div('Approval Threshold'),
                dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1),

            ], className='two columns'),
            html.Div([
                html.H3('Predictions'),
                html.Div('Predicted Status:'),
                html.Div(id='PredResults'),
                html.Br(),
                html.Div('Probability of Approval:'),
                html.Div(id='ApprovalProb'),
                html.Br(),
                html.Div('Probability of Denial:'),
                html.Div(id='DenialProb')
            ], className='two columns')
        ], className='twelve columns',
    ),

    html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
    ]
)


######### Define Callback: Predictions
@app.callback(
    [Output(component_id='PredResults', component_property='children'),
     Output(component_id='ApprovalProb', component_property='children'),
     Output(component_id='DenialProb', component_property='children'),
    ],
    [Input(component_id='Credit_History', component_property='value'),
     Input(component_id='LoanAmount', component_property='value'),
     Input(component_id='Loan_Amount_Term', component_property='value'),
     Input(component_id='ApplicantIncome', component_property='value'),
     Input(component_id='CoapplicantIncome', component_property='value'),
     Input(component_id='Property_Area', component_property='value'),
     Input(component_id='Gender', component_property='value'),
     Input(component_id='Education', component_property='value'),
     Input(component_id='Self_Employed', component_property='value'),
     Input(component_id='Threshold', component_property='value')
    ])
def func(*args):
    listofargs=[arg for arg in args[:9]]
    return make_predictions(listofargs, args[9])


######### Define Callback: Visualization

@app.callback(Output(component_id='fig1', component_property='figure'),
    [Input(component_id='Credit_History', component_property='value'),
     Input(component_id='LoanAmount', component_property='value'),
     Input(component_id='Loan_Amount_Term', component_property='value'),
     Input(component_id='ApplicantIncome', component_property='value'),
     Input(component_id='CoapplicantIncome', component_property='value'),
     Input(component_id='Property_Area', component_property='value'),
     Input(component_id='Gender', component_property='value'),
     Input(component_id='Education', component_property='value'),
     Input(component_id='Self_Employed', component_property='value')
    ])
def make_loans_cube(*args):
    newdata=pd.DataFrame([args], columns=features)
    newdata['Combined_Income']=newdata['ApplicantIncome'] + newdata['CoapplicantIncome']

    trace0=go.Scatter3d(
        x=approved['LoanAmount'],
        y=approved['Combined_Income'],
        z=approved['Loan_Amount_Term'],
        name='Approved',
        mode='markers',
        text = list(zip(
            ["Credit: {}".format(x) for x in approved['Credit_History']],
            ["<br>Education: {}".format(x) for x in approved['Education']],
            ["<br>Property Area: {}".format(x) for x in approved['Property_Area']],
            ["<br>Gender: {}".format(x) for x in approved['Gender']],
            ["<br>Education: {}".format(x) for x in approved['Education']],
            ["<br>Self-Employed: {}".format(x) for x in approved['Self_Employed']]
                )) ,
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>'+
            '<br><b>Income: $%{y:.0f}</b>'+
            '<br><b>Term: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=6, color='blue', opacity=0.4))

    trace1=go.Scatter3d(
        x=denied['LoanAmount'],
        y=denied['Combined_Income'],
        z=denied['Loan_Amount_Term'],
        name='Denied',
        mode='markers',
        text = list(zip(
            ["Credit: {}".format(x) for x in denied['Credit_History']],
            ["<br>Education: {}".format(x) for x in denied['Education']],
            ["<br>Property Area: {}".format(x) for x in denied['Property_Area']],
            ["<br>Gender: {}".format(x) for x in denied['Gender']],
            ["<br>Education: {}".format(x) for x in denied['Education']],
            ["<br>Self-Employed: {}".format(x) for x in denied['Self_Employed']]
                )) ,
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>'+
            '<br><b>Income: $%{y:.0f}</b>'+
            '<br><b>Term: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=6, color='red', opacity=0.4))

    trace2=go.Scatter3d(
        x=newdata['LoanAmount'],
        y=newdata['Combined_Income'],
        z=newdata['Loan_Amount_Term'],
        name='Applicant',
        mode='markers',
        text = list(zip(
            ["Credit: {} ".format(x) for x in newdata['Credit_History']],
            ["<br>Education: {} ".format(x) for x in newdata['Education']],
            ["<br>Property Area: {}".format(x) for x in newdata['Property_Area']],
            ["<br>Gender: {}".format(x) for x in newdata['Gender']],
            ["<br>Education: {}".format(x) for x in newdata['Education']],
            ["<br>Self-Employed: {}".format(x) for x in newdata['Self_Employed']]
                )) ,
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>'+
            '<br><b>Income: $%{y:.0f}</b>'+
            '<br><b>Term: %{z:.0f}</b>'+
            '<br>%{text}',
        hoverinfo='text',
        marker=dict(size=15, color='yellow'))


    layout = go.Layout(title="Loan Status",
                        showlegend=True,
                            scene = dict(
                            xaxis=dict(title='Loan Amount'),
                            yaxis=dict(title='Combined Income'),
                            zaxis=dict(title='Term')
                    ))
    fig=go.Figure([trace0, trace1, trace2], layout)
    return fig

    # trace0=go.Scatter3d(
    #     x=[7,4,7],
    #     y=[6,2,1],
    #     z=[5,9,3],
    #     name='new data',
    #     mode='markers',
    #     hoverinfo='text',
    #     marker=dict(size=15, color='blue'))
    #
    # layout = go.Layout(title=str(denied.shape),
    #                     showlegend=True,
    #                         scene = dict(
    #                         xaxis=dict(title='Loan Amount'),
    #                         yaxis=dict(title='Combined Income'),
    #                         zaxis=dict(title='Term')
    #                 ))
    # fig=go.Figure([trace0], layout)
    # return fig



############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
