# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:56:11 2022

@author: kyan
"""

#Verification Forecast of Daily TILA distribution, outbound Call Volume, weekly TL needs 
import pandas as pd
import sqlalchemy as sq
import pyodbc
cnx = pyodbc.connect('DRIVER={SQL Server Native Client 11.0};Server=OpsETL;Database=OpsAnalytics;Trusted_Connection=yes')
batch_m = """
SELECT TOP 1 BatchID FROM Verifications.DailyForecast ORDER BY PublishedDate DESC
"""
batch_n = """
SELECT TOP 1 ForecastBatchID FROM Staffing.vForecastPublished WHERE IsCurrent = 1
"""
used = pd.read_sql_query(batch_m, cnx)
new = pd.read_sql_query(batch_n, cnx)
if used['BatchID'][0] == new['ForecastBatchID'][0]:
    print ("no new forecast")
    

if used['BatchID'][0] != new['ForecastBatchID'][0]:
    #import sqlalchemy as sq
    #import pyodbc
    from datetime import date
    from datetime import timedelta
    #import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    #cnx = pyodbc.connect('DRIVER={SQL Server Native Client 11.0};Server=OpsETL;Database=OpsAnalytics;Trusted_Connection=yes')    
    #Future Monthly TILA Projections from Naveen
    q1 = """
    SELECT MonthStart, SUM(KPIVALUE_FCST) AS GrossTILA
    FROM Staffing.vForecastPublished
    WHERE KPICategory = 'gross TILA #' 
    AND IsCurrent = 1 
    GROUP BY MonthStart
    ORDER BY MonthStart"""
    
    
    #Past Daily TILA Actuals
    q2 = """
    SELECT TermsApprovalDate,count(ListingID) AS GrossTILA,sum(cast(IsManual AS INT)) AS ManualTILA 
    FROM OpsAnalytics.Verifications.vListing
    WHERE IsActive = 0 
    AND TermsApprovalDate >= '2017-01-01'
    GROUP BY TermsApprovalDate
    ORDER BY TermsApprovalDate
    """
    
    #Past Daily Inbound Call Volume Actuals
    q3 = """
    SELECT SkillCallDate ,count(SkillCallID) AS CallVolume
    FROM OpsAnalytics.Five9.vSkillCall
    WHERE Skill = 'Verification Active Answer'
    GROUP BY SkillCallDate
    ORDER BY SkillCallDate
    """
    
    #Past Daily Decisions Actuals
    q4 = """
    SELECT SubmitDate,count (DISTINCT ListingID) AS Listings, count(DecisionID) AS Decisions
    FROM Verifications.vDecisions
    WHERE SubmissionType IN ('Approval Review','Cancellation Review','Escalation Review')
    GROUP BY SubmitDate
    ORDER BY SubmitDate"""
    
    
    #add TL prediction
    q5 = """ 
    DECLARE @startDate DATE = DATEADD(DAY, -1 * 182, GETDATE())
    DECLARE @endDate DATE = DATEADD(DAY, -1, GETDATE())
    
    SELECT SecondReviewDate, ListingID, ListingReviewer
    FROM OpsAnalytics.Verifications.vSecondReview
    WHERE ListingReviewer IS NOT NULL
    AND CAST(SecondReviewDate AS DATE)  BETWEEN @startDate AND @endDate
    """
    MonthlyTILA = pd.read_sql_query(q1,cnx)
    DailyTILA = pd.read_sql_query(q2,cnx)
    DailyCall = pd.read_sql_query(q3,cnx)
    decisions = pd.read_sql_query(q4,cnx)
    reviews = pd.read_sql_query(q5, cnx)
    
    cnx.close()
    
    
    DailyTILA['Date']=pd.to_datetime(DailyTILA['TermsApprovalDate'])
    DailyTILA['Year']=DailyTILA['Date'].dt.year
    DailyTILA['Month'] = DailyTILA['Date'].dt.month
    DailyTILA['YearMonth'] = DailyTILA['Date'].dt.strftime('%Y-%m')
    weekday_char = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    DailyTILA['Weekday'] = DailyTILA['Date'].dt.dayofweek.apply(lambda x: weekday_char[x])
    
    #past 28-day avg. manual% --for final daily manual TILA calculation
    MAX = DailyTILA['Date'].max()
    MIN = DailyTILA['Date'].max()-timedelta(days=28)
    df_manual=DailyTILA.loc[(DailyTILA['Date'] >= MIN) & (DailyTILA['Date']<MAX)]
    df_manual['Manual%']=df_manual['ManualTILA']/df_manual['GrossTILA']
    Manual=df_manual['Manual%'].mean()
    
    
    MonthlyTotal = DailyTILA.groupby(by=['YearMonth']).agg({'GrossTILA':'sum','Date':'count'})
    MonthlyTotal.columns = ['TotalOfMonth','DaysOfMonth']
    
    
    df = DailyTILA.merge(MonthlyTotal, on = 'YearMonth', how='left')
    df['Daily%'] = df['GrossTILA']/df['TotalOfMonth']
    
    wf = df.filter(items=['Date','Weekday','DaysOfMonth','Daily%']).set_index('Date')
    
    threshold = pd.to_datetime(date(date.today().year,date.today().month,1))
    mf = wf.iloc[wf.index < threshold]
    
    X = pd.get_dummies(mf, columns=['Weekday'], drop_first=True).drop(['Daily%'], axis = 1)
    y = mf['Daily%']
    
    #use previous month's data to test
    #splitdate = pd.to_datetime(date(date.today().year,date.today().month-1,1))
    #X_train = X.iloc[X.index < splitdate]  
    #X_test = X.iloc[X.index >= splitdate] 
    #y_train = y.iloc[y.index < splitdate] 
    #y_test = y.iloc[y.index >= splitdate] 
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    regressor = RandomForestRegressor(n_estimators = 6, random_state = 0)
    regressor.fit(X_train, y_train)
    regressor.score(X_test,y_test)
    
    
    #create forecast input
    startdate = pd.to_datetime(date(date.today().year,date.today().month+2,1))
    enddate = pd.to_datetime(date(date.today().year,date.today().month+5,1)-timedelta(days=1))
    inputs = pd.DataFrame(data=pd.date_range(start=startdate,end=enddate), columns=['Date'])
    daysofmonth = [31,28,31,30,31,30,31,31,30,31,30,31]
    inputs['DaysOfMonth'] = inputs['Date'].dt.month.apply(lambda x: daysofmonth[x-1])
    inputs['Weekday'] = inputs['Date'].dt.dayofweek.apply(lambda x: weekday_char[x])
    
    
    
    X_fcst = pd.get_dummies(inputs,columns=['Weekday'], drop_first=True).set_index('Date')
    
    
    
    
    inputs['MonthStart'] = 0
    for i in range(len(inputs)):
        inputs['MonthStart'][i] = date(inputs['Date'][i].year,inputs['Date'][i].month,1)
    
    
    
    
    outputs = inputs.merge(MonthlyTILA, how = 'left', on = 'MonthStart')
    outputs['Daily%']=regressor.predict(X_fcst)
    #Scale the daily%
    op_month = outputs.groupby(by=['MonthStart']).agg({'Daily%':'sum'})
    outputs = outputs.merge(op_month, how = 'left', on='MonthStart')
    outputs['Daily%'] = outputs['Daily%_x']*(1+(1-outputs['Daily%_y']))
    outputs = outputs.drop(columns=['Daily%_x','Daily%_y'])
    outputs['DailyGrossTILA'] = outputs['GrossTILA']*outputs['Daily%']
    outputs['Past28dayAvgManual%'] = Manual
    outputs['DailyManualTILA'] = outputs['DailyGrossTILA']*outputs['Past28dayAvgManual%']
    
    #add daily call volume prediction
    df2 = df.merge(DailyCall, left_on = 'TermsApprovalDate', right_on = 'SkillCallDate')
    wf2 = df2.filter(items=['Date','GrossTILA','DaysOfMonth','Weekday','CallVolume']).set_index('Date')
    X2 = pd.get_dummies(wf2, columns=['Weekday'], drop_first=True).drop(['CallVolume'], axis = 1)
    y2 = wf2['CallVolume']
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.2,random_state=2)
    print(X2_train.shape, y2_train.shape, X2_test.shape, y2_test.shape)
    

    regressor2 = RandomForestRegressor(n_estimators = 35, random_state = 2)
    regressor2.fit(X2_train, y2_train)
    regressor2.score(X2_test,y2_test)
    
    
    X2_fcst = outputs.filter(items=['DailyGrossTILA']).merge(X_fcst, 
                                                             left_on = outputs['Date'], right_on = X_fcst.index).set_index('key_0')
    
    
    outputs['DailyCallVolume'] = regressor2.predict(X2_fcst)
    
    std_weekday = outputs[~outputs['Weekday'].isin(['Sat', 'Sun'])]['DailyCallVolume'].std()
    std_weekend = outputs[outputs['Weekday'].isin(['Sat', 'Sun'])]['DailyCallVolume'].std()
    
    
    
    #1.64 is the t-value for alpha=0.05
    def get_vol_lb(x, y, std_inputs, flag='lower'):
        if x in ['Sat', 'Sun']:
            if flag == 'lower':
                return y - 1.64*std_inputs[0]
            else:
                return y + 1.64*std_inputs[0]
        else:
            if flag == 'lower':
                return y - 1.64*std_inputs[1]
            else:
                return y + 1.64*std_inputs[1]
    
    
    outputs['CallVolume_LowerBound'] = outputs[['Weekday' ,'DailyCallVolume']].apply(lambda x: get_vol_lb(x.iloc[0], x.iloc[1], 
                                                                                            [std_weekend, std_weekday]), axis=1)
    
    outputs['CallVolume_UpperBound'] = outputs[['Weekday' ,'DailyCallVolume']].apply(lambda x: get_vol_lb(x.iloc[0], x.iloc[1], 
                                                                                            [std_weekend, std_weekday], flag='upper'), axis=1)
    
    
    outputs['CallVolume_LowerBound'] = outputs['CallVolume_LowerBound'].apply(lambda x : x if x > 0 else 0)
    
    
    
    df3 = df.merge(decisions, left_on = 'TermsApprovalDate', right_on = 'SubmitDate')
    
    
    
    
    
    wf3 = df3.filter(items=['Date','ManualTILA','DaysOfMonth','Weekday','Decisions']).set_index('Date')
    X3 = pd.get_dummies(wf3, columns=['Weekday'], drop_first=True).drop(['Decisions'], axis = 1)
    y3 = wf3['Decisions']
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.2,random_state=3)
    print(X3_train.shape, y3_train.shape, X3_test.shape, y3_test.shape)
    
    
    regressor3 = RandomForestRegressor(n_estimators = 35, random_state = 3)
    regressor3.fit(X3_train, y3_train)
    regressor3.score(X3_test,y3_test)
    
    
    X3_fcst = outputs.filter(items=['DailyManualTILA']).merge(X_fcst, 
                                                             left_on = outputs['Date'], right_on = X_fcst.index).set_index('key_0')
    
    outputs['DailyDecision']=regressor3.predict(X3_fcst)
    
    
    reviews['Date']=reviews['SecondReviewDate'].dt.strftime('%Y-%m-%d')
    reviews_dp =  reviews.groupby(by=['Date','ListingReviewer']).agg({'ListingID':pd.Series.nunique}).reset_index()
    reviews_tl = reviews_dp.query('ListingID > 10')#get rid of AMs's performance
    avg_tl =  reviews_tl.groupby(by=['Date']).agg({'ListingReviewer':pd.Series.nunique,'ListingID':'sum'}).reset_index()
    avg_tl['ListingPerTL'] = avg_tl['ListingID']/avg_tl['ListingReviewer']
    LPTL = avg_tl[~pd.to_datetime(avg_tl['Date']).dt.weekday.isin([5,6])]['ListingPerTL'].mean()
    outputs['Past6monthAvgListPerTL']=LPTL
    outputs['DailyTL']=round(outputs['DailyDecision']/outputs['Past6monthAvgListPerTL'],0)
    
    
    #start with Saturday
    outputs['WeekfSat'] = outputs['Date'].apply(lambda x: (x + timedelta(days=2)).week)
    op_wk = outputs.groupby(by=['WeekfSat']).agg({'DailyDecision':'sum','Date':'count'})
    op_wk.columns=['WeeklyDecisions','DaysOfWeek']
    outputs = outputs.merge(op_wk, how='left', left_on='WeekfSat', right_on=op_wk.index)
    outputs['WeeklyTL']=round(outputs['WeeklyDecisions']/outputs['Past6monthAvgListPerTL']/5,0)
    outputs['IsPartialWeek'] = outputs['DaysOfWeek'].apply(lambda x : 1 if x < 7 else 0)
    outputs['BatchID']=new['ForecastBatchID'][0]
    outputs['PublishedDate']=date.today()
    #insert into SQL table
    engine = sq.create_engine("mssql+pyodbc://@OpsETL/OpsAnalytics?driver=SQL+Server+Native+Client+11.0")
    outputs.to_sql('DailyForecast',engine,index=False,if_exists = 'append',schema='Verifications') #replace

