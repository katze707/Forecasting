import numpy as np
import pip
#pip.main(["install","matplotlib.pyplot"])
import matplotlib.pyplot as plt
#import pip
#pip.main(["install","scikit-learn"])
import sklearn
import scipy
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import timedelta
#import seaborn as sns
import os
#email automation
from typing import List
#SQL
import sqlalchemy as sq
#pip.main(["install","pyodbc"])
import pyodbc
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split



engine = sq.create_engine("mssql+pyodbc://@opsetl/OpsAnalytics?driver=SQL+Server+Native+Client+11.0")
cnx = engine.connect()
dq = """
SELECT * FROM OpsAnalytics.Verifications.wrk_verification_dialer_train
"""
print(datetime.today())
print('Training data fetched')
data = pd.read_sql_query(dq, cnx)
df = pd.get_dummies(data, columns=['Abbreviation'], drop_first=False)

wf= df.groupby(by=['ListingID']).agg({'Amount':'mean', 'BorrowerAPR':'mean',
        'FinanceCharge':'mean', 'MonthlyPayment':'mean', 'EffectiveYield':'mean', 'EstimatedLoss':'mean',
        'ProsperRating':'mean', 'TotalWorkflows':'mean', 'StatedIncome':'mean', 'VerifiedIncome':'mean',
        'IsFraud':'mean', 'IsPendingEngagement':'mean', 'IsPendingEngagementEver':'mean',
        'IsPriorBorrower':'mean', 'IsFirstTouchSLAMet':'mean',
        'IsLegalNameOnFile':'mean', 'IsOutboundSale':'mean', 'IsOriginated':'mean',
        
        'Abbreviation_ADDRESS':'sum', 'Abbreviation_APR':'sum',
        'Abbreviation_BOV':'sum', 'Abbreviation_BOV-PV':'sum', 'Abbreviation_BOV-TR':'sum',
        'Abbreviation_FS':'sum', 'Abbreviation_FS-CALL':'sum', 'Abbreviation_HVR-BS':'sum',
        'Abbreviation_HVR-TR':'sum', 'Abbreviation_IDV':'sum', 'Abbreviation_LDR':'sum',
        'Abbreviation_MERCHANT':'sum', 'Abbreviation_MIL':'sum', 'Abbreviation_OFAC':'sum',
        'Abbreviation_POE':'sum', 'Abbreviation_POE-CALL':'sum', 'Abbreviation_POE-SE':'sum',
        'Abbreviation_POE-TR':'sum', 'Abbreviation_POI':'sum', 'Abbreviation_POI-SE-BBS':'sum',
        'Abbreviation_POI-SE-Y1':'sum', 'Abbreviation_POI-SE-Y2':'sum', 'Abbreviation_POS':'sum',
        'Abbreviation_PV':'sum', 'Abbreviation_SA':'sum', 'Abbreviation_SSC':'sum',
        'Abbreviation_UB':'sum'})

wf=wf.fillna(0)

X = wf.drop(['IsOriginated'],axis=1)
Index = X.columns
ListingID = X.index
y = wf['IsOriginated']

X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.33, random_state=69)

XGB = XGBClassifier(random_state = 0)
XGB.fit(X_train,y_train)
XGB.score(X_test,y_test)
print(datetime.today())
print('Model trained')
#predict active none doc listings
cnx.execute("SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;")
dq3 = """
SELECT distinct a.ListingID,
       a.Amount,a.BorrowerAPR,a.FinanceCharge, 
       a.MonthlyPayment, a.EffectiveYield,a.EstimatedLoss, 
       a.ProsperRating, a.TotalWorkflows, a.StatedIncome,
       a.VerifiedIncome,a.IsFraud,a.IsPendingEngagement,
       a.IsPendingEngagementEver, 
       a.IsPriorBorrower, a.IsFirstTouchSLAMet,
       a.IsLegalNameOnFile, a.IsOutboundSale,
       b.Abbreviation
FROM OpsAnalytics.Verifications.vListing AS a
LEFT JOIN OpsAnalytics.Verifications.vWorkflow AS b
ON a.ListingID = b.ListingID
LEFT JOIN
(SELECT ListingID,	
		SUM(TotalAttempts) as total
 From OpsAnalytics.Verifications.CallListHistory
group by ListingID) c
ON a.ListingID = c.ListingID
WHERE a.IsActive = 1 AND
a.IsGDSComplete = 1 AND
DateDiff(hour, a.GDSRunDateTime, GETDATE()) > 1 AND
a.IsGDS = 0 AND
a.CurrentDocStatus = 'None' AND
(a.TermsApprovalDate >= DATEADD(d,-7,GETDATE()) or 
	(a.TermsApprovalDate >= DATEADD(d,-29,GETDATE()) and		
	(c.total IS NULL OR c.total <2) ))
"""
data_pred = pd.read_sql_query(dq3, cnx)


inputs = pd.get_dummies(data_pred, columns=['Abbreviation'], drop_first=False).fillna(0)

#get missing columns in new test
missing_cols = set(X.columns)-set(inputs.columns)
#Add missing columns in new set with default value equal to 0
for c in missing_cols:
    inputs[c] = 0
    
#Ensure the order of columns in new set are the same with the train set -  
#can skip, 'Cause have to groupby ListingID to reorder anyway
inputs = inputs[X.columns]
inputs['ListingID'] = data_pred['ListingID']
X_pred= inputs.groupby(by=['ListingID']).agg({'Amount':'mean', 'BorrowerAPR':'mean',
       'FinanceCharge':'mean', 'MonthlyPayment':'mean', 'EffectiveYield':'mean', 'EstimatedLoss':'mean',
       'ProsperRating':'mean', 'TotalWorkflows':'mean', 'StatedIncome':'mean', 'VerifiedIncome':'mean',
       'IsFraud':'mean', 'IsPendingEngagement':'mean', 'IsPendingEngagementEver':'mean',
        'IsPriorBorrower':'mean', 'IsFirstTouchSLAMet':'mean',
       'IsLegalNameOnFile':'mean', 'IsOutboundSale':'mean',
       
       'Abbreviation_ADDRESS':'sum', 'Abbreviation_APR':'sum',
       'Abbreviation_BOV':'sum', 'Abbreviation_BOV-PV':'sum', 'Abbreviation_BOV-TR':'sum',
       'Abbreviation_FS':'sum', 'Abbreviation_FS-CALL':'sum', 'Abbreviation_HVR-BS':'sum',
       'Abbreviation_HVR-TR':'sum', 'Abbreviation_IDV':'sum', 'Abbreviation_LDR':'sum',
       'Abbreviation_MERCHANT':'sum', 'Abbreviation_MIL':'sum', 'Abbreviation_OFAC':'sum',
       'Abbreviation_POE':'sum', 'Abbreviation_POE-CALL':'sum', 'Abbreviation_POE-SE':'sum',
       'Abbreviation_POE-TR':'sum', 'Abbreviation_POI':'sum', 'Abbreviation_POI-SE-BBS':'sum',
       'Abbreviation_POI-SE-Y1':'sum', 'Abbreviation_POI-SE-Y2':'sum', 'Abbreviation_POS':'sum',
       'Abbreviation_PV':'sum', 'Abbreviation_SA':'sum', 'Abbreviation_SSC':'sum',
       'Abbreviation_UB':'sum'})

fo = X_pred.copy()
fo['prob_P'] =[ x[1] for x in XGB.predict_proba(X_pred)]
fo['PotentialGain'] = fo['Amount']*fo['prob_P']*0.05
print(datetime.today())
print('Predictions generated')

#Store model metadata and generate pkModelID
cols = ['fkForecastID', 'ModelType', 'ModelName', 'ModelDescription', 'CreatedDate', 'TrainingStartDate', 'TrainingEndDate', 'ObservationsUsed', 'TrainingDataSQL']
data = [[20, 'XGBoost', 'XBG Listing Score ' + str(datetime.today()), 'XGBoost model to predict likelihood of success when attempting to engage a listing.', str(date.today()), str(date.today()-timedelta(days=29)), str(date.today()), str(len(data_pred.index)), str(dq3).replace("'", "''")]]
model_description = pd.DataFrame(data, columns=cols)
model_description.to_sql('Model', cnx, index=False, if_exists='append', schema='Forecast')

#fetch pkModelID
model_output = pd.read_sql_query('SELECT pkModelID = MAX(pkModelID) FROM Forecast.Model WHERE fkForecastID = 20', cnx)

of = fo.filter(items=['PotentialGain', 'prob_P', 'Amount'])
of['ModelID'] = model_output.iloc[0]['pkModelID']
of['RowLoadDateTime'] = datetime.today()
lists = pd.DataFrame(data=fo.sort_values(by=['PotentialGain'], ascending=False).index,columns=['ListingID'])
lists = lists.merge(of, on = 'ListingID')
print(datetime.today())
for i in range(0, len(lists)):
    print(lists.iloc[[i]])

#insert into SQL table
lists.to_sql('wrk_verification_dialer',engine,index=True,if_exists = 'replace',schema='Verifications')
cnx.close()
print(datetime.today())
print('SQL insert complete')