import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class PreProcess(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self
    
    def transform(self, df, *_):
        
        _df = df.copy()
        
        #Drop features
        _df = _df.rename(columns={"Department Name": "DepartmentName"})
        _df = _df[['DepartmentName','InterventionReasonCode', 'ResidentIndicator', 'InterventionDateTime',
                   'SearchAuthorizationCode', 'StatuteReason', 'TownResidentIndicator']]

        #InterventionReasonCode
        _df.InterventionReasonCode = _df.InterventionReasonCode.str.lower().str.strip()    
           
        #SearchAuthorizationCode
        _df.SearchAuthorizationCode = _df.SearchAuthorizationCode.str.lower().str.strip()
        
        #DepartmentName  
        _df = _df.rename(columns={"Department Name": "DepartmentName"})
        _df.DepartmentName = _df.DepartmentName.str.lower().str.strip()
        
        #StatuteReason
        _df.StatuteReason = _df.StatuteReason.str.lower().str.strip()
        
        #Conversion to DateTime type
        _df.InterventionDateTime = pd.to_datetime(_df.InterventionDateTime, format='%m/%d/%Y %I:%M:%S %p')
    
        #Creating time features
        _df = _df.assign(HourDay = _df.InterventionDateTime.dt.hour
                         + _df.InterventionDateTime.dt.minute / 60,
                         DayWeek = _df.InterventionDateTime.dt.dayofweek,
                         Month = _df.InterventionDateTime.dt.month
                        )
        #Drop inital DateTime column
        _df = _df.drop(columns='InterventionDateTime')
        
        
        return _df
