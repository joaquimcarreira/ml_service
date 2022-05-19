import sklearn 
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
class Dates_enginering(BaseEstimator,TransformerMixin):
    '''
    From date column object, creates day and month features (year is omitted)
    '''
    def __init__(self,date_column,drop_original=True):
        self.date_column = date_column
        self.drop_original = drop_original

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        # From date, create month and day columns
        if self.date_column:
            #assert(self.date_column in X.columns)
            X = pd.DataFrame(X,columns=["date_published"])
            X["day"] = pd.to_datetime(X[self.date_column]).dt.day
            X["month"] = pd.to_datetime(X[self.date_column]).dt.month
            X["week_day"] = pd.to_datetime(X[self.date_column]).dt.dayofweek
            if self.drop_original:
                X.drop(columns=[self.date_column],inplace=True)

        return X

class RandomForestClassifier():
    def __init__(self):
        path_to_artifacts = "../../research/"
        #the random forest is alrady a sklearn pipeline with preprocessors
        self.rf = joblib.load(path_to_artifacts+"rf_v1.joblib")
        self.labels = joblib.load(path_to_artifacts+"label_encoders.joblib")
    
    def predict(self,input_data):
        input_data = pd.DataFrame(input_data,index=[0])
        try:
            results = dict()
            results["probability"] = self.rf.predict_proba(input_data)
            results["prediction"] = self.labels.inverse_transform(
                                    self.rf.predict(input_data)
                                    )
        except Exception as e:
            return {"status":"Error","message":str(e)}
        return results
