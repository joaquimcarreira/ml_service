#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
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


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
