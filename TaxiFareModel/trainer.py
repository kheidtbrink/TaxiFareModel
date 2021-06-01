# imports

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer:
    def __init__(self, X_train, y_train):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)
        
    
    def holdout(self,X,y,**kwargs):
        """holdout is best on train_test_split of sklearn - use accepted 
        args and kwargs from sklearn return the train test split"""
        return train_test_split(X, y, **kwargs)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    print(df.head())
    # clean data
    df_clean = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    #X,y = get_data()
    # hold out
    # train
    # evaluate
    print('TODO')
