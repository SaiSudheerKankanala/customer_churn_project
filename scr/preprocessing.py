from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scr.config import randomstate,unnecessary_columns,logger

class Preprocessing:
    def __init__(self,data):
        self.data = data
    def preprocessing(self):
        logger.info('Preprocessing method started')
        self.data = self.data.drop(columns = unnecessary_columns)
        logger.info('unnecessary columns are removed')
        numericals = self.data.select_dtypes(include = ["number"]).columns.tolist()
        categorical = self.data.select_dtypes(exclude = ["number"]).columns.tolist()
        num_pipeline = Pipeline([
            ('impute',SimpleImputer(strategy = 'mean'))
        ])
        cat_pipeline = Pipeline([
            ('impute',SimpleImputer(strategy = 'most_frequent')),
            ('encoder',OneHotEncoder(drop = 'first',handle_unknown = 'ignore'))
        ])
        preprocessed = ColumnTransformer([
            ('num',num_pipeline,numericals),
            ('cat',cat_pipeline,categorical)
        ])
        logger.info('Preprocessing had been sucessfully performed')
        return preprocessed