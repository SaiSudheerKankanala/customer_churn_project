from sklearn.model_selection import train_test_split,RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scr.config import randomstate,Target_column,test_size,rf_param,xgb_param,logger,MLFLOW_TRACKING_URL,MLFLOW_EXPERIMENT_NAME
from scr.preprocessing import Preprocessing
import mlflow


class data_split:
    def __init__(self,data):
        self.data = data
    def split(self):
        x = self.data.drop(columns = Target_column)
        y = self.data[Target_column].map({'Yes':1,
                                          'No':0})
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = test_size, random_state = randomstate)
        logger.info(f"The shape of the x_train is {x_train.shape},The shape of the y_train is {y_train.shape}")
        logger.info(f"The shape of the x_test is {x_test.shape},The shape of the y_test is {y_test.shape}")
        x_train_val,x_val,y_train_val,y_val = train_test_split(x_train,y_train,test_size = test_size,random_state = randomstate)
        logger.info(f"The shape of the x_train_val is {x_train_val.shape},The shape of the y_train_val is {y_train_val.shape}")
        logger.info(f"The shape of the x_val is {x_val.shape},The shape of the y_val is {y_val.shape}")
        return x_train_val,y_train_val,x_val,y_val,x_test,y_test

class model_building:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def training(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("Training the model")
        models = {
            'logistic':LogisticRegression(max_iter=1000),
            'rf':RandomForestClassifier(random_state = randomstate),
            'xgb':XGBClassifier(random_state = randomstate,
                                eval_metric = 'logloss')
        }
        grid_param = {
            'rf':rf_param,
            'xgb':xgb_param
        }
        trained_models = {}
        for name, model in models.items():
            preprocessed = Preprocessing(self.x_train).preprocessing()
            steps = [
                        ('preprocessed',preprocessed),
                    ]
            if name == 'logistic':
                steps.append(('scaler',RobustScaler()))
                steps.append(('feature_selection',RFE(estimator = LogisticRegression(),
                                                     n_features_to_select= 5)))
            steps.append(('smote',SMOTE(random_state=randomstate)))
            steps.append(('model',model))
            pipeline = Pipeline(steps)

            if name in grid_param:
                cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=randomstate)
                search = RandomizedSearchCV(estimator = pipeline,
                                            param_distributions= {
                                                f"model__{key}":value
                                                for key, value in grid_param[name].items()
                                            },
                                            scoring = 'recall',
                                            cv = cv,
                                            refit = True,
                                            n_iter= 20,
                                            n_jobs = -1,
                                            verbose = 2
                )
                search.fit(self.x_train,self.y_train)
                trained_models[name] = search.best_estimator_
                best_params = {key.replace("model__","").value
                               for key,value in search.best_estimator_.items()
                               }
                mlflow.log_params(best_params)
                mlflow.log_metrics('best_cv_recall',float(search.best_score_))
            else:
                pipeline.fit(self.x_train,self.y_train)
                mlflow.log_params(model.get_params(deep = False))

                trained_models[name] = pipeline
        return trained_models
