from scr.config import randomstate
from scr.data_injection import data_injection
from scr.train import data_split, model_building
from scr.evaluation import Evaluation
import joblib

def main():
    data = data_injection().load_data()
    x_train_val,y_train_val,x_val,y_val,x_test,y_test = data_split(data).split()
    trained_models= model_building(x_train_val,y_train_val).training()
    best_model_name,test_metrics = Evaluation(x_val,y_val,x_test,y_test,trained_models).evaluating_models()
    best_model = trained_models[best_model_name]

    background_data = x_train_val.sample(n = min(50,len(x_train_val)),random_state = randomstate)
    model_artifact = {
    "model": best_model,
    "background_data": background_data}

    joblib.dump(model_artifact,'artifacts/best_model.pkl')
    


if __name__ == "__main__":
    main()
