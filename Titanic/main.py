import pandas as pd
import numpy as nu
import Processing  as pr
import MainModel as mm


def main():
    pr_Inst = pr.Get_Data()
    test_data, df = pr_Inst.read_data()
    df = pr_Inst.missing_value(df)
    df_ = df.copy()
    df_ = pr_Inst.feature_and_processing(df_)
    display(df_.head())
    print(df_.columns)

    data_Inst = mm.Create_Dataset()
    X, y, test_X = data_Inst.split_data(df_)
    X_train, X_test, y_train, y_test = data_Inst.split_train_test(X, y)

    
    ml_Inst = mm.Model_Trial()
    model = ml_Inst.model_svc(X_train, y_train)
    accuracy, cl_report = ml_Inst.model_validate(model, X_test, y_test)
    print(f"accuracy_score: {accuracy}")
    print("score_validation")
    print(cl_report)
    
    mo_Inst = mm.Model_Output()
    mo_Inst.set_param_grid({'kernel': ['linear', 'rbf', 'poly'], 'C': [2**i for i in range(-2, 3)]})
    mo_Inst.set_cv(n_splits=10, n_repeats=3, random_state=0)
    mo_Inst.parameter_tuning_fit(X, y)
    y_pred = mo_Inst.predict(test_X) 
    mo_Inst.result_to_csv(test_data, y_pred)
    
    
if __name__ == '__main__' :
    main()
# result score 0.78708
