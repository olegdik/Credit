import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

# model = joblib.load('model/loan_pipe.pkl')
file_name = 'model/credit.pkl'
with open(file_name, 'rb') as file:
    model = dill.load(file)
    # print(model['metadata'])

#

class FormOriginal(BaseModel):
    id: int
    rn: int
    pre_since_opened: int
    pre_since_confirmed: int
    pre_pterm: int
    pre_fterm: int
    pre_till_pclose: int
    pre_till_fclose: int
    pre_loans_credit_limit: int
    pre_loans_next_pay_summ: int
    pre_loans_outstanding: int
    pre_loans_total_overdue: int
    pre_loans_max_overdue_sum: int
    pre_loans_credit_cost_rate: int
    pre_loans5: int
    pre_loans530: int
    pre_loans3060: int
    pre_loans6090: int
    pre_loans90: int
    is_zero_loans5: int
    is_zero_loans530: int
    is_zero_loans3060: int
    is_zero_loans6090: int
    is_zero_loans90: int
    pre_util: int
    pre_over2limit: int
    pre_maxover2limit: int
    is_zero_util: int
    is_zero_over2limit: int
    is_zero_maxover2limit: int
    enc_paym_0: int
    enc_paym_1: int
    enc_paym_2: int
    enc_paym_3: int
    enc_paym_4: int
    enc_paym_5: int
    enc_paym_6: int
    enc_paym_7: int
    enc_paym_8: int
    enc_paym_9: int
    enc_paym_10: int
    enc_paym_11: int
    enc_paym_12: int
    enc_paym_13: int
    enc_paym_14: int
    enc_paym_15: int
    enc_paym_16: int
    enc_paym_17: int
    enc_paym_18: int
    enc_paym_19: int
    enc_paym_20: int
    enc_paym_21: int
    enc_paym_22: int
    enc_paym_23: int
    enc_paym_24: int
    enc_loans_account_holder_type: int
    enc_loans_credit_status: int
    enc_loans_credit_type: int
    enc_loans_account_cur: int
    pclose_flag: int
    fclose_flag: int


class Prediction(BaseModel):
    pred: int


@app.post('/predict', response_model=Prediction)
def predict(form: FormOriginal):
    def delete(df):
        df = df.drop(['id', 'rn', 'enc_paym_0', 'enc_paym_1', 'enc_paym_2',
       'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7',
       'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12',
       'enc_paym_13', 'enc_paym_14', 'enc_paym_15', 'enc_paym_16',
       'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20',
       'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24','pclose_flag',
       'fclose_flag'], axis=1)
        return df

    def union(df):
        df['is_zero_loans'] = df[['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060',
                                  'is_zero_loans6090', 'is_zero_loans90']].apply(
            lambda row: 1 if all(val == 1 for val in row) else 0, axis=1)
        df = df.drop(
            ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90'], axis=1)
        return df

    df = pd.DataFrame.from_dict([form.dict()])
    delete(df)
    union(df)
    y = model['model'].predict(df)

    return {
        "pred": y[0]
    }


@app.get('/status')
def status():
    return "I'm OK!"


@app.get('/version')
def version():
    return model['metadata']

def main():
    def delete(df):
        df = df.drop(['id', 'rn', 'enc_paym_0', 'enc_paym_1', 'enc_paym_2',
       'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7',
       'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12',
       'enc_paym_13', 'enc_paym_14', 'enc_paym_15', 'enc_paym_16',
       'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20',
       'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24','pclose_flag',
       'fclose_flag'], axis=1)


    def union(df):
        df['is_zero_loans'] =  df[['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060',
       'is_zero_loans6090', 'is_zero_loans90']].apply(lambda row: 1 if all(val == 1 for val in row) else 0, axis=1)
        df = df.drop(
            ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90'], axis=1)


    with open('model/data/2.json') as json_file:
        test = json.load(json_file)
        print(test[0])
        #q = dict(test)
    df = pd.DataFrame.from_dict([test][0])
    delete(df)
    union(df)
    y = model['model'].predict(df)

    print(y[0])


if __name__ == '__main__':
    main()

