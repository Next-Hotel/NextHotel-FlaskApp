import pandas as pd
import json

def rekomendasi_hotel(data, new_sorted): 
        output_dataset = pd.read_csv("https://storage.googleapis.com/data-hotel/list-hotel/output_dataset.csv", encoding='unicode_escape')
        df = pd.read_csv("https://storage.googleapis.com/data-hotel/list-hotel/data_preprocessing.csv", encoding='unicode_escape')
        
        # Input Parameter
        test = pd.read_json(json.dumps(data))
        test = test['interest']
        weights = []
        for index, row in new_sorted.iterrows():
            for i in range(len(test)):
                weights.append(new_sorted['{}'.format(test.iat[i])])
            break
        for i in range(len(test)):
            # weights[i] = weights[i] / new_sorted['{}'.format(test.iat[i])].max()
            weights[i] = weights[i] * 2
        
        df_test = df[test]
        df_test['predict_score'] = new_sorted['predict_score']
        df_test = df_test[:70]
        df_test['final_score'] = df_test.sum(axis=1)

        output_dataset['final_score'] = df_test['final_score']
        output_dataset = output_dataset.sort_values(by=['final_score'], ascending=False)

        # return data dalam bentuk json
        return output_dataset.to_json(orient ='table')
