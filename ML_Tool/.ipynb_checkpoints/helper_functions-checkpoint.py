import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

def load_data(data_path = "churn.csv"):

    true = ["yes",
            "True",
            "True."]

    false = ["no",
             "False",
             "False."]

    drop_col = ["State",
                "Account Length",
                "Area Code",
                "Phone"]

    df = pd.read_csv(data_path,
                     true_values = true, 
                     false_values = false)    
    
    
    
    df.drop(drop_col, 
            axis = 1, 
            inplace = True)

    df.rename(columns={'Churn?':'Churn'}, 
              inplace=True)

    # columns won't rescale unless dtype = "float"
    df2 = df[df.columns[2:-1]].astype(float)

    scaler = MinMaxScaler(feature_range=(-1,1))
    cols = df2.columns[2:-1]
    # scale int and float type cols
    for col in cols:
        df2[col] = scaler.fit_transform(df2[col].values.reshape((-1,1)))


    # Move columns with boolean values back into dataframe# Move  
    df2[df.columns[0]] = df[df.columns[0]].values
    df2[df.columns[1]] = df[df.columns[1]].values
    df2[df.columns[-1]] = df[df.columns[-1]].values
    
    return df2