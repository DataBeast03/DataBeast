import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import  MinMaxScaler

def load_process_save_classification(data_path = "churn.csv"):
    '''Load, clean, and save churn data to pickle file '''

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


    # Move columns with boolean values back into dataframe  
    df2[df.columns[0]] = df[df.columns[0]].values
    df2[df.columns[1]] = df[df.columns[1]].values
    df2[df.columns[-1]] = df[df.columns[-1]].values
    
    X_churn = df2[df2.columns[:-1]].values
    Y_churn = df2.Churn.values
    
    classification_data = "./ETL_data/churn.pkl"
    pickle.dump([X_churn, Y_churn], open(classification_data, "wb"))

def load_process_save_regression(data_path = "./google-play-store-apps/googleplaystore.csv"):
    '''Load, clean, and save google playstore data to pickle file'''
    # load data ----
    df = pd.read_csv(data_path)

    # size column ---- 
    def size_text_to_int(size):
        if "M" in size:
            size = size.rstrip("M")
            if "." in size:
                size = "".join(size.split("."))
            size = size + "00000"
            size = int(size)
        else:
            size = size.rstrip("k")
            if "." in size:
                size = "".join(size.split("."))
            size = size + "00"
            size = int(size)

        return size

    category_labels = df.Category.unique()
    category_ids = np.arange(0, category_labels.shape[0])

    # replace text category labels with index category labels
    df.replace(to_replace=category_labels, value=category_ids, inplace=True)

    # keep apps with Millions or Thousands Bytes in Size
    size_vals = [val for val in df.Size.unique() if "M"  in val or  "k" in val]

    # create size mask to drop all non M or k vals
    size_mask = df.Size.isin(size_vals)
    df = df[size_mask]

    df.Size = df.Size.apply(lambda size: size_text_to_int(size))



    # intalls column ------ 

    def text_to_int(install_num):
        install_num = install_num.rstrip("+")
        install_num = "".join(install_num.split(","))
        return int(install_num)

    df.Installs=df.Installs.apply(lambda text_val: text_to_int(text_val))

    # text column ------

    # replace "free" and "paided" types to booleans
    type_label = df.Type.unique()
    df.Type.replace(to_replace=type_label, value=[0,1], inplace=True)

    # rating column ------

    # replace text ratings with ints
    rating_labels = df["Content Rating"].unique()

    rating_indices = np.arange(0, rating_labels.shape[0])

    df["Content Rating"].replace(to_replace=rating_labels, value=rating_indices, inplace=True)

    # genre column ------

    # replace text labels as ints
    genres_labels = df.Genres.unique()

    genre_indices = np.arange(0, genres_labels.shape[0])

    df.Genres.replace(to_replace=genres_labels, value=genre_indices, inplace=True)

    # drop columns ------

    # drop these columns
    cols_to_drop = ["App", "Current Ver", "Android Ver", "Last Updated"]
    df.drop(columns=cols_to_drop, inplace=True)

    # price column ------

    # clean up price values
    def price_to_float(price):
        if "$" in price:
            price = price.lstrip("$")    
        return float(price)

    df.Price=df.Price.apply(lambda price: price_to_float(price))

    # drop NAs ------

    # drop NAs
    df.dropna(inplace=True)

    # type cast columns -------

    # type cast these columns
    df.Rating = df.Rating.astype("int")
    df.Reviews = df.Reviews.astype("int")
    df.Price = df.Price.astype("int")

    # split data into X and Y ------

    x_cols = df.columns[2:-1].tolist()
    x_cols.append("Category")

    X_data = df[x_cols].values
    Y_data = df.Rating.values

    # save data ------

    regression_data = "./ETL_data/googleAppStore_data.pkl"
    pickle.dump([X_data, Y_data], open(regression_data, "wb"))