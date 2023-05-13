import re
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from helper_files.helpers_ import DataAnalysisPipeline
import warnings
from helper_files.dictionaries import fill_na_dict, ordinal_encode_dict, numerical_, hyperparameters, outlier_list

def model_pipeline(data, path, csv=True):
    
    ### FILLING NA VALUES
    for key, value in fill_na_dict.items():
        data[key] = data[key].fillna(value)
    for index in range(len(data)):
        if (data["MasVnrType"].iloc[index] == "None") & (data["MasVnrArea"].iloc[index] > 0):
            data.at[index, 'MasVnrArea'] = 0
    
    data["LotFrontage"].fillna(data["LotFrontage"].median(), inplace=True)
    
    data["Electrical"].fillna(data["Electrical"].value_counts().idxmax(), inplace=True)

    for col in data.columns:
        if (data[col].isnull().any() == True):
            if data[col].dtype != object:
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].value_counts().idxmax(), inplace=True)

    ### OUTLIERS
    for col in outlier_list:  
        DataAnalysisPipeline(data).iqr_values(col=col, 
                                        upper_quantile=0.90, 
                                        lower_quantile=0.01, 
                                        info=False, 
                                        replace_with_limit=True)

    ### ENCODING 
    not_ordinal = []
    for key, value in ordinal_encode_dict.items():
        data[key] = data[key].fillna(ordinal_encode_dict[key][0])
        try:
            DataAnalysisPipeline(data).ordinal_encoding(col=key, 
                                                  ordinal_units_list=value,
                                                  add_ordinal_list_name="No")
        except:
            print("key :",key, "nunique :", data[key].nunique(), "na_val :", data[key].isnull().sum())
            not_ordinal.append(key)

    ###NEW FEATURES 
    data["NEW_AGE_FOR_MONTH"] = ((data["YrSold"] - data["YearBuilt"]) * 12) + data["MoSold"]

    data["NEW_REMOD_AGE_FOR_MONTH"] = ((data["YrSold"] - data["YearRemodAdd"]) * 12) + data["MoSold"]

    data["NEW_TOTAL_PORCH"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]

    data["NEW_GARAGE_AREA_PER_CAR"] = (data["GarageArea"] / data["GarageCars"]) * data["GarageFinish"]

    data["NEW_GARAGE_AGE_FOR_MONTH"] = ((data["YrSold"] - data["GarageYrBlt"]) * 12) + data["MoSold"]

    data["NEW_AREA_PER_ROOM_ABOVE_GRADE"] =  data["TotRmsAbvGrd"] / data["GrLivArea"]

    data["NEW_CENTRAL_AIR_FIREPLACES"] = data["Fireplaces"] + data["CentralAir"]

    data["NEW_KITCHEN_TOTAL_ROOMS"] = data["KitchenAbvGr"] / (data["BsmtFullBath"] + data["BsmtHalfBath"] + data["FullBath"] + data["HalfBath"] + data["BedroomAbvGr"] + data["KitchenAbvGr"] + data["TotRmsAbvGrd"])

    data["NEW_GARAGE_AREA_PER_CAR"].fillna(0, inplace=True)

    data.drop(["MiscFeature", "MiscVal"], axis=1, inplace=True)

    ### SCALING
    # Numerical Scale
    rs = RobustScaler()
    for col in numerical_:
        if (col != "SalePrice") & (col != "Id"):
            array = np.array(data[col]) 
            array = array.reshape(-1, 1)
            data[col] = rs.fit_transform(array)

    # Categorical Scale
    mms = MinMaxScaler()
    for col in data.columns:
        try:
            if (col not in numerical_) & (col != "SalePrice") & (col != "Id"):
                array = np.array(data[col]) 
                array = array.reshape(-1, 1)
                data[col] = mms.fit_transform(array)
        except:
            print(col)

    data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    ### MODEL

    model_path = input("Please input model path :")
    model = lgb.Booster(model_file=model_path)

    X = data.drop(["Id", "SalePrice", "YearBuilt", "YearRemodAdd",
         "GarageYrBlt"], axis=1)
    
    y = data["SalePrice"]        

    predictions = model.predict(X)
    prediction_rmse = np.sqrt(mean_squared_error(y, predictions))
    print("Entire Data RMSE: ", prediction_rmse)
    
    predictions = pd.DataFrame(data=predictions, columns=['Predictions_Y'])
    predictions["Based_Y"] = data["SalePrice"]
    predictions["Id"] = data["Id"]

    print("\n", predictions.head(10))

    if csv:
        predictions.to_excel(path)

    predictions["diff"] = predictions["Based_Y"] - predictions['Predictions_Y']
    predictions["diff%"] = predictions["diff"].abs() / predictions["Based_Y"]
    
    print("\nNumber of units that (prediction-reel)/reel : x => x < %10 ::" , predictions[predictions["diff%"] <= 0.1].shape[0])
    print("\nNumber of units that (prediction-reel)/reel : x => %10 < x < %15 ::" , predictions[(predictions["diff%"] > 0.1) & (predictions["diff%"] <= 0.15)].shape[0])
    print("\nNumber of units that (prediction-reel)/reel : x => %15 < x < %20 ::" , predictions[(predictions["diff%"] > 0.15) & (predictions["diff%"] <= 0.20)].shape[0])
    print("\nNumber of units that (prediction-reel)/reel : x => %20 < x < %25 ::" , predictions[(predictions["diff%"] > 0.20) & (predictions["diff%"] <= 0.25)].shape[0])
    print("\nNumber of units that (prediction-reel)/reel : x => %25 < x ::" , predictions[(predictions["diff%"] > 0.25)].shape[0])

    
    return print("Prediction finished.")


def main():
    path_x = input(r"Please input the path of X :")
    path_y = input(r"Please input the path of y :")

    path = input(r"Please input the path of predictions that you save :")

    dataframe_x = pd.read_csv(path_x)

    if path_y == "No":
        dataframe = dataframe_x
        model_pipeline(data=dataframe, path=path)
        return print("Function is finished.")

    dataframe_y = pd.read_csv(path_y)
    dataframe = dataframe_x.merge(dataframe_y, on="Id")

    model_pipeline(data=dataframe, path=path)
    return print("Function finished.")

if __name__ == '__main__':
    main()
