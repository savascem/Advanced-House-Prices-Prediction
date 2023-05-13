import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, auc

def plot_importance(model, features, feature_num=20,save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:feature_num])
    plt.title('Features Importances')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

def error_metrics(x_train_or_test, y_train_or_test, title, model_type, final_model=None, metric="rmse", plot_graph=False, figure_height=10, figure_size=10):
    for i in range(len(x_train_or_test)):
        y_pred = final_model.predict(x_train_or_test[i])
        mae = mean_absolute_error(y_train_or_test[i], y_pred)
        mse = mean_squared_error(y_train_or_test[i], y_pred)
        rmse = mean_squared_error(y_train_or_test[i], y_pred, squared=False)
        r2 = r2_score(y_train_or_test[i], y_pred)

        y_true_log = np.log1p(y_train_or_test[i])
        y_pred_log = np.log1p(y_pred)
        rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

        print(f"\n\n{title[i]} Error:\n")
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R^2 Score:", r2)
        print("RMSLE : ", rmsle)
    
    if plot_graph:
        if model_type == 'LGBM':            
            train_scores = final_model.evals_result_['training'][metric]
            test_scores = final_model.evals_result_['valid_1'][metric]

            train_area = auc(range(len(train_scores)), train_scores)
            test_area = auc(range(len(test_scores)), test_scores)
            area_between_curves = abs(train_area - test_area)

            print("\n\nAreas under & between curves: {:.2f}".format(train_area))
            print("\nArea Under Train set:")
            print("Area Under Test set: {:.2f}".format(test_area))
            print("Area Between curves: {:.2f}".format(area_between_curves))
            print("\n\nError Graphic for per iter:")

            plt.figure(figsize=(figure_size, figure_height))
            plt.plot(train_scores, label='Train')
            plt.plot(test_scores, label='Test')
            plt.fill_between(range(len(train_scores)), train_scores, test_scores, alpha=0.1, color='orange')
            plt.text(len(train_scores)//2, (max(train_scores)+min(test_scores))/2, 
                     "Area Between curves: {:.2f}".format(area_between_curves), ha="center", va="center")
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('LGBM Model')
            plt.legend()
            plt.show()
        
        elif model_type == 'XGBOOST':
            train_scores = final_model.evals_result()['validation_0'][metric]
            test_scores = final_model.evals_result()['validation_1'][metric]

            train_area = auc(range(len(train_scores)), train_scores)
            test_area = auc(range(len(test_scores)), test_scores)
            area_between_curves = abs(train_area - test_area)

            print("\n\nAreas under & between curves: {:.2f}".format(train_area))
            print("\nArea Under Train set:")
            print("Area Under Test set: {:.2f}".format(test_area))
            print("Area Between curves: {:.2f}".format(area_between_curves))
            print("\n\nError Graphic for per iter:")

            plt.figure(figsize=(figure_size, figure_height))
            plt.plot(range(1, len(train_scores)+1), train_scores, label='Train')
            plt.plot(range(1, len(test_scores)+1), test_scores, label='Test')
            plt.fill_between(range(len(train_scores)), train_scores, test_scores, alpha=0.1, color='orange')
            plt.text(len(train_scores)//2, (max(train_scores)+min(test_scores))/2, 
                     "Area Between curves: {:.2f}".format(area_between_curves), ha="center", va="center")
            plt.xlabel('n_estimators')
            plt.ylabel('RMSE')
            plt.title('XGBoost Model')
            plt.legend()
            plt.show()
        else:
            print("Please enter your 'model_type'!!!")


class DataAnalysisPipeline:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def info(self, only_isnull=False):
        columns_info = pd.DataFrame({"nunique" : self.dataframe.nunique().values,
                                     "isnull" : self.dataframe.isnull().sum().values,
                                     "type" : self.dataframe.dtypes.values}, 
                                     index = self.dataframe.nunique().index)
        columns_info = pd.concat([columns_info, self.dataframe.describe().T], axis=1)
        columns_info["iscat"] = columns_info["nunique"].apply(lambda x : "Categorical" if x < 10 else "Numerical")
        if only_isnull:
            return columns_info[columns_info["isnull"] > 0]
        else:
            return columns_info
        
    def get_variable_type(self, id, threshold=10):
        numerical_but_categorial = []
        categorical = []
        categorical_over_threshold = []
        numerical = []
        for col in self.dataframe.columns:
            if col != id:
                if (self.dataframe[col].nunique() <= threshold) & (self.dataframe[col].dtype != object) & (self.dataframe[col].dtype != bool):
                    numerical_but_categorial.append(col)
                elif (self.dataframe[col].dtype == bool) | ((self.dataframe[col].dtype == object) & (self.dataframe[col].nunique() <= threshold)):
                    categorical.append(col)
                elif (self.dataframe[col].dtype == object) & (self.dataframe[col].nunique() > threshold):
                    categorical_over_threshold.append(col)
                else:
                    numerical.append(col)
        return numerical, numerical_but_categorial, categorical, categorical_over_threshold
    
    def analysis_target_with_observation_units(self, target, id="id", observation_unit="all", threshold = 10):
    
        def main_func(observation_unit, target, id, threshold = threshold):
            if (observation_unit != target) & (self.dataframe[observation_unit].nunique() < threshold):
                if (self.dataframe[target].dtype == object) or (self.dataframe[target].dtype == bool):
                    ratio_df = self.dataframe.groupby([observation_unit,target]).agg({id:["count", lambda x: x.count()/len(self.dataframe)]})
                    ratio_df.columns = ["count", "ratio"]
                    print("\n", f"***** {observation_unit} *****")
                    print(ratio_df)
                    
                else:
                    ratio_df = self.dataframe.groupby([observation_unit]).agg({target:["count", 
                                                                                      lambda x: x.count()/len(self.dataframe), 
                                                                                      "mean", 
                                                                                      "median",
                                                                                      lambda x: x.min(),
                                                                                      lambda x: x.max(),
                                                                                      "sum",
                                                                                      lambda x: x.sum()/self.dataframe[target].sum()]})
                    ratio_df.columns = ["count", "count_ratio", "mean", "median", "min", "max", "sum", "sum_ratio"]
                    print("\n", f"***** {observation_unit} *****")
                    print(ratio_df)

            else:
                return None

        if observation_unit == "all":
            for col in self.dataframe.columns:
                main_func(observation_unit=col , target=target, id=id, threshold = threshold)

        else:
            main_func(observation_unit=observation_unit , target=target, id=id, threshold = threshold)

        return None
    
    def iqr_values(self, col, upper_quantile=0.75, lower_quantile=0.25, info=False, replace_with_limit=False):
        IQR = self.dataframe[col].quantile(0.75) - self.dataframe[col].quantile(0.25)
        lower_limit = self.dataframe[col].quantile(lower_quantile) - IQR*1.5
        upper_limit = self.dataframe[col].quantile(upper_quantile) + IQR*1.5
        under_lower = self.dataframe[self.dataframe[col] < lower_limit].shape[0]
        over_upper = self.dataframe[self.dataframe[col] > upper_limit].shape[0]

        if replace_with_limit:
            for index, val in enumerate(self.dataframe[col]):
                if val > upper_limit:
                    self.dataframe[col].iloc[index] = upper_limit
                elif val < lower_limit:
                    self.dataframe[col].iloc[index] = lower_limit

        if info:
            print(f"Low Limit : {lower_limit}\n",
                  f"Up Limit : {upper_limit}\n",
                  f"{under_lower} observation units under 'Low Limit'\n",
                  f"{over_upper} observation units over 'Up Limit'")


        return lower_limit, upper_limit, under_lower, over_upper

    def numerical_plot(self, col, target):
        fig, axes = plt.subplots(ncols=3, figsize=(22,8))

        sns.set(style="darkgrid")
        sns.histplot(self.dataframe[col], color='red',ax=axes[0])
        axes[0].set_title(f"Histogram of {col}", fontsize=17.5, color='red')

        sns.scatterplot(x=self.dataframe[col], y=self.dataframe[target], color='green', ax=axes[1])
        axes[1].set_title(f"Correlation of {col} Data", fontsize=17.5, color='green')
        axes[1].set_xlabel(f"{col}")
        axes[1].set_ylabel(target)

        lower_limit, upper_limit, under_lower, over_upper = self.iqr_values(col, upper_quantile=0.75, lower_quantile=0.25)

        axes[2].boxplot(self.dataframe[col], vert=False)
        axes[2].set_title(f"Boxplot of {col}", fontsize=17.5, color='darkblue')

        axes[2].text(20, 1.45, f"Low Limit : {lower_limit}", color='blue', fontsize=15)
        axes[2].text(20, 1.4, f"Up Limit : {upper_limit}", color='blue', fontsize=15)
        axes[2].text(20, 1.35, f"{under_lower} observation units under 'Low Limit'", color='blue', fontsize=15)
        axes[2].text(20, 1.3, f"{over_upper} observation units over 'Up Limit'", color='blue', fontsize=15)

        fig.suptitle(f"{col} | Number of Unique values for '{col}' : {self.dataframe[col].nunique()}", fontsize=20)

        plt.show()

    def combine_units(self, target, observation_units):
        ratio_df = self.dataframe.groupby(observation_units).agg({target:["count", 
                                                                      lambda x: x.count()/len(self.dataframe), 
                                                                      "mean", 
                                                                      "median",
                                                                      lambda x: x.min(),
                                                                      lambda x: x.max(),
                                                                      "sum",
                                                                      lambda x: x.sum()/self.dataframe[target].sum()]})
        ratio_df.columns = ["count", "count_ratio", "mean", "median", "min", "max", "sum", "sum_ratio"]
        return ratio_df

    def ordinal_encoding(self, col, ordinal_units_list, add_ordinal_list_name="No", new_feature=False):
        ordinal_encoder = OrdinalEncoder(categories=[ordinal_units_list])
        array = np.array(self.dataframe[col]) 
        array = array.reshape(-1, 1)

        if new_feature:
            self.dataframe[f"NEW_{col}"] = ordinal_encoder.fit_transform(array)
        else:
            self.dataframe[col] = ordinal_encoder.fit_transform(array)

        if add_ordinal_list_name != "No":
            add_ordinal_list_name.append(col)

