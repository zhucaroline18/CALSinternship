import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import math

dataset = pd.read_csv('smallModel/Calculated_Value_Dataset_Updated.csv')

target_features_comp = ['ME, kcal per kg','NDF,g',
                        'ADF,g','NFC,g','Crude fiber,g','Starch,g','CP,g',
                        'Arginine,g','Histidine,g','Isoleucine,g','Leucine,g',
                        'Lysine,g','Methionine,g','Phenylalanine,g',
                        'Threonine,g','Tryptophan,g','Valine,g','Alanine,g',
                        'Aspartic acid,g','Cystine,g','Met + Cys,g',
                        'Glutamic acid,g','Glycine,g','Proline,g','Serine,g',
                        'Tyrosine,g','Phe + Tyr,g','Ether extract,g','SFA,g',
                        'MUFA,g','PUFA,g','n-3 PUFA,g','n-6 PUFA,g',
                        'n-3:n-6 ratio,g','C14,g','C15:0,g','C15:1,g',
                        'C16:0,g','C16:1,g','C17:0,g','C17:1,g','C18:0,g',
                        'C18:1,g','C18:2 cis n-6 LA,g','C18:3 cis n-3 ALA,g',
                        'C20:0,g','C20:1,g','C20:4n-6 ARA,g','C20:5n-3 EPA,g',
                        'C22:0,g','C22:1,g','C22:5,g','C22:6n-3 DHA,g',
                        'C24:0,g','Ash,mg','Vitamin A IU per kg',
                        'beta-carotene,mg','Vitamin D3 IU per kg',
                        'Vitamin D3 25-Hydroxyvitamin D, IU','Vitamin E IU per kg',
                        'Vitamin K ppm','AST ppm','Thiamin ppm',
                        'Riboflavin ppm','Niacin ppm','Pantothenic acid ppm',
                        'Pyridoxine ppm','Biotin ppm','Folic acid ppm',
                        'Vitamin B12 ppm','Choline ppm','Calcium,g',
                        'Total Phosphorus,g','Inorganic available P,g',
                        'Ca:P ratio','Na,mg','Cl,mg','K,mg','Mg,mg','S,mg',
                        'Cu ppm','I ppm','Fe,mg','Mn,mg','Se,mg','Zn,mg']

fatty_acids = ['SFA,g',
                        'MUFA,g','PUFA,g','n-3 PUFA,g','n-6 PUFA,g',
                        'n-3:n-6 ratio,g','C14,g','C15:0,g','C15:1,g',
                        'C16:0,g','C16:1,g','C17:0,g','C17:1,g','C18:0,g',
                        'C18:1,g','C18:2 cis n-6 LA,g','C18:3 cis n-3 ALA,g',
                        'C20:0,g','C20:1,g','C20:4n-6 ARA,g','C20:5n-3 EPA,g',
                        'C22:0,g','C22:1,g','C22:5,g','C22:6n-3 DHA,g',
                        'C24:0,g']

target_labels_1 = ['average feed intake g per d','bodyweightgain,g']

target_labels_2 = ['akp U per ml','alt (U per L)','glucose (g per L)',
                 'nefa,umol per L','pip mg per dL','tc mg per g','tg mg per g',
                 'trap U per L','uric acid mmol per L','BCA']

#The following have too little data: Plasma C16:1, Plasma C18:1, Plasma C18:3, 
#Plasma C20:5, Liver C18:1
target_labels_3 = ['Plasma SFA','Plasma PUFA','Plasma n-3','Plasma n-6',
                 'Plasma C16:0 ','Plasma C18:0 ',
                 'Plasma C18:2 ','Plasma C22:6',
                 'Liver SFA','Liver MUFA','Liver PUFA','Liver n-3','Liver n-6',
                 'Liver C16:00 ','Liver C16:1 ','Liver C18:0 ',
                 'Liver C18:2','Liver C18:3 ','Liver C20:5','Liver C22:6',
                 'Breast SFA','Breast MUFA','Breast PUFA','Breast n-3',
                 'Breast n-6','Breast C16:0','Breast C16:01','Breast C18:0',
                 'Breast C18:01','Breast C18:2','Breast C18:3 ','Breast C20:4',
                 'Breast C20:5','Breast C22:6','Thigh SFA','Thigh MUFA',
                 'Thigh PUFA','Thigh n-3','Thigh n-6','Thigh C16:0',
                 'Thigh C16:01','Thigh C18:0','Thigh C18:1','Thigh C18:2',
                 'Thigh C18:3','Thigh C20:4','Thigh C22:6']

target_labels_4 = ['breast mTOR','breast S6K1','breast 4E-BP1','breast MURF1',
                   'breast MAFbx','breast AMPK','breast LAT1','breast CAT1',
                   'breast SNAT2','breast VDAC1','breast ANTKMT','breast AKT1',
                   'IGF1','IGFR','IRS1','FOXO1','LC3-1','MyoD','MyoG','Pax3',
                   'Pax7','Mrf4','Mrf5','liver mTOR','liver S6K1',
                   'liver 4E-BP1','liver MURF1','liver MAFbx','liver AMPK']

#target_features_comp = fatty_acids

def train(features, target):

    scaler = MinMaxScaler()

    temp = dataset.dropna(subset=target)
    data = temp[features]
    labels = temp[target]
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    data = data.fillna(data.mean())
    #data = data.fillna(0) #Necessary incase an entire column is NaN, but shouldn't affect anything




    # Split the data into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=100,        # Number of boosting stages
        learning_rate=0.1,       # Learning rate (shrinkage)
        max_depth=3,             # Maximum depth of each tree
        subsample=0.8,           # Fraction of samples used for fitting each tree
    )

    # Train the model
    model.fit(data_train, labels_train)

    # Make predictions
    pred_train = model.predict(data_train)
    pred_test = model.predict(data_test)

    residuals = labels_train - pred_train

    n = len(labels_train)
    sigma_squared = np.var(residuals)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)


    # Evaluate the model
    k = len(features)
    train_rmse = math.sqrt(mean_squared_error(labels_train, pred_train))
    train_r2 = r2_score(labels_train, pred_train)
    train_mape = mean_absolute_percentage_error(labels_train, pred_train)
    train_aic = 2 * k - 2 * (-n * log_likelihood)
    train_bic = k * math.log(n) - 2 * (-n * log_likelihood)

    test_rmse = math.sqrt(mean_squared_error(labels_test, pred_test))
    test_r2 = r2_score(labels_test, pred_test)
    test_mape = mean_absolute_percentage_error(labels_test, pred_test)
    
    print(f'''{target} evaluation:
    Training RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, MAPE: {train_mape:.4f}
    Testing RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, MAPE: {test_mape:.4f}''')
    
    metrics = pd.DataFrame({
        'Metric':['Training RMSE','Training R2','Training MAPE','Testing RMSE','Testing R2','Testing MAPE'],
        'Value':[train_rmse, train_r2, train_mape, test_rmse, test_r2, test_mape]
    })

    importance = pd.DataFrame({
        'Feature': data.columns,
        'Importance': model.feature_importances_
    })

    return metrics, importance

def evaluate(targets):
    for i in targets:
        metrics, importance = train(target_features_comp, i)
        
        importance.sort_values(by='Importance', ascending=False)
        importance['Rank'] = range(len(importance['Feature']))
        filtered_importance = importance[importance['Feature'].isin(fatty_acids)]

        metric_string = f'''{i} evaluation:
        Training RMSE: {metrics[0]:.4f}, R2: {metrics[1]:.4f}, MAPE: {metrics[2]:.4f}
        Testing RMSE: {metrics[3]:.4f}, R2: {metrics[4]:.4f}, MAPE: {metrics[5]:.4f}'''

        with open(f'smallModel/outputs/{i}.txt', "w") as file:
            file.write(metric_string + '\n\n' + importance.head().to_string() 
                       + '\n\n' + filtered_importance.to_string())
            
def fill_csv(name):

    importance_frame = pd.DataFrame({'Variable':target_features_comp})
    metric_frame = pd.DataFrame({'Metric':['Training RMSE','Training R2',
                                           'Training MAPE','Testing RMSE',
                                           'Testing R2','Testing MAPE']})

    targets = target_labels_1 + target_labels_2 + target_labels_3 + target_labels_4

    for label in targets:
        metrics, importance = train(target_features_comp, label)
        importance_frame[label] = importance['Importance']
        metric_frame[label] = metrics['Value']

    with open(f'smallModel/outputs/{name}_importances.csv', "w") as file:
        file.write(importance_frame.to_csv())
    with open(f'smallModel/outputs/{name}_metrics.csv', "w") as file:
        file.write(metric_frame.to_csv())


#evaluate(target_labels_1)
#evaluate(target_labels_2)
#evaluate(target_labels_3)
#evaluate(target_labels_4)

fill_csv('test')