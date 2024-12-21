import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

dataset = pd.read_csv('smallModel/Calculated_Value_Dataset_Updated.csv')

target_features_comp = ['ME, kcal per kg','Overall','NDF,g',
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

target_labels_1 = ['average feed intake g per d','bodyweightgain,g']

target_labels_2 = ['akp U per ml','alt (U per L)','glucose (g per L)',
                 'nefa,umol per L','pip mg per dL','tc mg per g','tg mg per g',
                 'trap U per L','uric acid mmol per L','BCA']

target_labels_3 = ['Plasma SFA','Plasma MUFA','Plasma PUFA','Plasma n-3','Plasma n-6',
                 'Plasma C16:0 ','Plasma C16:1 ','Plasma C18:0 ','Plasma C18:1 ',
                 'Plasma C18:2 ','Plasma C18:3 ','Plasma C20:5','Plasma C22:6',
                 'Liver SFA','Liver MUFA','Liver PUFA','Liver n-3','Liver n-6',
                 'Liver C16:00 ','Liver C16:1 ','Liver C18:0 ','Liver C18:1',
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

def train(features, target):

    temp = dataset.dropna()
    data = temp[features]
    data = data.fillna(data.mean())
    labels = temp[target]


    # Split the data into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=100,        # Number of boosting stages
        learning_rate=0.1,       # Learning rate (shrinkage)
        max_depth=3,             # Maximum depth of each tree
        subsample=0.8,           # Fraction of samples used for fitting each tree
        random_state=42          # For reproducibility
    )

    # Train the model
    model.fit(data_train, labels_train)

    # Make predictions
    pred_train = model.predict(data_train)
    pred_test = model.predict(data_test)

    # Evaluate the model
    train_mse = mean_squared_error(labels_train, pred_train)
    test_mse = mean_squared_error(labels_test, pred_test)
    train_r2 = r2_score(labels_train, pred_train)
    test_r2 = r2_score(labels_test, pred_test)

    print(f'Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}')
    print(f'Testing MSE: {test_mse:.4f}, R2: {test_r2:.4f}')

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': data.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(importance_df)

for i in target_labels_1:
    print(i)
    train(target_features_comp, i)
