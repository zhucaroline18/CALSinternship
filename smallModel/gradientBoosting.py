import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

dataset = pd.read_csv('smallModel/Calculated_Value_Dataset_Updated.csv')

target_features_comp = ['time period','ME','kcal per kg','Overall','NDF,g',
                        'ADF,g','NFC,g','Crude fiber,g','Starch,g','CP,g',
                        'Arginine,g','Histidine,g','Isoleucine,g','Leucine,g',
                        'Lysine,g','Methionine,g','Phenylalanine,g',
                        'Threonine,g','Tryptophan,g','Valine,g','Alanine,g',
                        'Aspartic acid,g','Cystine,g','Met + Cys,g',
                        'Glutamic acid,g','Glycine,g','Proline,g','Serine,g',
                        'Tyrosine,g','Phe + Tyr,g','Ether extract,g','SFA,g',
                        'MUFA,g','PUFA,g','n-3 PUFA,g','n-6 PUFA,g',
                        'n-3/n-6 ratio,g','C14,g','C15:0,g','C15:1,g',
                        'C16:0,g','C16:1,g','C17:0,g','C17:1,g','C18:0,g',
                        'C18:1,g','C18:2 cis n-6 LA,g','C18:3 cis n-3 ALA,g',
                        'C20:0,g','C20:1,g','C20:4n-6 ARA,g','C20:5n-3 EPA,g',
                        'C22:0,g','C22:1,g','C22:5,g','C22:6n-3 DHA,g',
                        'C24:0,g','Ash,mg','Vitamin A IU/kg',
                        'beta-carotene,mg','Vitamin D3 IU/kg',
                        'Vitamin D3 25-Hydroxyvitamin D, IU','Vitamin E IU/kg',
                        'Vitamin K ppm','AST ppm','Thiamin ppm',
                        'Riboflavin ppm','Niacin ppm','Pantothenic acid ppm',
                        'Pyridoxine ppm','Biotin ppm','Folic acid ppm',
                        'Vitamin B12 ppm','Choline ppm','Calcium,g',
                        'Total Phosphorus,g','Inorganic available P,g',
                        'Ca:P ratio','Na,mg','Cl,mg','K,mg','Mg,mg','S,mg',
                        'Cu ppm','I ppm','Fe,mg','Mn,mg','Se,mg','Zn,mg']
target_labels = []

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
model.fit(data, labels)

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


