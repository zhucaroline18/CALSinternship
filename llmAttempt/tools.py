import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# columns is a list of the column names of the columns of interest.
# this can refer to an input or output in our data where the data
# represents nutritional intake inputs for chickens and the health 
# parameter outputs. The columns of interest should only include numerical values.
# filename is the location of the csv file that contains the data.
# The function returns the correlation matrix between the columns 
# of interest. the correlation matrix refers to the Pearson correlation 
# matrix where r = 1 represents a perfect positive linear relationship, 
# r = -1 represents a perfect negative linear relationship and r = 0
# represents no linear relationships 
def pearson_correlation(columns, filename):
    df = pd.read_csv(filename)

    df_selected = df[columns]
    correlation_matrix = df_selected.corr(method = 'pearson')

    return correlation_matrix

def getColumns(filename):
    df = pd.read_csv(filename)
    return df.columns

def get_all_data(filename):
    df = pd.read_csv(filename)
    return df.to_numpy()

def pearson_correlation2(columns, filename):
    df_original = pd.read_csv(filename)
    df = df_original[columns].dropna()   #not sure if i should drop na or not?
    corr_matrix = df.corr()

    def corr_pvalues(df):
        df = df.dropna()
        pvals = pd.DataFrame(np.ones((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns)
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    _, pval = pearsonr(df[col1], df[col2])
                    pvals.loc[col1, col2] = pval
        return pvals
    
    pval_matrix = corr_pvalues(df)
    print("Correlation matrix:\n", corr_matrix)
    print("\nP-value matrix:\n", pval_matrix)

if __name__ == "__main__":
    columns = ['Vitamin A IU/kg','thigh PH', 'beta-carotene']
    #print(getColumns('totalDataLLM.csv'))
    print(get_all_data('totalDataLLM2.csv'))
    #pearson_correlation2(columns, 'totalDataLLM2.csv')