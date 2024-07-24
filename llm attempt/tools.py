
import numpy as np
from csv import reader
from scipy import stats
import pandas as pd

# columns is a list of the column names of the columns of interest.
# this can refer to an input or output in our data where the data
# represents nutritional intake inputs for chickens and the health 
# parameter outputs 
# filename is the location of the csv file that contains the data.
# The function returns the correlation matrix between the columns 
# of interest. the correlation matrix refers to the Pearson correlation 
# matrix where r = 1 represents a perfect positive linear relationship, 
# r = -1 represents a perfect negative linear relationship and r = 0
# represents no linear relationships 
def ANOVA(columns, filename):
    df = pd.read_csv(filename)

    df_selected = df[columns]
    correlation_matrix = df_selected.corr(method = 'pearson')

    return correlation_matrix