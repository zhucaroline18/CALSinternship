import os 
os.environ['GROQ_API_KEY'] = ""
from groq import Groq
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

filename = 'totalDataLLM.csv'

class Agent:
    def __init__(self, client: Groq, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message = ""):
        if message: 
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="llama3-70b-8192", messages=self.messages
        )
        return completion.choices[0].message.content

#specify to look at all the nutrients
system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

You will be asked questions about nutrient intake and how it affects chicken health. Don't make assumptions about which variables are important. You should test all of them. 

Your available actions are:

pearson_correlation:
e.g. pearson_correlation(['Vitamin A IU/kg', 'thigh PH', 'beta-carotene'])
takes in a list of column names regarding the columns of interest. This can refer to a nutrient intake inputs for chickens or the chicken health parameter outputs. The columns of interest should only include numerical values. 
The function returns the correlation matrix in the form of a NumPy matrix between the columns of interest. The correlation matrix refers to the Pearson correlation matrix where r = 1 represents a perfect positive linear relationship, r = -1 represents a perfect negative linear relationship and r = 0 represents no linear relationships. 

p_value_of_correlation:
e.g. p_value_of_correlation(['Vitamin A IU/kg', 'thigh PH', 'beta-carotene'])
takes in a list of column names regarding the columns of interest. This can refer to a nutrient intake inputs for chickens or the chicken health parameter outputs. The columns of interest should only include numerical values. 
The value it returns is the p value matrix in the form of a NumPy matrix. It tells you the p-value of each relationship and whether or not that correlation is significant.

get_inputs:
e.g. get_inputs()
returns a list of the available input columns to examine. The first 5 columns contain string data and the rest contain numerical data. All the inputs are considered nutrients

get_outputs:
e.g. get_outputs()
returns a list of the available output columns to examine.

get_all_data():
e.g. get_all_data()
Returns all the inputs and outputs in the form of a NumPy matrix. All the available data will be returned. In each row of the matrix returned, the first 92 values will be input and the rest will be output

In each session, you will be called until you have an answer, in which case, output it as the Answer.
Now begin.
""".strip()

def pearson_correlation(columns):
    df_original = pd.read_csv(filename)
    for i in columns:
        if i not in df_original.columns:
            return f"{i} not a valid column"
    df = df_original[columns].dropna() # not sure if i should drop na or not?
    corr_matrix = df.corr()
    return(corr_matrix)

def p_value_of_correlation(columns): 
    df_original = pd.read_csv(filename) 
    df = df_original[columns].dropna() 
    for i in columns:
        if i not in df.columns: 
            return f"{i} not a valid column"
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
    
    return corr_pvalues(df).to_numpy()

def get_all_data():
    df = pd.read_csv(filename)
    return df.to_numpy()

def get_inputs():
    df = pd.read_csv(filename)
    return df.columns[:92].tolist()

def get_outputs():
    df = pd.read_csv(filename)
    return df.columns[92:].tolist()

def loop(max_iterations = 25, query: str = ""):
    agent = Agent(client=client, system=system_prompt)

    tools = ["pearson_correlation", "p_value_of_correlation", "get_inputs", "get_outputs"]

    next_prompt = query

    i = 0

    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            pattern = (r"Action:\s*(\w+)\s*\(([^)]*)\)")

            match = re.search(pattern, result)

            if match:
                method_name = match.group(1)
                data_list = match.group(2)
            else:
                print("No match found")

            if method_name in tools:
                result_tool = eval(f"{method_name}({data_list})")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue

        if "Answer" in result:
            break

if __name__ == "__main__":
    loop(query = "what should I feed a chicken in order to affect gene expression in the breast?")
