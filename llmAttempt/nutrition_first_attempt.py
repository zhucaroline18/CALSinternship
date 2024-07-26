import os 
# os.environ['GROQ_API_KEY'] = "?"
from groq import Groq
import re

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

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

pearson_correlation:
e.g. pearson_correlation(['Vitamin A IU/kg', 'thigh PH', 'beta-carotene'])
takes in a list of column names regarding the columns of interest. This can refer to a nutrient intake inputs for chickens or the chicken health parameter outputs. The columns of interest should only include numerical values. 
The function returns two matrices. The first is the correlation matrix between the columns of interest. The correlation matrix refers to the Pearson correlation matrix where r = 1 represents a perfect positive linear relationship, r = -1 represents a perfect negative linear relationship and r = 0 represents no linear relationships. 
The second value it returns is the p value matrix. It tells you the p-value of each relationship and whether or not that correlation is significant.

get_columns:
e.g. get_columns()
returns a list of the available columns to examine the correlations between. The first 5 columns contain string data and the rest contain numerical data

In each session, you will be called until you have an answer, in which case, output it as the Answer.
Now begin.
""".strip()

def pearson_correlation(columns):
    df = pd.read_csv(filename)

    df_selected = df[columns]
    correlation_matrix = df_selected.corr(method = 'pearson')

    return correlation_matrix

def get_columns():
    df = pd.read_csv(filename)
    return df.columns

agent = Agent(client = client, system = system_prompt)

def loop(max_iterations = 20, query: str = ""):
    agent = Agent(client=client, system=system_prompt)

    tools = ["pearson_correlation", "get_columns"]

    next_prompt = query

    i = 0

    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            chosen_tool = action[0][0]
            arg = action[0][1]

            if chosen_tool in tools:
                result_tool = eval(f"{chosen_tool}('{arg}')")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue

        if "Answer" in result:
            break

if __name__ == "__main__":
    loop(query = "is there a significant relationship between Vitamin A IU/g fed to a chicken and the chicken's thigh PH?")