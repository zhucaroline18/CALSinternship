import os 
os.environ['GROQ_API_KEY'] = ""
#from groq import Groq
import ollama
import re




filename = 'totalDataLLM.csv'

class Agent:
    def __init__(self, client = "llama3.1", system:str = "") -> None:
        self.client = client
        self.system = system
        self.messages:list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})


    def __call__(self, message = ""):
        if message: 
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = ollama.chat(
            self.client, messages=self.messages
        )
        return completion['message']['content']

system_prompt = """
Answer the following questions and obey the following commands as best you can. 

You have access to the following tools:

pearson_correlation:
e.g. pearson_correlation(['Vitamin A IU/kg', 'thigh PH', 'beta-carotene'])
takes in a list of column names regarding the columns of interest. This can refer to a nutrient intake inputs for chickens or the chicken health parameter outputs. The columns of interest should only include numerical values. 
The function returns two matrices. The first is the correlation matrix between the columns of interest. The correlation matrix refers to the Pearson correlation matrix where r = 1 represents a perfect positive linear relationship, r = -1 represents a perfect negative linear relationship and r = 0 represents no linear relationships. 
The second value it returns is the p value matrix. It tells you the p-value of each relationship and whether or not that correlation is significant.

get_columns:
e.g. get_columns()
returns a list of the available columns to examine the correlations between. The first 5 columns contain string data and the rest contain numerical data

You will receive a message from the human, then you should start a loop and do one of three things:

Option 1: You use a tool to answer the question. 
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [pearson_correlation, get_columns]
Action Input: "the input to the action to be sent to the tool"

After this, the human will respond with an observation and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Option 3: You give the human a final answer.
For this, you should use the following format:
Action: Final Answer
Action Input: "your answer to the human's original question, summarizing your response to their query"

Begin! 
""".strip()

def pearson_correlation(columns):
    df = pd.read_csv(filename)

    df_selected = df[columns]
    correlation_matrix = df_selected.corr(method = 'pearson')

    return correlation_matrix

def get_columns():
    df = pd.read_csv(filename)
    return df.columns


def loop(max_iterations = 20, query: str = ""):
    agent = Agent("llama3.1", system=system_prompt)

    tools = ["pearson_correlation", "get_columns"]

    next_prompt = query

    i = 0

    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            pattern = r"Action:\s*(\w+)\((\[[^\]]+\])\)"

            # Search for the pattern in the input string
            match = re.search(pattern, input_string)

            if match:
                method_name = match.group(1)
                data_list = match.group(2)
            else:
                print("No match found")

            if chosen_tool in tools:
                result_tool = eval(f"{method_name}({data_list})")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue

        if "Final Answer" in result:
            break
    #print(agent.messages)

if __name__ == "__main__":
    #loop(query = "is there a significant relationship between Vitamin A fed to a chicken and the chicken's thigh PH?")
    print("Enter prompt:")
    prompt = input()
    loop(query = prompt)