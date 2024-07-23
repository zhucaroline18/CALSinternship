import json 
def create_directory(directory_name):
    # Function to create a direcotry
    subprocess.run(["mkdir", directory_name])
    return json.dumps({"directory_name": directory_name})

tool_create_directory = {
    "type" : "function", 
    "function": {
        "name": "create_directory",
        "description": "Create a directory given a directory name.",
        "parameters": {
            "type": "object", 
            "properties": {
                "directory_name": {
                    "type": "string"
                    "description": "The name of the directory to create",
                }
            }
            "required" : ["directory_name"]
        }
    }
}
tools = [tool_create_directory]

def run_terminal_task():
    messages = [("role":"user", "context":
    "Create a folder called 'lucas-the-agent-master'.")]

    tools = [tool_create_directory]
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo-16k"
        messages = messages,
        tools = tools,
        tool choice = "auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls:
        available_fuctions = {
            "create_directory": create_directory,
        }

        messages.append()
