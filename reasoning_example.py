import reasoning_gym
from ollama import chat
from ollama import ChatResponse
from rich import print
from utils import *
from prompts import *
SEED = 100

env_name = "propositional_logic"

dataset = reasoning_gym.create_dataset(env_name,seed=42, size=20)


for example in dataset:
    question = example["question"]
    answer = example['metadata']['example_answer']

    # print(f"[bold white]System: {system_prompt}[/bold white]")
    # print(f"[bold blue]Question: [/bold blue]\n" + question)

    # if answer is not None:
    #     print("[bold green]Answer: [/bold green]\n" + answer)
    
    response: ChatResponse = chat(model='deepseek-r1:1.5b', messages=[
    {
        'role': 'system',
        'content': system_prompt,
    },
    {
        'role': 'user',
        'content': question,
    },
    ])
    llm_response = response['message']['content']

    pred_answer = extract_answer(llm_response)

    score_function = reasoning_gym.get_score_answer_fn(example["metadata"]["source_dataset"])

    print(f"Extracted Answer: {pred_answer}")
    print(f"True Answer: {answer}")
    reward = score_function(pred_answer, example)
    if reward > 0:
        print(f"[bold green]Correct! Reward: {reward}[/bold green]")
    else:
        print(f"[bold red]Incorrect! Reward: {reward}[/bold red]")