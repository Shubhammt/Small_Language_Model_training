from reasoning_gym import get_score_answer_fn, create_dataset
from prompts import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich import print

SEED = 100
batch_size = 2
n_rollouts = 3
buffer_size = 6
max_seq_length = 512
max_new_tokens = 100
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

env_name = "propositional_logic"
dataset = create_dataset(env_name,seed=42, size=1)


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

for entry in dataset:
    question = entry["question"]
    answer = entry['metadata']['example_answer']

    validation_object = entry["metadata"]["source_dataset"]
    score_fn = get_score_answer_fn(validation_object)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}, # Obtained from reasoning-gym
    ]
    templated_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        [templated_string],
        return_tensors="pt",
        padding_side="left",
        # max_length=512,
        # padding='max_length',
        # truncation=True,
    )

    generated_response = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens, # The max number of tokens to generate
        do_sample=True,                # Probabilistic sampling
        top_p=0.95,                    # Nucleus sampling
        num_return_sequences=n_rollouts,        # Number of sequences per question
        temperature=1,                 # Increase randomness
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.batch_decode(generated_response, skip_special_tokens = True)[0])