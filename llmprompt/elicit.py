from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#model_name_or_path = "TheBloke/Starling-LM-7B-alpha-GPTQ"
model_name_or_path = "TheBloke/vicuna-7B-v1.5-GPTQ"

# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "do you know Pascal language? Please, answer with this format: \nANSWER: [ANSWER]"
#prompt_template=f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
prompt_template=f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"

print("\n\n*** Generate:")

input_ids = tokenizer([prompt_template, prompt_template], return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
for i,o in enumerate(output):
    print(i, tokenizer.decode(o))
