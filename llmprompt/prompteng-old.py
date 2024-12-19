import random
import json
import sys

class PromptEngLLM:
    def __init__(self,task,model,tokenizer):
        self.task = task
        self.testset = 'llmprompt/testsets/{}.json'.format(task)
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, individual):
        print('INDIVIDUAL', individual, file=sys.stderr)
        fitness = 0
        if self.task == 'causal_judgment' or self.task == 'implicatures' or self.task == 'navigate' or self.task == 'logical_fallacy_detection':
            with open(self.testset,'r') as f:
                tests = f.readlines()

            #sampfit = random.choices(tests,k=50)
            sampfit = tests

            #templates = [] 
            outstr = []
            for s in sampfit:
                js = json.loads(s)
                #print('INPUT:',js['input'])
                test = individual.replace('[[INPUT]]', 'QUESTION: ' + js['input'])
                
                prompt_template=f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {test} ASSISTANT:"
                #prompt_template=f"GPT4 Correct User: {test}<|end_of_turn|>GPT4 Correct Assistant:"
                
                #templates.append(prompt_template)
            
                input_ids = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
                output = self.model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=8)#512)
                
                ans = self.tokenizer.decode(output[0])
                outstr.append(ans)
                print(ans, file=sys.stderr)
                
                 
            for s in outstr:    
                #answer = s[s.index("GPT4 Correct Assistant:")+23:-15].lower()
                answer = s[s.index("ASSISTANT:")+10:-4].lower()[:4]

                if js['target_scores']['True'] == 1:
                    target = 'yes'
                else:
                    target = 'no'
                    
                #print('ANSWER',answer)
                #print('TARGET',target)

                if target in answer:
                    fitness += 1 

            #print("{:.2f}".format(fitness/len(sampfit)))
            #print("{:.2f}".format(fitness/len(sampfit)), "INDIVIDUO:", individual)   
            #print(self.tokenizer.decode(output[0]))
            
            return fitness/len(sampfit), {}

if __name__ == "__main__":
    import sge
    from sge.parameters import params
     
    #####LOAD MODEL#####
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    model_name_or_path = "TheBloke/vicuna-7B-v1.5-GPTQ"
    #model_name_or_path = "TheBloke/Starling-LM-7B-alpha-GPTQ"
    
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    ####################
    
    sge.setup()
    #params['GRAMMAR'] = 'llmprompt/grammars/{}.bnf'.format(params['TASK'])
    params['EXPERIMENT_NAME'] = 'llmprompt/evo-results/{}2'.format(params['TASK'])
    eval_func = PromptEngLLM(params['TASK'],model,tokenizer)
    sge.llm_evolutionary_algorithm(evaluation_function=eval_func)
