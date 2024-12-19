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
        
        #if self.task == 'causal_judgment' or self.task == 'implicatures' or self.task == 'navigate' or self.task == 'logical_fallacy_detection':
        with open(self.testset,'r') as f:
            tests = f.readlines()

        sampfit = random.choices(tests,k=100)
        #sampfit = tests

        for s in sampfit:
            js = json.loads(s)
            test = individual.replace('[[INPUT]]', 'QUESTION: ' + js['input'])
            
            #prompt_template=f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {test} ASSISTANT:"
            prompt_template=f"GPT4 Correct User: {test}<|end_of_turn|>GPT4 Correct Assistant:"
        
            input_ids = self.tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = self.model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=32)#512)
            
            ans = self.tokenizer.decode(output[0])
            print(ans, file=sys.stderr)
                    
            answer = ans[ans.index("GPT4 Correct Assistant:")+23:-15].lower()
            #answer = ans[ans.index("ASSISTANT:")+10:-4].lower()[:20]
            
            if self.task == 'causal_judgment':
                if js['target_scores']['Yes'] == 1:
                    target = 'yes'
                else:
                    target = 'no'

                if target in answer[:20]:
                    fitness += 1 
            elif self.task == 'navigate':
                if js['target_scores']['True'] == 1:
                    target = 'yes'
                else:
                    target = 'no'

                if target in answer[:20]:
                    fitness += 1
            elif self.task == 'logical_fallacy_detection':
                if js['target_scores']['Valid'] == 1:
                    target = 'yes'
                else:
                    target = 'no'

                if target in answer[:20]:
                    fitness += 1            
            elif self.task == 'implicatures':
                if js['target_scores']['yes'] == 1:
                    target = 'yes'
                else:
                    target = 'no'

                if target in answer[:20]:
                    fitness += 1  
            if self.task == 'epistemic_reasoning':
                if js['target_scores']['entailment'] == 1:
                    target = 'entailment'
                else:
                    target = 'non-entailment'

                if target == 'non-entailment' and target in answer[:30]:
                    fitness += 1         
                elif target == 'entailment' and 'non' not in answer[:30]:
                    fitness += 1

            if self.task == 'winowhy':
                if js['target_scores']['correct'] == 1:
                    target = 'yes'
                else:
                    target = 'no'

                if target in answer[:20]:
                    fitness += 1
            if self.task == 'snarks':
                if js['target_scores']['(a)'] == 1:
                    target = '(a)'
                else:
                    target = '(b)'

                if target in answer[:20]:
                    fitness += 1
            if self.task == 'hyperbaton':
                if js['target_scores']['a'] == 1:
                    target = 'a'
                else:
                    target = 'b'

                if target in answer[:20]:
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
    
    #model_name_or_path = "TheBloke/vicuna-7B-v1.5-GPTQ"
    model_name_or_path = "TheBloke/Starling-LM-7B-alpha-GPTQ"
    
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    ####################
    
    sge.setup()
    params['GRAMMAR'] = 'llmprompt/grammars/{}.bnf'.format(params['TASK'])
    params['EXPERIMENT_NAME'] = 'llmprompt/evo-results-starling/mutations-{}22'.format(params['TASK'])

    eval_func = PromptEngLLM(params['TASK'],model,tokenizer)
    #sge.llm_evolutionary_algorithm(evaluation_function=eval_func)
    
    #####EVOLUTION STRATEGY#####
    
    #load population
    pop = []
    for i in range(11):
        with open('llmprompt/evo-results-starling/{}22/run_1/iteration_{}.json'.format(params['TASK'],i)) as f:
            p = json.loads(f.read())
            for ind in p:
                pop.append(ind)
    pop.sort(key=lambda ind: ind['fitness'], reverse=True)
    
    #remove duplicates
    i = 0
    population = []
    while i < len(pop)-1:
        if pop[i]['phenotype'] == pop[i+1]['phenotype']:
            i += 1
        else:
            population.append(pop[i])
            i += 1
    population.append(pop[i])
    
    """
    #select only individuals with a given nonterminal
    for mg in [7]:
        curpop = population
        for p in curpop:
            if len(p['genotype'][mg]) == 0:
                curpop.remove(p)
    """
    sge.llm_evolution_strategy(evaluation_function=eval_func, population=population[:params['LAMBDA']], mu=params['MU'],mutgenes=[7,8,9,10])
    
