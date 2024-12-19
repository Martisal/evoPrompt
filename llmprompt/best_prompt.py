import json

tasks = ['epistemic_reasoning','implicatures','logical_fallacy_detection','navigate','snarks','winowhy','hyperbaton','causal_judgment']
minfit = [0.66,0.65,0.69,0.61,0.66,0.74,0.68,0.59] #starling
#minfit = [0.56,0.66,0.64,0.6,0.63,0.59,0.82] #vicuna

for j,task in enumerate(tasks):
    allpop = []
    for m in ['mutations','mutREQ','mutCONTEXT','mutSBS']:
        pop = []
        for i in range(10):
            with open('evo-results-starling/{}-{}22/run_1/iteration_{}.json'.format(m,task,i)) as f:
                p = json.loads(f.read())
            for ind in p:
                if ind['fitness'] >= minfit[j]:                
                    pop.append(ind)
            pop.sort(key=lambda ind: ind['fitness'], reverse=True)
        
        #remove duplicates
        i = 0
        
        while i < len(pop)-1:
            if pop[i]['phenotype'] == pop[i+1]['phenotype']:
                i += 1
            else:
                allpop.append(pop[i])
                i += 1
        if i < len(pop):
            allpop.append(pop[i])

    prompts = []
    card = []
    fit = []

    for p in allpop:
        if p['phenotype'] not in prompts:
            prompts.append(p['phenotype'])
            card.append(1)
            fit.append(p['fitness'])
        else:
            card[prompts.index(p['phenotype'])] += 1
            if fit[prompts.index(p['phenotype'])] > p['fitness']:
                fit[prompts.index(p['phenotype'])] = p['fitness']

    m = max(card)
    f = 0
    for i in range(len(card)):
        if card[i] == m:
            if fit[i] > f:
                prompt = prompts[i]

    print('TASK:',task)
    print()
    print(card)
    print(fit)
    #print(prompts)
    print(prompt)
    print('-------------')
    
