import json

tasks = ['epistemic_reasoning','implicatures','logical_fallacy_detection','navigate','snarks','winowhy','hyperbaton','causal_judgment']

#starling
#minfit = [0.66,0.65,0.69,0.61,0.66,0.74,0.68,0.59] 

#vicuna
minfit = [0.56,0.66,0.64,0.6,0.63,0.59,0.82,0.62]


for j,task in enumerate(tasks):
    allpop = []
    for m in ['mutations','mutREQ','mutCONTEXT','mutSBS']:
        pop = []
        for i in range(10):
            with open('evo-results/{}-{}22/run_1/iteration_{}.json'.format(m,task,i)) as f:
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

    tot = len(allpop)
    sbs = 0
    role = 0
    ex = 0 
    for p in allpop:
        
        #context
        if len(p['genotype'][7]) > 0:
            role += 1
        
        #examples
        if len(p['genotype'][9]) > 0:
            ex += 1

        #sbs
        if len(p['genotype'][10]) > 0:
            sbs += 1

    print(task, 'role: {:.3f} - examples: {:.3f} - sbs: {:.3f}'.format(role/tot,ex/tot,sbs/tot))               
   
