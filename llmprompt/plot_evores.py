import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

tasks = ['causal_judgment','epistemic_reasoning','hyperbaton','implicatures','logical_fallacy_detection','navigate','snarks','winowhy']

x = [i for i in range(11)]
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(11,4))

for i,t in enumerate(tasks):

    with open('evo-results/{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()

    best = []
    mean = []
    var = []

    for gen in values[:11]:
        best.append(float(gen.split()[1]))
        mean.append(float(gen.split()[2]))
        var.append(float(gen.split()[3]))

    if i < 4:
        r = 0
        c = i
    else:
        r = 1
        c = i-4

    axs[r,c].plot(x,best,color='steelblue')
    axs[r,c].errorbar(x,mean,yerr=var,color='steelblue',linestyle='dashed')
    
    #axs[r,c].set_yscale('logit')
    #axs[r,c].set_ylim([0.3,1])
    
    axs[r,c].set_title(t, fontsize='small', loc='left')
    axs[r,c].grid()

plt.setp(axs[1, 0], xlabel='generations')
plt.setp(axs[1, 1], xlabel='generations')
plt.setp(axs[1, 2], xlabel='generations')
plt.setp(axs[1, 3], xlabel='generations')
plt.setp(axs[0, 0], ylabel='fitness')
plt.setp(axs[1, 0], ylabel='fitness')

custom_lines = [Line2D([0], [0], color='steelblue'),
                Line2D([0], [0], color='steelblue', linestyle='dashed')]

fig.legend(custom_lines, ['best fitness', 'mean fitness'])
fig.legend()

fig.suptitle('Results after evolutionary search',fontsize=11)

fig.tight_layout()
fig.savefig('plots/evores.eps', format='eps')
fig.savefig('plots/evores.png', format='png')

plt.show()
