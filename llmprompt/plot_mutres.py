import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

tasks = ['causal_judgment','epistemic_reasoning','hyperbaton','implicatures','logical_fallacy_detection','navigate','snarks','winowhy']

base = []
allmut = []
context = []
req = []
sbs = []

for i,t in enumerate(tasks):

    with open('evo-results/{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()
        base.append(float(values[10].split()[1]))

    with open('evo-results/mutations-{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()
        allmut.append(float(values[9].split()[1]))

    with open('evo-results/mutCONTEXT-{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()
        context.append(float(values[9].split()[1]))

    with open('evo-results/mutREQ-{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()
        req.append(float(values[9].split()[1]))

    with open('evo-results/mutSBS-{}22/run_1/progress_report.csv'.format(t),'r') as f:
        values = f.readlines()
        sbs.append(float(values[9].split()[1]))

fitness = {
        'base':(base,'darkslategrey'),
        'combined':(allmut,'plum'),
        'context':(context,'darkkhaki'),
        'request':(req,'salmon'),
        'step-by-step':(sbs,'lightgreen')
}

x = np.arange(len(tasks))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(11,5))

for attribute, measurement in fitness.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement[0], width, label=attribute, color=measurement[1])
    ax.bar_label(rects, padding=5, fontsize=5.5)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('fitness',fontsize=10)
ax.set_xlabel('task',fontsize=10)
ax.set_title('Best individual after constrained evolution strategy',fontsize=11)
ax.set_xticks(x + 2*width, tasks, fontsize=8)
ax.set_yticks([0.2,0.4,0.6,0.8],['0.2','0.4','0.6','0.8'],fontsize=8)
ax.legend(fontsize=8)
#ax.set_ylim(0, 250)

fig.tight_layout()
fig.savefig('plots/mutres.eps', format='eps')
fig.savefig('plots/mutres.png', format='png')


plt.show()
