# Evolutionary Prompt Engineering

This repository contains code and data for replicating the experiments presented in the paper:

M. Saletta and C. Ferretti. "*Exploring the Prompt Space of Large Language Models through Evolutionary Sampling*". In: GECCO '24: Genetic and Evolutionary Computation Conference, pp. 1345-1353. 2024.

The code is built on top of [DSGE](https://github.com/nunolourenco/sge3). To use it, first clone the DSGE repository, and then copy the [llmprompt](./llmprompt) folder into `/PATH/TO/sge3/sge/`.
Then, replace the following files:
* `/PATH/TO/sge3/sge/engine.py` with [engine.py](./engine.py)
* `/PATH/TO/sge3/sge/operators/mutation.py` with [mutation.py](./mutation.py)
* `/PATH/TO/sge3/sge/parameters.py` with [./parameters.py](./mutation.py)

To replicate the experiments, just move to the `/PATH/TO/sge3/sge/` directory and run the following command: 
```
python -m llmprompt.prompteng --task <TASK> --experiment_name <./PATH/TO/RESULTS> --seed <INTEGER> --parameters llmprompt/sge-params.yml
```

Currently, the implemented tasks are those referred in the paper:
* causal_judgment
* implicatures
* epistemic_reasoning
* hyperbaton
* logical_fallacy_detection
* navigate
* snarks
* winowhy

For further references, see the DSGE documentation.

# Citation

If you find this repository useful for your work, please include the following citation:

```
@inproceedings{evoPrompt,
  author       = {Martina Saletta and
                  Claudio Ferretti},
  title        = {Exploring the Prompt Space of Large Language Models through Evolutionary
                  Sampling},
  booktitle    = {Proceedings of the Genetic and Evolutionary Computation Conference,
                  {GECCO} 2024, Melbourne, VIC, Australia, July 14-18, 2024},
  publisher    = {{ACM}},
  year         = {2024}
}
```
