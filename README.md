# REINFORCE-js
[![js-google-style](https://img.shields.io/badge/code%20style-google-blue.svg)](https://google.github.io/styleguide/jsguide.html)

[dqn-solver]: examples/dqn-solver.md
[dqn-solver-src]: examples/dqn/dqn-solver.md
[td-solver-src]: examples/td/td-solver.md

**REINFORCE-js** is an object-oriented Javascript (baked with [Typescript](https://github.com/Microsoft/TypeScript)) port of _Andrej Karpathy's_ Reinforcement Learning library that implements several common RL algorithms.
In particular, the library currently includes:

* **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks ([DQNSolver Details][dqn-solver])
* **Dynamic Programming** methods
* (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
* **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures for dealing with continuous action spaces. (_very alpha, likely buggy or at the very least finicky and inconsistent_)

For further Information see the [reinforce-js](https://github.com/karpathy/reinforcejs) repository.

## Use as Project Dependency

### How to install as a dependency:

Download available `@npm`: [reinforce-js](https://www.npmjs.com/package/reinforce-js)

Install via command line:

```
npm install --save reinforce-js
```

The project directly ships with the transpiled Javascript code, Map-files and Declaration-files.

### How to use the Library in Production:

Currently exposed Classes (more to be expected soon):

* **Solver** - Generic Solver Interface
* **Env** - Generic *Environment* for a Solver
* **Opt** - Generic *Options* for a Solver

DQN-Solver: ([Code-Example][dqn-solver-src] and [General Information][dqn-solver])
* **DQNSolver** - Concrete *Deep Q-Learning* Solver
* **DQNOpt** - Concrete *Options* for DQNSolver creation
* **DQNEnv** - Concrete *Environment* for DQNSolver creation

TD-Solver: ([Code-Example][td-solver-src])
* **TDSolver** - Concrete *Temporal Difference* Solver
* **TDOpt** - Concrete *Options* for TDSolver creation
* **TDEnv** - Concrete *Environment* for TDSolver creation

Planned to be implemented:

* **DPSolver** - Concrete *Dynamic Programming* Solver
* **DPOpt** - Concrete *Options* for DPSolver creation
* **SimpleReinforcementSolver** - Concrete *Simple Reinforcement* Solver
* **SimpleReinforcementOpt** - Concrete *Options* for SimpleReinforcementSolver creation
* **RecurrentReinforcementSolver** - Concrete *Recurrent Reinforcement* Solver
* **RecurrentReinforcementOpt** - Concrete *Options* for RecurrentReinforcementSolver creation
* **DeterministPGSolver** - Concrete *Deterministic Policy Gradient* Solver
* **DeterministPGOpt** - Concrete *Options* for DeterministPGSolver creation

These classes can be imported from this `npm` module, e.g.:
```typescript
import { DQNSolver, DQNOpt, DQNEnv } from 'reinforce-js';
```

For JavaScript usage `require` classes from this `npm` module as follows:
```javascript
const DQNSolver = require('reinforce-js').DQNSolver;
const DQNOpt = require('reinforce-js').DQNOpt;
const DQNEnv = require('reinforce-js').DQNEnv;
```

## Further Info for Production Usage

The transpiled Javascript-target is `ES6`, with a `CommonJS` module format.

## For Contributors

1. `Clone` this project to a working directory.
2. `npm install` to setup the development dependencies.
3. To compile the codebase:

```
tsc -p .
```

This project relies on Visual Studio Codes built-in Typescript linting facilities. Let's follow primarily the [Google TypeScript Style-Guide](https://github.com/google/ts-style) through the included *tslint-google.json* configuration file.

### Dependencies

This Library relies on the object-oriented _Deep Recurrent Neural Network_ library:

* **GitHub**: [recurrent-js](https://github.com/mvrahden/recurrent-js)
* **npm**: [recurrent-js](https://www.npmjs.com/package/recurrent-js)

### Work in Progress
Please be aware that this repository is still _under construction_. Changes are likely to happen.
There are still classes to be added, e.g. *DPSolver*, *SimpleReinforcementSolver*, *RecurrentReinforcementSolver*, *DeterministPG* and their individual *Opts* and *Envs*

## License

As of License-File: [MIT](LICENSE)
