# sensor-data-gans

This project was part of the course "Applied Machine Lerning for Digital Health" 
at Hasso Plattner Instute in the summer term 2020.

The aim of this project is to teach generative adversarial networks (GANs) 
to generate synthetic sensor data, which later are used to improve human activity recognition (HAR) classifiers.

This project utilizes the [MotionSense Dataset](https://github.com/mmalekzadeh/motion-sense) to train GANs able to generate sensor data for the activities:
* standing
* walking
* jogging

Because the training of GANs are very hard and involves many trials, architectural design and parameter changes, 
this project was designed to adapt to this challenges.

Centerpieces of this project are the components:
* [GanTrainer](./gans/gan_trainer.py): Handles training and validation datasets, trains and evaluates GANs, saves the best GANs
* [GanEvaluator](./gans/gan_evaluator.py): Handles training and test datasets, loads saved GANs and evaluates them

## How To:

make sure you are in sensor-data-gans  

start the python environment with all dependencies:
```
pipenv shell
```

start notebook or jupyater lab:
```
jupyter notebook
```
```
jupyter notebook
```

### Training:

For training please check notebook files:
* [_example_training.ipynb](./gans/notebooks/_example_training.ipynb)
* [CNN_GAN_training_routine.ipynb](./gans/notebooks/CNN_GAN_training_routine.ipynb)
* [LSTM_GAN_training_routine.ipynb](./gans/notebooks/LSTM_GAN_training_routine.ipynb)

### Evaluation:
For evaluation please check notebook files:
* [_example_evaluation.ipynb](./gans/notebooks/_example_evaluation.ipynb)
* [eval_action_0_standing.ipynb](./gans/notebooks/eval_action_0_standing.ipynb)
* [eval_action_1_walking.ipynb](./gans/notebooks/eval_action_1_walking.ipynb)
* [eval_action_2_standing.ipynb](./gans/notebooks/eval_action_2_jogging.ipynb)







