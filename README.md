# Ground-Truth-Function-Optimization-Machine-Learning-Project

## Structure

[1. Project Description](#project-description) 

[2. Installation](#installation)

[3. Quick guide to git on Linux](#quick-guide-to-git-on-Linux)

[4. Folders](#folders)

[5. Papers](#papers)
[](#)

## Project Description

#### Motivation:
When developing algorithms, it's essential to evaluate their performance. For classical machine learning, this involves using test datasets to assess model performance. However, not all algorithms work with static datasets; some work with functions or distributions. For instance, optimization algorithms require test functions to evaluate their performance. These test functions should closely resemble the real-world ground truth function that represents the problem to be optimized. Creating appropriate test functions that meet various requirements (e.g., difficulty, cost, diversity, flexibility, and transparency) can be challenging.

#### Problem:
The goal of this project is to generate test functions for optimization algorithms using artificial intelligence and machine learning models. The focus is on choosing suitable approaches/models and practical implementation to assess their performance. The synthetic ground truth function for evaluation will be the BBOB function suite.

## Installation

To clone the repository, change to the directory you want the git to be initialized and use following command:

```
git clone https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project.git
```

To install python and libarys from the `environment.yml` file, first install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html). After that and cloning the repository, activate anaconda and use following command to create a new environment from the `environment.yml` file:

```
conda env create -n MLP_Env -f environment.yml
```

After that just activate the environment with:

```
conda activate MLP_Env
```

And you can start executing code from this repository.

If you download new dependencies in the MLP_Env please use following command to update the environment.yml and push the changes:

```
conda env export > environment.yml
```

## Quick guide to git on Linux

If you are not familiar, here are the most common commands you need for using git in the command line on Linux.

To install git use:

```
sudo apt install git
```

To pull the newest version of a repository use with your credentials:
```
git pull
```

To commit changes use following commmands:
```
git add *
git commit -m "<Your commit message>"
git push
```


## Folders

|Folder|Description|
|---|---|
|[bin](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/bin)|Contains scripts and main files|
|[config](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/config)|Contains config files|
|[data](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/data)|Contains raw and processed data|
|[notebooks](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/notebooks)|Contains jupyter notebooks for EDA or Analysis|
|[src](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/src)|Contains code and functions which are used in bin (main files)|
|[tests](https://github.com/JanLucasK/Ground-Truth-Function-Optimization-Machine-Learning-Project/tree/main/tests)|Contains tests to ensure working code with small examples|


## Papers

In this project mutliple ideas anc arcitectures from papers were used

|Paper|Example Code|
|---|---|
|[PriorVAE](https://arxiv.org/abs/2110.10422)|[Github Repository](https://github.com/elizavetasemenova/priorcvae)|
|[πVAE](https://arxiv.org/abs/2002.06873)|[Github Repository](https://github.com/s-mishra/pivae)|
|||
