# Fair Universe Machine Learning Challenge

### Link to the competition website: https://www.codabench.org/competitions/674/


## Introduction
This challenge lies in the fact that the distribution of the signal and background clusters can vary significantly between training and test data, with shifts in the center and variance of Gaussian distributions, which makes classification difficult. This problem is known as the domain adaptation problem in machine learning and has been studied extensively in recent years. The objective of this ML challenge is to develop a binary classifier that can accurately classify signal versus background particles in the presence of distribution shift, with the goal of enhancing the accuracy of the detection of the creation of the particle of interest.

In this repo we try to solve this challenge using different methods. This repo was forked from [[here](https://github.com/ihsaan-ullah/fair-universe)].

This repo was used to experminet and try different methodes to solve the challenge. 


## Data
 
In the Data_Generator directory, it is explained how the data is generated for the this challenge.


## Structure

In the following we explain breifly what each file does:

- `sample_code_submission/GNB.py`: This file contains the implementation of Gaussian Naive Bayes (GNB) method.
- `sample_code_submission/GDA.py`: This file contains the implementation of Gaussian Discriminiant Analysis (GDA) method.
- `my_visualize.py`: This file is used to visualize the decison boundary of the GDA and GNB methods
- `experiment.ipynb`: This file is used to experminet and see the result of implementation of two hand coded methods including GDA, GNB.
- `sample_code_submission/dann.py`: This file contains the implementation of Domain Adversarial Neural Network (DANN) method which is a deep learning method for domain adaptation according to the following paper by [[this paper](https://arxiv.org/abs/1505.07818)].
- `sample_code_submission/model.py`: This file contains contains all the baselines used in this challenge.

## Setup
1 - **Clone the repo**:
 ```
git clone https://github.com/mhdirnjbr/fair-universe-ml-challenge.git
 ```

2 - **Install the requirements**: 
 ```
 pip install numpy pandas matplotlib seaborn scikit-learn
 ```
3 - **Run the code**:
 
Eache notebook corresponds to a specific case of data:
- Baselines_Translation.ipynb : Data is translated
- Baselines_Scaling.ipynb : Data is scaled and translated
- Baselines_Stretch.ipynb : Data is stretched and translated
- Baselines_Box.ipynb : Data is translated and using a box to remove outliers
