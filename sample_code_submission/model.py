import pickle
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from GDA import GaussianDiscriminativeAnalysisClassifier
from GNB import GaussianNaiveBayesClassifier

from sklearn import preprocessing
from dann import DANN
import torch
import torch.nn as nn
import torch.optim as optim


MODEL_CONSTANT = "Constant"
MODEL_NB = "NB"
MODEL_LDA = "LDA"
MODEL_RR = "RR"
MODEL_TREE = "Tree"
MODEL_MLP = "MLP"
MODEL_RF = "RF"
MODEL_SVM = "SVM"
MODEL_KN = "KN"
MODEL_ADA = "ADA"
MODEL_GDA = "GDA"
MODEL_GNB = "GNB"
MODEL_DANN = "DANN"


PREPROCESS_TRANSLATION = "translation"
PREPROCESS_SCALING = "scaling"

AUGMENTATION_TRANSLATION = "translation"
AUGMENTATION_TRANSLATION_SCALING = "translation-scaling"
AUGMENTATION_BOX = "box"

#------------------------------
# Baseline Model
#------------------------------
class Model:

    def __init__(self, 
                 model_name=MODEL_ADA, 
                 X_train=None, 
                 Y_train=None, 
                 X_test=None,
                 preprocessing=False, 
                 preprocessing_method = PREPROCESS_SCALING,
                 data_augmentation=False,
                 data_augmentation_type=AUGMENTATION_TRANSLATION
        ):

        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.preprocessing = preprocessing
        self.preprocessing_method = preprocessing_method
        self.data_augmentation = data_augmentation
        self.data_augmentation_type = data_augmentation_type
        self.box = None

        self._set_model()

    def _set_model(self):

        if self.model_name == MODEL_CONSTANT:
            self.clf = None 
        if self.model_name == MODEL_NB:
            self.clf = GaussianNB()
        if self.model_name == MODEL_MLP:
            self.clf = MLPClassifier()
        if self.model_name == MODEL_LDA:
            self.clf = LinearDiscriminantAnalysis()
        if self.model_name == MODEL_RR:
            self.clf = RidgeClassifier()
        if self.model_name == MODEL_TREE:
            self.clf = tree.DecisionTreeClassifier()
        if self.model_name == MODEL_RF:
            self.clf = RandomForestClassifier()
        if self.model_name == MODEL_SVM:
            self.clf = SVC(probability = True,kernel='rbf', gamma=1)
        if self.model_name == MODEL_KN:
            self.clf = KNeighborsClassifier(n_neighbors=5)
        if self.model_name == MODEL_ADA:
            self.clf = AdaBoostClassifier(n_estimators=100)
        if self.model_name == MODEL_GDA:
            self.clf = GaussianDiscriminativeAnalysisClassifier()
        if self.model_name == MODEL_GNB:
            self.clf = GaussianNaiveBayesClassifier()
        if self.model_name == MODEL_DANN:
            self.clf = DANN(input_dim=2, hidden_dim=100, output_dim=2, domain_dim=2)

        self.is_trained=False

    def _preprocess_translation(self):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        translation = test_mean- train_mean

        X_test_preprocessed = self.X_test - translation

        return X_test_preprocessed
    
    def _preprocess_scaling(self):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        train_std = np.std(self.X_train).values
        test_std = np.std(self.X_test).values


        translation = test_mean- train_mean
        scaling = test_std/train_std


        X_test_preprocessed = (self.X_test - translation)/scaling

        return X_test_preprocessed

    def _augment_data_translation(self):

        random_state = 42
        size = 1000


        # Mean of Train and Test
        train_mean = np.mean(self.X_train, axis=0).values
        test_mean = np.mean(self.X_test, axis=0).values

        # Esitmate z0
        translation = test_mean - train_mean



        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            # randomly choose an alpha

            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

            # transform z0 by alpha
            translation_ = translation * alphas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.X_train)
            train_df["labels"] = self.Y_train

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values
    

            train_data_augmented.append(data_sampled + translation_)
            train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)


        return augmented_data, augmented_labels
    
    def _augment_data_scaling(self):

        random_state = 42
        size = 1000

        # Mean of Train and Test
        train_mean = np.mean(self.X_train, axis=0).values
        test_mean = np.mean(self.X_test, axis=0).values

        train_std = np.std(self.X_train, axis=0).values
        test_std = np.std(self.X_test, axis=0).values

        # Esitmate z0
        translation = test_mean- train_mean
        scaling = test_std/train_std


        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            
            # uniformly choose alpha between -3 and 3
            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

            # uniformly choose beta between 1 and 1.5
            betas = np.repeat(np.random.uniform(1.0, 1.5, size=size).reshape(-1,1), 2, axis=1 )

            # translation
            translation_ = translation * alphas
            # sclaing
            scaling_ = scaling * betas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.X_train)
            train_df["labels"] = self.Y_train

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values

            transformed_train_data = (data_sampled + translation_)*scaling_
    

            train_data_augmented.append(transformed_train_data)
            train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)


        return augmented_data, augmented_labels
    
    def _augment_data_box(self):

        X = np.array(self.X_train)
        Y = np.array(self.Y_train)
        center = np.mean(X, axis=0)
    
    
        max_x, min_x = np.max(X[:, 0]), np.min(X[:, 0])
        max_y, min_y = np.max(X[:, 1]), np.min(X[:, 1])

        num_points = 1000

        max_distance = np.max(np.linalg.norm(X - center, axis=1))
        min_distance = np.min(np.linalg.norm(X - center, axis=1))

        generated_points = []
        for i in range(num_points):
            std_x = np.std(X[:, 0])
            std_y = np.std(X[:, 1])

            x, y = np.random.normal(center, [std_x, std_y], 2)
            if (x > max_x or x < min_x or y > max_y or y < min_y):
                generated_points.append([x, y])

        generated_points = np.array(generated_points)

        augmented_data = np.vstack((X, generated_points))
        augmented_labels = np.hstack((np.array(Y), np.zeros(len(generated_points))))
        
        X = np.array(self.X_test)
        center = np.mean(X, axis=0)
    
        max_x, min_x = np.max(X[:, 0]), np.min(X[:, 0])
        max_y, min_y = np.max(X[:, 1]), np.min(X[:, 1])

        num_points = 1000

        max_distance = np.max(np.linalg.norm(X - center, axis=1))
        min_distance = np.min(np.linalg.norm(X - center, axis=1))

        generated_points = []
        while len(generated_points) < num_points:
            std_x = np.std(X[:, 0])
            std_y = np.std(X[:, 1])

            x, y = np.random.normal(center, [std_x, std_y], 2)

            if (x > max_x or x < min_x or y > max_y or y < min_y):
                generated_points.append([x, y])

        generated_points = np.array(generated_points)

        augmented_data_test = np.vstack((X, generated_points))


        return augmented_data, augmented_labels, augmented_data_test, len(generated_points)
        
    def fit(self, X=None, y=None,Xt = None):

        if self.model_name != MODEL_CONSTANT:
            
                
            if X is None:
                X = self.X_train
            if y is None:
                y = self.Y_train
            if Xt is None :
                Xt = self.X_test
            
            if self.data_augmentation:
                if self.data_augmentation_type == AUGMENTATION_TRANSLATION:
                    X, y = self._augment_data_translation()
                elif self.data_augmentation_type == AUGMENTATION_BOX:
                    X, y, Xt, l = self._augment_data_box()
                    self.X_test = pd.DataFrame(data = Xt, columns = ["x1","x2"])
                    self.box = l
                else:
                    X, y = self._augment_data_scaling()
            if self.model_name == MODEL_DANN:
                if self.data_augmentation_type == AUGMENTATION_BOX and self.data_augmentation:
                    X_Trains = torch.tensor(X).float()
                    X_Tests = torch.tensor(Xt).float()
                else:
                    X_Trains = torch.tensor(X.values).float()
                    X_Tests = torch.tensor(Xt.values).float()
                Y_Trains = torch.tensor(y, dtype=torch.long)
                optimizer = optim.Adam(self.clf.parameters(), lr=0.001)

                n_epochs = 100
                for epoch in range(n_epochs):
                    self.clf.train()
                    optimizer.zero_grad()
                    loss = self.clf.get_loss(X_Trains, Y_Trains, X_Tests, alpha=0.5)
                    loss.backward()
                    optimizer.step()
            else :
                self.clf.fit(X, y)
            self.is_trained=True

    def predict(self, X=None):
        if X is None:
            X = self.X_test
            if self.preprocessing:
                if self.preprocessing_method == PREPROCESS_TRANSLATION:
                    X = self._preprocess_translation()
                    if self.data_augmentation_type == AUGMENTATION_BOX and self.data_augmentation:
                        X = X[:-self.box]
                else:
                    X = self._preprocess_scaling()
                    if self.data_augmentation_type == AUGMENTATION_BOX and self.data_augmentation:
                        X = X[:-self.box]

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        
        if self.model_name == MODEL_DANN:
            X_Tests = torch.tensor(X.values)
            X_Tests = X_Tests.float()
            self.clf.eval()
            with torch.no_grad():
                label_output, _ = self.clf(X_Tests, alpha=0)
                pred = torch.argmax(label_output, dim=1)
            return pred
        else :
            return self.clf.predict(X)

    def decision_function(self, X=None):
        
        if X is None:
            X = self.X_test
            if self.preprocessing:
                if self.preprocessing_method == PREPROCESS_TRANSLATION:
                    X = self._preprocess_translation()
                    if self.data_augmentation_type == AUGMENTATION_BOX and self.data_augmentation:
                        X = X[:-self.box]
                else:
                    X = self._preprocess_scaling()
                    if self.data_augmentation_type == AUGMENTATION_BOX and self.data_augmentation:
                        X = X[:-self.box]

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])
        
        

        if self.model_name == MODEL_NB or self.model_name == MODEL_TREE or self.model_name == MODEL_SVM or self.model_name == MODEL_MLP or self.model_name == MODEL_RF or self.model_name == MODEL_KN or self.model_name == MODEL_ADA:
            return self.clf.predict_proba(X)[:, 1]
        elif self.model_name == MODEL_GDA or self.model_name == MODEL_GNB:
            predicted_score = self.clf.predict_proba(X)
            # Transform with log
            epsilon = np.finfo(float).eps
            predicted_score = -np.log((1/(predicted_score+epsilon))-1)
            return predicted_score[:, 1]
        elif self.model_name == MODEL_DANN :
            X_Tests = torch.tensor(X.values).float()
            self.clf.eval()
            with torch.no_grad():
                label_output, _ = self.clf(X_Tests, alpha=0)
                proba = torch.softmax(label_output, dim=1)[:, 1]
            return proba
        else:
            return self.clf.decision_function(X) 
        
    def save(self, name):
        pickle.dump(self.clf, open(name + '.pickle', "wb"))

    def load(self, name):
        modelfile = name + '.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
