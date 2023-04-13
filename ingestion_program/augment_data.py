#---------------------------
# Imports
#---------------------------
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy


#---------------------------
# Get augmented data
#---------------------------
def get_augmented_data(train_set, test_set):

        
        random_state = 42
        train_mean = np.mean(train_set["data"]).values
        test_mean = np.mean(test_set["data"]).values

        size = 1000

        # Esitmate z0
        translation = test_mean- train_mean

   

        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
                # randomly choose an alpha

                alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

                # transform z0 by alpha
                translation_ = translation * alphas

                np.random.RandomState(random_state)
                train_df = deepcopy(train_set["data"])
                train_df["labels"] = train_set["labels"]

                df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
                data_sampled = df_sampled.drop("labels", axis=1)
                labels_sampled = df_sampled["labels"].values
                # data_sampled = train_set["data"].sample(n=size, random_state=random_state, replace=True)
                # labels_sampled = np.random.choice(train_set["labels"], size)

                train_data_augmented.append(data_sampled + translation_)
                train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)

        return {
                "data" : augmented_data,
                "labels" : augmented_labels
        }


def get_augmented_data_scaling(train_set, test_set):

        
        random_state = 42
        train_mean = np.mean(train_set["data"]).values
        test_mean = np.mean(test_set["data"]).values


        train_std = np.std(train_set["data"]).values
        test_std = np.std(test_set["data"]).values

        size = 1000


        translation = test_mean - train_mean
        scaling = test_std/train_std


        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):


                # for translation
                alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

                # for scaling
                betas = np.repeat(np.random.uniform(1.0, 1.5, size=size).reshape(-1,1), 2, axis=1 )


                # translation
                translation_ = translation * alphas
                # sclaing
                scaling_ = scaling * betas

                np.random.RandomState(random_state)
                train_df = deepcopy(train_set["data"])
                train_df["labels"] = train_set["labels"]

                df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
                data_sampled = df_sampled.drop("labels", axis=1)
                labels_sampled = df_sampled["labels"].values
                # data_sampled = train_set["data"].sample(n=size, random_state=random_state, replace=True)
                # labels_sampled = np.random.choice(train_set["labels"], size)


                transformed_train_data = (data_sampled + translation_)*scaling_

                train_data_augmented.append(transformed_train_data)
                train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)

        return {
                "data" : augmented_data,
                "labels" : augmented_labels
        }
    
def generate_augmented_dataset_box_estimate(train_sets):
    
    Xdf = [train_set["data"] for train_set in train_sets]
    Yar = [train_set["labels"] for train_set in train_sets]
    sets_augmented = []
    for i in range(len(Xdf)):
        X = np.array(Xdf[i])
        Y = np.array(Yar[i])
        center = np.mean(X, axis=0)
    
        max_x, min_x = np.max(X[:, 0]), np.min(X[:, 0])
        max_y, min_y = np.max(X[:, 1]), np.min(X[:, 1])

        num_points = 1000

        max_distance = np.max(np.linalg.norm(X - center, axis=1))
        min_distance = np.min(np.linalg.norm(X - center, axis=1))

        generated_points = []
        while len(generated_points) < num_points:
            #distance = np.random.normal(center, max_distance,2)

            #angle = np.random.uniform(0, 2 * np.pi)
            std_x = np.std(X[:, 0])
            std_y = np.std(X[:, 1])
            #x = center[0] + distance * np.cos(angle)
            #y = center[1] + distance * np.sin(angle)
            x, y = np.random.normal(center, [std_x, std_y], 2)
            #d = np.absolute(np.linalg.norm([x, y] - center))
            #alpha = 100
            #proba = 1 / ((d + 1))
            #decider = np.random.choice([0, 1], p=[1-proba, proba])
            #p_not_trigger = np.exp(-alpha * d)
            if (x > max_x or x < min_x or y > max_y or y < min_y): #and np.random.rand() > p_not_trigger:
                generated_points.append([x, y])

        generated_points = np.array(generated_points)

        augmented_data = np.vstack((X, generated_points))
        augmented_labels = np.hstack((np.array(Y), np.zeros(len(generated_points))))
        augmented_data2 = pd.DataFrame(augmented_data, columns=['x1','x2'])
        sets_augmented.append({
        "data" : augmented_data2, 
        "labels" : augmented_labels
    })
    
    return sets_augmented