import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_decision_boundary(X, y, X_test, y_test, clf):

    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.5)
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].set_title('Decision Boundary')

    axs[1].contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.5)
    axs[1].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y,
                   cmap=plt.cm.RdBu_r, edgecolor='black')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].set_title('Training Data')

    axs[2].contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.5)
    axs[2].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                   c=y_test, cmap=plt.cm.RdBu_r, edgecolor='black')
    axs[2].set_xlabel('Feature 1')
    axs[2].set_ylabel('Feature 2')
    axs[2].set_title('Test Data')

    plt.show()