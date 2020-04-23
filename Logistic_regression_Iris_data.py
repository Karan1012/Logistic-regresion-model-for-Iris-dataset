from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris.data[:, :4]
labels = (iris.target != 0) * 1


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(feature, target, weights):
    scores = np.dot(feature, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(feature, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((feature.shape[0], 1))
        feature = np.hstack((intercept, feature))
    ll = []
    epoch = []
    weights = np.zeros(feature.shape[1])
    # print(feature.shape[0])
    for step in range(num_steps):
        # print(step)
        for i in range(len(target)):
            scores = np.dot(feature[i, :], weights)
            # print(scores)
            predictions = sigmoid(scores)

            # Update weights with gradient
            output_error_signal = target[i] - predictions
            gradient = np.dot(feature[i, :].T, output_error_signal)
            weights += learning_rate * gradient

            # Print log-likelihood every so often
            # if step % 1000 == 0:
            # print(log_likelihood(feature, target, weights))
        ll.append(log_likelihood(feature, target, weights))
        epoch.append(step)
    plt.plot(epoch, ll)
    plt.show()
    return weights


weights = logistic_regression(features, labels, num_steps=10000, learning_rate=1e-4, add_intercept=True)

data = np.hstack((np.ones((features.shape[0], 1)),
                  features))
final_scores = np.dot(data, weights)
preds = np.round(sigmoid(final_scores))

print('Accuracy from scratch: {0}'.format((preds == labels).sum().astype(float) / len(preds)))
