import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


num_observations = 500

x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

features = np.vstack((x1, x2)).astype(np.float32)
labels = np.hstack((np.zeros(num_observations),
                    np.ones(num_observations)))

plt.scatter(features[:, 0], features[:, 1],
            c=labels)
plt.show()


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

        ll.append(log_likelihood(feature, target, weights))
        epoch.append(step)
    plt.plot(epoch, ll)
    return weights


weights = logistic_regression(features, labels, num_steps=10000, learning_rate=1.4e-2, add_intercept=True)

clf = LogisticRegression(fit_intercept=True, C=1e15)
clf.fit(features, labels)
print("Weights from sk-learn:")
print(clf.intercept_, clf.coef_)
print("Weights from scratch code:")
print(weights)
print(features.shape[0])
data = np.hstack((np.ones((features.shape[0], 1)),
                  features))

print(data.shape)
final_scores = np.dot(data, weights)
preds = np.round(sigmoid(final_scores))

print('Accuracy from scratch: {0}'.format((preds == labels).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(features, labels)))

plt.figure(figsize=(12, 8))
plt.scatter(features[:, 0], features[:, 1], c=preds == labels - 1, alpha=.8, s=50)
plt.show()
