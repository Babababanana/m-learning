## Training, Validation, Test Set

To measure the performance of a model, we usually split our data into 3 different sets, training set, validation set and test set (e.g. 60%, 20%, 20%).

- training set - to train our model
- validation set - to select model
- test set - to estimate generalization error

## Overfitting

Reduce overfitting
1. Add more training data
2. Data augmentation
3. Reduce model complexity
4. Randomly dropout some subset of nodes in a given layer during training
5. Regularization

## Regularization

### Weight Decay (L2 Regularization)
The idea of L2 regularization is to add an extra term to the cost function called the regularization term.
