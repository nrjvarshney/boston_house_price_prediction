
# coding: utf-8

# In[1]:


from keras.datasets import boston_housing


# In[2]:


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


# In[3]:


train_data.shape, test_data.shape


# In[6]:


train_targets[0],len(train_targets)


# It would be problematic to feed into a neural network values that all take wildly different ranges. The network might be able to automatically adapt to such heterogeneous data, but it would definitely make learning more difficult. A widespread best practice to deal with such data is to do feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation.

# In[7]:


mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# In[8]:


from keras import models
from keras import layers

# Because you’ll need to instantiate the same model multiple times, you use a function to construct it.
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# In[1]:


# the network ends with a single unit and no actdivation. This is a typical setup for scaler regression. If you applied a sigmoid
# then network could predict between 0 and 1 only.
# hence the last layer is purely linear, so that model can predict in any range.
# validating your approach using K-fold validation


# To evaluate your network while you keep adjusting its parameters (such as the number of epochs used for training), you could split the data into a training set and a validation set, as you did in the previous examples. But because you have so few data points, the validation set would end up being very small (for instance, about 100 examples).As a consequence, the validation scores might change a lot depending on which data
# points you chose to use for validation and which you chose for training: the validation scores might have a high variance with regard to the validation split. This would prevent you from reliably evaluating your model.
# 
# The best practice in such situations is to use K -fold cross-validation.It consists of splitting the available data into K partitions (typically K = 4 or 5), instantiating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of
# the K validation scores obtained

# In[10]:


import numpy as np
k = 4

num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0)
#     Trains the model(in silent mode,verbose = 0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
        epochs=num_epochs, batch_size=1, verbose=0)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


# In[11]:


all_scores


# In[13]:


num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs, batch_size=1, verbose=0)
    
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)


# In[14]:


average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


# In[15]:


import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# It may be a little difficult to see the plot, due to scaling issues and relatively high variance. Let’s do the following:
# 
# Omit the first 10 data points, which are on a different scale than the rest of the curve.
#  Replace each point with an exponential moving average of the previous points, to obtain a smooth curve

# In[17]:


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# According to this plot, validation MAE stops improving significantly after 80 epochs. Past that point, you start overfitting.

# # Training the final model

# In[18]:


model = build_model()

# Trains it on the entirety of the data
model.fit(train_data, train_targets,
    epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


# In[19]:


test_mae_score


# In[20]:


# you're still off by about $2672


# Here’s what you should take away from this example:
# 
# 1. Regression is done using different loss functions than what we used for classification. Mean squared error ( MSE ) is a loss function commonly used for regression.
# 
# 2. Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is mean absolute error ( MAE ).
# 
# 3. When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step.
# 
# 4. When there is little data available, using K-fold validation is a great way to reliably evaluate a model.
# 
# 5. When little training data is available, it’s preferable to use a small network with few hidden layers (typically only one or two), in order to avoid severe overfitting.
