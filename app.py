import numpy
import pandas
import sklearn
from sklearn import linear_model

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pandas.read_csv('student-mat.csv', sep=';')
# print(data.head()) # Prints first 5 elements

# take relevant columns for our calculation
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
# print(data)

# What we are going to predict
predict = 'G3'

# Create a data set without the predict
x = numpy.array(data.drop([predict], 1))
# print(x)

# Get all actual predict values
y = numpy.array(data[predict])
# print(y)

# Keeping here as well to use it once training is over
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# # Iterate and find a best accuracy result and save to pickle
# best_result = 0
# for _ in range(1000):
#
#     # Create a section of x data (x_train), section of y (y_train) and test data
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     # Use liner regression only if there is correlation (best fit line). In this case predict final grade based on previous grades.
#     # y = mx+b
#     # m --> Slope = y2 - y1/ x2 - x1
#     # b = y interceptor, where the best fit line starts
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     if accuracy > best_result:  # Check if current is better than previous
#         best_result = accuracy
#         print(accuracy)
#
#         # Save the model
#         with open('studentmodel.pickle', 'wb') as f:
#             pickle.dump(linear, f)

# Read the pickle
pickle_in = open('studentmodel.pickle', 'rb')
# Load the pickle to linear
linear = pickle.load(pickle_in)
# Now training is over, comment out the training process once pickle is generated.

print('Co-efficient : \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()

