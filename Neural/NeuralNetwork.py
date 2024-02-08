from MachineLearning import *

x = [7.9, 3.1, 7.5, 1.8]

probs = predict(x)
pred_class = np.argmax(probs)
class_names = ['Setosa', 'Versicolor', 'Virginica']
print('Predicted class:', class_names[pred_class])