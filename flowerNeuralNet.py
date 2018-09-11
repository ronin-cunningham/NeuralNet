from matplotlib import pyplot as plt
import numpy as np
import os
SHOW_GRAPHS = False #switch to True if you wish to see the graphs




# each point is lenght, width, type. 1 = red, 0 = blue;
data = [[3,  1.5, 1],
        [2,   1,   0],
        [4,   1.5, 1],
        [3,   1,   0],
        [3.5, .5,  1],
        [2,   .5,  0],
        [5.5, 1,   1],
        [1,   1,   0]]
mystery_flower = [4.5, 1]



# network



def sigmoid(x):
    return 1/ (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))



T = np.linspace(-6, 6, 100)

plt.plot(T, sigmoid(T), c='r')
plt.plot(T, sigmoid_p(T), c='b')
if SHOW_GRAPHS == True:
    plt.show()
# Scatter data

plt.axis([0, 6, 0, 6])
plt.grid()
for i in range(len(data)):
    point = data[i]
    colour = 'r'
    if point[2] == 0:
        colour = 'b'
    plt.scatter(point[0], point[1], c=colour)
if SHOW_GRAPHS == True:
    plt.show()


# Training Loop

learning_rate = 0.2
costs = []

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

for i in range(50000):
    ri = np.random.randint(len(data))
    point = data[ri]

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]
    cost = np.square(pred - target)

    costs.append(cost)

    dcost_pred = 2 * (pred - target)
    dpred_z = sigmoid_p(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dz = dcost_pred * dpred_z

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db

plt.plot(costs)
if SHOW_GRAPHS == True:
    plt.show()

print("\n")

# model predictions
print("Model Predictions:")
for i in range(len(data)):
    point = data[i]
    print(point[2])
    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)
    print("pred: {}".format(pred))

print("\n")

z_mf = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
pred_mf = sigmoid(z_mf)
print("Mystery Flower Prediction: {}".format(pred_mf))

print("\n")

print("Your Input Prediction:")


def which_flower(length, width):
    z = length * w1 + width * w2 + b
    pred = sigmoid(z)
    if pred < 0.5:
        os.system("say this is a blue flower")
    else:
        os.system("say this is a red flower")
    print(pred)


which_flower(1, 2)

