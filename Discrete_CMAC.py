
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# take 100 points
x = np.arange(0, 100)
y = []


# 1D cosine function
for i in range(0, 100):
    Y = math.cos((x[i]) * (2*np.pi/100))
    y = np.append(y,Y)

# plotting the input cosine curve
plt.figure(1)
plt.plot((x * (2*np.pi/100)), y, '-k', label="Input Curve", linewidth=2)

# dividing into train and test set
np.random.shuffle(x)

x_train = x[:70]  # random 70 points as train
x_train = np.sort(x_train)


x_test = x[70:] # remaining 30 point as test
x_test = np.sort(x_test)

# Initializing weights

w_value = 0.0
w_zero = []
w_save =  w_zero
error = mean_sq_array = total_time= gen = w_rand = weight_save = []
mean_sq = 1
start = time.time()      

start = time.time()

for g in range(1, 37,2):
    print(g)
    weight = np.random.rand(35)  # 35 weights
    padding = (g - 1) / 2
    w_zero = np.array([0])

    for i in range(int(padding)):  # Creating a Layer of Padding on both the Ends
        weight = np.append(w_zero, weight)
        weight = np.append(weight, w_zero)

    while (mean_sq > 0.01): # ensure the error is less than 0.01
        w_rand = np.arange(0)

        # Training Phase
        for j in range(0, 70):
            q = j / 2

            for k in range(g):
                w_value = w_value + weight[int(k) + int(q)]

            w_new_value = w_value / g
            y_train = (math.cos(x_train[j] * (2*np.pi/100)))

            e = y_train - w_new_value
            error = np.append(error, e)
            corrected_val = e / g  # Error Correction

            for k in range(g):
                weight[int(k) + int(q)] = (weight[int(k) + int(q)]) + corrected_val

            w_value = 0.0

        mean_sq = np.mean(error ** 2)


    if g == 3:       # use different values of g
        w_save = w_rand
        weight_save = weight


    gen = np.append(gen, g) # store g
    end = time.time()
    total_time = np.append(total_time, (end - start)) # save times for different g
    mean_sq_array = np.append(mean_sq_array, mean_sq) # save error for different g
    mean_sq = 1


w_new = weight_save[1::2]
gen1 = 3
w_new_array = []


# Testing 
for i in range(0, 30):
    q = i / 2
    w_avg = w_new[int(q)] + w_new[int(q) - 1] + w_new[int(q) + 1]
    w_avg= w_avg / gen1
    w_new_array = np.append(w_new_array, w_avg)

# plotting results

plt.figure(1)
plt.plot((x_test * (2*np.pi/100)), w_new_array, '-r', label="Test (gen = 3)")
plt.title("Original vs test curve")
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.legend()

plt.figure(2)
plt.plot(gen, mean_sq_array)
plt.title("Effect of generalization on error")
plt.xlabel("Generalization")
plt.ylabel("Error")

plt.figure(3)
plt.plot(gen, total_time)
plt.title("Effect of generalization on time")
plt.xlabel("Generalization")
plt.ylabel("Time")
plt.show()