import os
import sys
import numpy as np
import matplotlib
print (matplotlib.get_backend())
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_loss = []
    validation_loss = []
    with open("track_train_loss.txt", 'r') as f:
        for line in f:
            train_loss.append(float(line[:-1]))
    f.close()
    with open("track_validation_loss.txt", 'r') as f:
        for line in f:
            validation_loss.append(float(line[:-1]))
    f.close()
    plt.figure()
    plt.plot([x for x in range(len(train_loss))],train_loss)
    plt.plot([x for x in range(len(train_loss))],validation_loss)
    plt.legend(["training loss", 'validation loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Average difference in percentage")
    plt.show()


