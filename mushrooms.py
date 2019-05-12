import sys
import numpy as np

def ReadMushrooms():
    labels = []
    filename = "mushrooms.txt"
    
    f = open(filename, "r")
    lines = f.readlines()

    features = []
    count = 0
    for l in lines:
        w = l.split()
        labels.append(int(w[0]))
        features_line = [0]*112
        for i in range(1, len(w)):
            u = w[i].split(":")
            features_line[int(u[0])-1] = 1
        features.append(features_line)

    return np.matrix(features), np.hstack(labels)


# X,b = ReadMushrooms()
# print(X)
# print(b)
