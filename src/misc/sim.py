import numpy as np
import scipy.special
import os

if __name__ == "__main__":
    similarity_matrix = np.load(os.path.join("../../data", "SIFT", "1M", "sim.npy"))
    similarity_matrix /= np.max(similarity_matrix, axis=-1, keepdims=True)
    similarity_matrix = scipy.special.softmax(similarity_matrix, -1)
    print("saving")
    np.save(os.path.join("../../data", "SIFT", "1M", "sim_softmax.npy"), similarity_matrix)
    print("saved")
    exit()