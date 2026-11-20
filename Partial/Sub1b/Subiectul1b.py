import numpy as np
from hmmlearn.hmm import CategoricalHMM
import matplotlib.pyplot as plt
import networkx as nx


states = ["1", "2", "3", "4"]
colors = ["R", "G", "B"]

model = CategoricalHMM(
    n_components=4,
    n_features=3,
    init_params="",
    params=""
)

model.startprob_ = np.array([
    1/4, 1/4, 1/4, 1/4
])

model.transmat_ = np.array([
    [0, 0.5, 0, 0.5],
    [0.5, 0, 0.5, 0],
    [0, 0.5, 0, 0.5], 
    [0.5, 0, 0.5, 0]
])

model.emissionprob_ = np.array([
    [0.7, 0.2, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
    [0.5, 0.2, 0.3]
])

observed_colors = [
    "R", "G", "B", "B", "R"
]

observations = np.array([
    [colors.index(color)] for color in observed_colors
])

log_probability, hidden_states = model.decode(
    observations,
    algorithm="viterbi"
)

decoded_states = [states[index] for index in hidden_states]

print("Camere:", decoded_states)
print("Probabilitate comuna:", np.exp(log_probability))

# Se obtine aceeasi secventa de camere ca la punctul a dar cu o probabilitate comuna diferita
