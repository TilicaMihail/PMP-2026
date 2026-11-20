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
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

model.emissionprob_ = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

observed_grades = [
    "FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"
]

observations = np.array([
    [grades.index(grade)] for grade in observed_grades
])

log_probability, hidden_states = model.decode(
    observations,
    algorithm="viterbi"
)

decoded_states = [states[index] for index in hidden_states]

print("Dificultati:", decoded_states)
print("Probabilitate comuna:", np.exp(log_probability))
