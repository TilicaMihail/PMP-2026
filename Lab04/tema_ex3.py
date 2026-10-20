import numpy as np
from itertools import product
from math import comb

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination




# A

rng = np.random.default_rng(67)
num_jocuri = 10000

incepatori = rng.integers(0, 2, size=num_jocuri)
zaruri = rng.integers(1, 7, size=num_jocuri)

probabilitati_stema = np.where(
    incepatori == 0,
    4 / 7,
    1 / 2
)

steme = rng.binomial(
    2 * zaruri,
    probabilitati_stema
)

castigatori = np.where(
    zaruri >= steme,
    incepatori,
    1 - incepatori
)

print("a) Probabilitati simulate de castig pentru Jucatorul x:")
print("J0:", np.mean(castigatori == 0))
print("J1:", np.mean(castigatori == 1))




# B

# S - cine incepe jocul (0 - Jucatorul x, 1 - Jucatorul y)
# N - zarul (1-6)
# M - cate steme au fost obtinute
# W - cine a castigat (0 - Jucatorul x, 1 - Jucatorul y)

model = DiscreteBayesianNetwork([
    ("S", "M"),
    ("N", "M"),
    ("S", "W"),
    ("N", "W"),
    ("M", "W")
])

cpd_s = TabularCPD(
    "S", 2,
    [[0.5], [0.5]]
)

cpd_n = TabularCPD(
    "N", 6,
    [[1 / 6] for _ in range(6)]
)

prob_m = [[] for _ in range(13)]

for s, n_idx in product(range(2), range(6)):
    n = n_idx + 1
    p = 4 / 7 if s == 0 else 1 / 2

    for m in range(13):
        if m <= 2 * n:
            prob = comb(2 * n, m) * p**m * (1 - p)**(2 * n - m)
        else:
            prob = 0.0

        prob_m[m].append(prob)

cpd_m = TabularCPD(
    "M", 13,
    prob_m,
    evidence=["S", "N"],
    evidence_card=[2, 6]
)

prob_w = [[], []]

for s, n_idx, m in product(range(2), range(6), range(13)):
    n = n_idx + 1

    w = s if n >= m else 1 - s

    prob_w[0].append(float(w == 0))
    prob_w[1].append(float(w == 1))

cpd_w = TabularCPD(
    "W", 2,
    prob_w,
    evidence=["S", "N", "M"],
    evidence_card=[2, 6, 13]
)

model.add_cpds(cpd_s, cpd_n, cpd_m, cpd_w)

assert model.check_model()

inference = VariableElimination(model)

result = inference.query(
    variables=["W"],
    show_progress=False
)

print("\nb) Probabilitatea ca Jucatorul x sa castige:")
print(result)




# C

result = inference.query(
    variables=["S"],
    evidence={"M": 1},
    show_progress=False
)

print("\nc) Probabilitatea ca Jucatorul x, dat fiind ca Jucatorul 1 a castigat:")
print(result)