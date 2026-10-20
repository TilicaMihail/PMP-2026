from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# B - bolnav
# T - tuse
# X radiografie anormala
# D - dificultati in respiratie

model = DiscreteBayesianNetwork([
    ("B", "T"),
    ("B", "X"),
    ("B", "D"),
    ("T", "D")
])

cpd_b = TabularCPD(
    "B", 2,
    [[0.9], [0.1]]
)

cpd_t = TabularCPD(
    "T", 2,
    [
        [0.7, 0.2],
        [0.3, 0.8]
    ],
    evidence=["B"],
    evidence_card=[2]
)

cpd_x = TabularCPD(
    "X", 2,
    [
        [0.9, 0.1],
        [0.1, 0.9]
    ],
    evidence=["B"],
    evidence_card=[2]
)

cpd_d = TabularCPD(
    "D", 2,
    [
        [0.9, 0.5, 0.6, 0.1],
        [0.1, 0.5, 0.4, 0.9]
    ],
    evidence=["B", "T"],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_b, cpd_t, cpd_x, cpd_d)

assert model.check_model()

inference = VariableElimination(model)

result_b = inference.query(
    variables=["B"],
    evidence={"D": 1},
    show_progress=False
)

print("P(B | D=1):")
print(result_b)

result_c = inference.query(
    variables=["X"],
    evidence={"B": 0, "D": 1},
    show_progress=False
)

print("P(X | B=0, D=1):")
print(result_c)