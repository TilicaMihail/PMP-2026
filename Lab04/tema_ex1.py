from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# C - usa aleasa de competitor
# P - usa cu premiu
# H - usa deschisa de host

model = DiscreteBayesianNetwork([
    ("C", "H"),
    ("P", "H")
])

cpd_c = TabularCPD("C", 3, [[1/3], [1/3], [1/3]])
cpd_p = TabularCPD("P", 3, [[1/3], [1/3], [1/3]])

cpd_h = TabularCPD(
    "H",
    3,
    [
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]
    ],
    evidence=["C", "P"],
    evidence_card=[3, 3]
)

model.add_cpds(cpd_c, cpd_p, cpd_h)

print(model.get_cpds())
print(model.check_model())

infer = VariableElimination(model)

posterior_p = infer.query(
    ["P"],
    evidence={"C": 0, "H": 2}
)

print(posterior_p)