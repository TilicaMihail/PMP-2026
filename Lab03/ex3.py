from itertools import product

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


model = DiscreteBayesianNetwork([
    ("S", "O"),
    ("S", "L"),
    ("S", "M"),
    ("L", "M")
])


cpd_s = TabularCPD(
    variable="S",
    variable_card=2,
    values=[[0.6], [0.4]]
)

cpd_o = TabularCPD(
    variable="O",
    variable_card=2,
    values=[
        [0.9, 0.3],  
        [0.1, 0.7]   
    ],
    evidence=["S"],
    evidence_card=[2]
)

cpd_l = TabularCPD(
    variable="L",
    variable_card=2,
    values=[
        [0.7, 0.2],  
        [0.3, 0.8]   
    ],
    evidence=["S"],
    evidence_card=[2]
)

cpd_m = TabularCPD(
    variable="M",
    variable_card=2,
    values=[
        [0.8, 0.4, 0.5, 0.1],  
        [0.2, 0.6, 0.5, 0.9]   
    ],
    evidence=["S", "L"],
    evidence_card=[2, 2]
)

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

assert model.check_model(), "Reteaua este incorecta!"

# a

print("a) Independente conditionate:")
print(model.get_independencies())

# b

inference = VariableElimination(model)

print("\nb) Clasificarea e-mailurilor:")
print("O L M | P(Spam) | Clasificare")
print("-" * 40)

for o, l, m in product([0, 1], repeat=3):

    rezultat = inference.query(
        variables=["S"],
        evidence={"O": o, "L": l, "M": m},
        show_progress=False
    )

    p_spam = float(rezultat.get_value(S=1))

    clasificare = "SPAM" if p_spam > 0.5 else "NON-SPAM"

    print(
        f"{o} {l} {m} | "
        f"{p_spam * 100:.2f}% | {clasificare}"
    )