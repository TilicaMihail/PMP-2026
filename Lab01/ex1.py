import random

def experiment():
    urna = ["R"] * 3 + ["A"] * 4 + ["N"] * 2

    zar = random.randint(1, 6)

    if zar in [2, 3, 5]:
        urna.append("N")
    elif zar == 6:
        urna.append("R")
    else:  
        urna.append("A")

    return random.choice(urna)

N = 1_000_000
rosii = 0

for _ in range(N):
    if experiment() == "R":
        rosii += 1

probabilitate_simulata = rosii / N
print("Probabilitatea simulata de a extrage o bila rosie este:", probabilitate_simulata)
print("Probabilitatea teoretica de a extrage o bila rosie este:", 19/60, "≈", 0.31666)


#probabilitatea teoretica este: P(r) = P1 * Pr1 + P2 * Pr2 + P3 * Pr3 = 3/6 * 3/10 + 1/6 * 4/10 + 2/6 * 3/10 = 19/60 ≈ 0.31666