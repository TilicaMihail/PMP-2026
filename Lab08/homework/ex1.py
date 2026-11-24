import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

marked_animals = 20
capture_size = 15
observed_marked = 4

# a

with pm.Model() as population_model:

    population_size = pm.DiscreteUniform(
        "population_size",
        lower=20,
        upper=200
    )

    pm.HyperGeometric(
        "marked_observations",
        N=population_size,
        k=marked_animals,
        n=capture_size,
        observed=observed_marked
    )

    population_trace = pm.sample(
        draws=3000,
        tune=2000,
        chains=4,
        cores=1,
        step=pm.Metropolis(),
        random_seed=42
    )

az.plot_posterior(
    population_trace,
    var_names=["population_size"],
    hdi_prob=0.95
)

plt.show()


# b

population_samples = (
    population_trace.posterior["population_size"]
    .values
    .reshape(-1)
)

posterior_mean = np.mean(population_samples)

posterior_median = np.median(population_samples)

posterior_hdi = az.hdi(
    population_samples,
    hdi_prob=0.95
)

classical_estimate = (
    marked_animals * capture_size / observed_marked
)

print("Media:", posterior_mean)
print("Mediana:", posterior_median)
print("HDI 95%:", posterior_hdi)
print("Estimarea clasică:", classical_estimate)


# c

probability_above_100 = np.mean(
    population_samples > 100
)

print("Probabilitatea estimata că dimensiunea populației este mai mare decât 100:", probability_above_100)

possible_populations = np.arange(20, 201)

likelihood_values = hypergeom.pmf(
    4,
    possible_populations,
    20,
    15
)

posterior_probabilities = (
    likelihood_values / likelihood_values.sum()
)

exact_probability = posterior_probabilities[
    possible_populations > 100
].sum()

print("Probabilitatea exacta că dimensiunea populației este mai mare decât 100:", exact_probability)


# d

random_generator = np.random.default_rng(42)

new_capture_samples = random_generator.hypergeometric(
    ngood=marked_animals,
    nbad=population_samples - marked_animals,
    nsample=capture_size
)

print("Media predictivă:", np.mean(new_capture_samples))

plt.hist(
    new_capture_samples,
    bins=np.arange(-0.5, 16.5, 1),
    density=True
)

plt.xlabel("Animale marcate în noua captură")
plt.ylabel("Probabilitate")
plt.title("Distribuția predictivă a posteriori")

plt.show()