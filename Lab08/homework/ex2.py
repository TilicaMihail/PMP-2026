import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

observations_a = np.array([
    3.2, 4.1, 2.8, 5.0, 3.7, 4.5, 3.0, 3.9
])

observations_b = np.array([
    2.1, 2.8, 3.4, 2.5, 3.0, 2.2, 3.6, 2.7
])

# a

with pm.Model() as treatments_model:

    rate_a = pm.Gamma(
        "rate_a",
        alpha=2,
        beta=4
    )

    rate_b = pm.Gamma(
        "rate_b",
        alpha=2,
        beta=4
    )

    pm.Exponential(
        "observed_a",
        lam=rate_a,
        observed=observations_a
    )

    pm.Exponential(
        "observed_b",
        lam=rate_b,
        observed=observations_b
    )

    mean_days_a = pm.Deterministic(
        "mean_days_a",
        1 / rate_a
    )

    mean_days_b = pm.Deterministic(
        "mean_days_b",
        1 / rate_b
    )

    treatments_trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=1,
        target_accept=0.95,
        random_seed=42
    )


# b

az.plot_posterior(
    treatments_trace,
    var_names=["mean_days_a", "mean_days_b"],
    hdi_prob=0.95
)

plt.show()

summary = az.summary(
    treatments_trace,
    var_names=[
        "rate_a",
        "rate_b",
        "mean_days_a",
        "mean_days_b"
    ],
    hdi_prob=0.95
)

print(summary)


# c

rate_a_samples = (
    treatments_trace.posterior["rate_a"]
    .values
    .reshape(-1)
)

rate_b_samples = (
    treatments_trace.posterior["rate_b"]
    .values
    .reshape(-1)
)

comparison = rate_b_samples > rate_a_samples

probability_b_better = np.mean(comparison)

print(probability_b_better)


# d

random_generator = np.random.default_rng(42)

mean_a_samples = 1 / rate_a_samples
mean_b_samples = 1 / rate_b_samples

new_patient_a = random_generator.exponential(
    scale=mean_a_samples
)

new_patient_b = random_generator.exponential(
    scale=mean_b_samples
)

probability_patient_b_faster = np.mean(
    new_patient_b < new_patient_a
)

print(probability_patient_b_faster)