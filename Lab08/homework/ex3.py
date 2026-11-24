import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 31)

observed_counts = np.array([
    3, 4, 2, 5, 3, 4, 3, 2, 4, 3,
    4, 5, 5, 6, 5, 7, 6, 8, 7, 9,
    8, 10, 9, 8, 11, 9, 10, 8, 9, 10
])


# a

with pm.Model() as change_model:

    initial_rate = pm.Exponential(
        "initial_rate",
        lam=0.2
    )

    final_rate = pm.Exponential(
        "final_rate",
        lam=0.2
    )

    transition_midpoint = pm.Uniform(
        "transition_midpoint",
        lower=1,
        upper=30
    )

    transition_speed = pm.Exponential(
        "transition_speed",
        lam=1
    )

    sigmoid_values = pm.math.sigmoid(
        transition_speed * (days - transition_midpoint)
    )

    daily_rate = pm.Deterministic(
        "daily_rate",
        initial_rate
        + (final_rate - initial_rate) * sigmoid_values
    )

    pm.Poisson(
        "daily_counts",
        mu=daily_rate,
        observed=observed_counts
    )

    change_trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=1,
        target_accept=0.95,
        random_seed=42
    )


# b

az.plot_posterior(
    change_trace,
    var_names=[
        "transition_midpoint",
        "transition_speed"
    ],
    hdi_prob=0.95
)

plt.show()

midpoint_samples = (
    change_trace.posterior["transition_midpoint"]
    .values
    .reshape(-1)
)

speed_samples = (
    change_trace.posterior["transition_speed"]
    .values
    .reshape(-1)
)

print("Mediana tau:", np.median(midpoint_samples))
print("Mediana k:", np.median(speed_samples))

print("HDI 95% tau:", az.hdi(
    midpoint_samples,
    hdi_prob=0.95
))

print("HDI 95% k:", az.hdi(
    speed_samples,
    hdi_prob=0.95
))


# c

initial_rate_samples = (
    change_trace.posterior["initial_rate"]
    .values
    .reshape(-1)
)

final_rate_samples = (
    change_trace.posterior["final_rate"]
    .values
    .reshape(-1)
)

probability_increase = np.mean(
    final_rate_samples > initial_rate_samples
)

print("Probabilitatea ca rata finala sa fie mai mare:",
      probability_increase)


daily_rate_samples = (
    change_trace.posterior["daily_rate"]
    .values
    .reshape(-1, 30)
)

estimated_rate = np.median(
    daily_rate_samples,
    axis=0
)

lower_bound = np.quantile(
    daily_rate_samples,
    0.025,
    axis=0
)

upper_bound = np.quantile(
    daily_rate_samples,
    0.975,
    axis=0
)

plt.figure(figsize=(11, 5))

plt.scatter(
    days,
    observed_counts,
    label="Observatii"
)

plt.plot(
    days,
    estimated_rate,
    label="Rata estimata"
)

plt.fill_between(
    days,
    lower_bound,
    upper_bound,
    alpha=0.2,
    label="Interval credibil 95%"
)

plt.xlabel("Zi")
plt.ylabel("Numar de solicitari")
plt.title("Evolutia estimata a ratei zilnice")
plt.legend()

plt.show()


# d

with change_model:

    predictive_trace = pm.sample_posterior_predictive(
        change_trace,
        var_names=["daily_counts"],
        random_seed=42
    )

predicted_counts = (
    predictive_trace.posterior_predictive["daily_counts"]
    .values
    .reshape(-1, 30)
)

predicted_median = np.median(
    predicted_counts,
    axis=0
)

predicted_lower = np.quantile(
    predicted_counts,
    0.025,
    axis=0
)

predicted_upper = np.quantile(
    predicted_counts,
    0.975,
    axis=0
)

plt.figure(figsize=(11, 5))

plt.scatter(
    days,
    observed_counts,
    label="Observatii reale"
)

plt.plot(
    days,
    predicted_median,
    label="Mediana predictiva"
)

plt.fill_between(
    days,
    predicted_lower,
    predicted_upper,
    alpha=0.2,
    label="Interval predictiv 95%"
)

plt.xlabel("Zi")
plt.ylabel("Numar de solicitari")
plt.title("Distributia predictiva a posteriori")
plt.legend()

plt.show()