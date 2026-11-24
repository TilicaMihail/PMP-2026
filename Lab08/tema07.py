from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.stats import beta as beta_distribution
from scipy.stats import hypergeom

output_directory = Path("grafice")
output_directory.mkdir(parents=True, exist_ok=True)
random_generator = np.random.default_rng(42)

# 1.a
marked_animals = 20
capture_size = 15
observed_marked = 4

with pm.Model() as population_model:
    population_size = pm.DiscreteUniform(
        "population_size", lower=20, upper=200
    )
    pm.HyperGeometric(
        "observed_marks",
        N=population_size,
        k=marked_animals,
        n=capture_size,
        observed=observed_marked,
    )
    posterior_1 = pm.sample(
        draws=5000,
        tune=2500,
        chains=4,
        cores=1,
        step=pm.Metropolis(),
        random_seed=42,
        return_inferencedata=True,
    )

population_samples = posterior_1.posterior["population_size"].values.reshape(-1)
print(az.summary(posterior_1, var_names=["population_size"], hdi_prob=0.95))

candidate_populations = np.arange(20, 201)
likelihood_values = hypergeom.pmf(
    observed_marked, candidate_populations, marked_animals, capture_size
)
exact_posterior = likelihood_values / likelihood_values.sum()

plt.figure(figsize=(11, 4))
plt.bar(candidate_populations, exact_posterior, alpha=0.7, label="Posterior exact")
plt.hist(
    population_samples,
    bins=np.arange(19.5, 201.5, 1),
    density=True,
    alpha=0.4,
    label="Posterior PyMC",
)
plt.xlabel("Numărul total de animale N")
plt.ylabel("Probabilitate / frecvență")
plt.title("Ex. 1: posteriorul populației")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory / "ex1_posterior_N.png", dpi=160)
plt.show()

# 1.b
population_mean = np.sum(candidate_populations * exact_posterior)
population_median = candidate_populations[
    np.searchsorted(np.cumsum(exact_posterior), 0.5)
]

smallest_width = np.inf
credible_interval = None
for lower_index in range(len(candidate_populations)):
    for upper_index in range(lower_index, len(candidate_populations)):
        interval_probability = exact_posterior[lower_index : upper_index + 1].sum()
        if interval_probability >= 0.95:
            interval_width = upper_index - lower_index
            if interval_width < smallest_width:
                smallest_width = interval_width
                credible_interval = (
                    int(candidate_populations[lower_index]),
                    int(candidate_populations[upper_index]),
                )
            break

classical_estimate = marked_animals * capture_size / observed_marked
print(f"Media exactă: {population_mean:.3f}")
print(f"Mediana exactă: {population_median}")
print(f"Interval contiguu de probabilitate 95%: {credible_interval}")
print(f"Estimarea clasică M*n/Y: {classical_estimate:.3f}")

# 1.c
probability_over_100_exact = exact_posterior[candidate_populations > 100].sum()
probability_over_100_mcmc = np.mean(population_samples > 100)
print(f"P(N > 100 | Y=4) exact: {probability_over_100_exact:.5f}")
print(f"P(N > 100 | Y=4) PyMC: {probability_over_100_mcmc:.5f}")

# 1.d
new_marked_values = np.arange(capture_size + 1)
exact_predictive = np.array([
    np.sum(
        exact_posterior
        * hypergeom.pmf(
            new_marked_count,
            candidate_populations,
            marked_animals,
            capture_size,
        )
    )
    for new_marked_count in new_marked_values
])

new_capture_samples = random_generator.hypergeometric(
    ngood=marked_animals,
    nbad=population_samples - marked_animals,
    nsample=capture_size,
)
print(f"Media predictivă exactă: {np.dot(new_marked_values, exact_predictive):.3f}")

plt.figure(figsize=(10, 4))
plt.bar(new_marked_values, exact_predictive, label="Predictiv exact")
plt.hist(
    new_capture_samples,
    bins=np.arange(-0.5, 16.5, 1),
    density=True,
    alpha=0.4,
    label="Predictiv din posteriorul PyMC",
)
plt.xlabel("Număr marcate într-o nouă captură")
plt.ylabel("Probabilitate")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory / "ex1_posterior_predictive.png", dpi=160)
plt.show()

# 2.a
observations_a = np.array([3.2, 4.1, 2.8, 5.0, 3.7, 4.5, 3.0, 3.9])
observations_b = np.array([2.1, 2.8, 3.4, 2.5, 3.0, 2.2, 3.6, 2.7])

with pm.Model() as treatments_model:
    rate_a = pm.Gamma("rate_a", alpha=2, beta=4)
    rate_b = pm.Gamma("rate_b", alpha=2, beta=4)
    pm.Exponential("observed_a", lam=rate_a, observed=observations_a)
    pm.Exponential("observed_b", lam=rate_b, observed=observations_b)
    mean_days_a = pm.Deterministic("mean_days_a", 1 / rate_a)
    mean_days_b = pm.Deterministic("mean_days_b", 1 / rate_b)
    posterior_2 = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=2,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
    )

print(az.summary(
    posterior_2,
    var_names=["rate_a", "rate_b", "mean_days_a", "mean_days_b"],
    hdi_prob=0.95,
))

# 2.b
plt.figure(figsize=(10, 4))
az.plot_posterior(
    posterior_2,
    var_names=["mean_days_a", "mean_days_b"],
    hdi_prob=0.95,
)
plt.tight_layout()
plt.savefig(output_directory / "ex2_timpi_medii_HDI.png", dpi=160)
plt.show()

shape_a = 2 + len(observations_a)
shape_b = 2 + len(observations_b)
rate_parameter_a = 4 + observations_a.sum()
rate_parameter_b = 4 + observations_b.sum()

print(f"E[mean_days_A | date] exact = {rate_parameter_a / (shape_a - 1):.3f}")
print(f"E[mean_days_B | date] exact = {rate_parameter_b / (shape_b - 1):.3f}")

# 2.c
mean_a_samples = posterior_2.posterior["mean_days_a"].values.reshape(-1)
mean_b_samples = posterior_2.posterior["mean_days_b"].values.reshape(-1)
rate_a_samples = posterior_2.posterior["rate_a"].values.reshape(-1)
rate_b_samples = posterior_2.posterior["rate_b"].values.reshape(-1)

probability_mean_b_faster = np.mean(mean_b_samples < mean_a_samples)
threshold = rate_parameter_b / (rate_parameter_a + rate_parameter_b)
probability_exact = beta_distribution.sf(threshold, shape_b, shape_a)
print(f"P(timp mediu B < timp mediu A) PyMC: {probability_mean_b_faster:.4f}")
print(f"P(timp mediu B < timp mediu A) exact: {probability_exact:.6f}")

# 2.d
probability_predictive_integrated = np.mean(
    rate_b_samples / (rate_a_samples + rate_b_samples)
)
number_of_posterior_draws = len(mean_a_samples)
new_patient_times_a = random_generator.exponential(
    scale=mean_a_samples[:, None],
    size=(number_of_posterior_draws, 20),
)
new_patient_times_b = random_generator.exponential(
    scale=mean_b_samples[:, None],
    size=(number_of_posterior_draws, 20),
)
probability_predictive_simulated = np.mean(
    new_patient_times_b < new_patient_times_a
)

print(f"P(pacient nou B mai rapid) prin integrare MC: {probability_predictive_integrated:.4f}")
print(f"P(pacient nou B mai rapid) prin simulări: {probability_predictive_simulated:.4f}")
print(f"P(timp mediu B mai mic) din 2c: {probability_mean_b_faster:.4f}")

# 3.a
days = np.arange(1, 31)
daily_observations = np.array([
    3, 4, 2, 5, 3, 4, 3, 2, 4, 3,
    4, 5, 5, 6, 5, 7, 6, 8, 7, 9,
    8, 10, 9, 8, 11, 9, 10, 8, 9, 10,
])

assert len(days) == len(daily_observations) == 30

with pm.Model() as change_model:
    initial_rate = pm.Exponential("initial_rate", lam=0.2)
    final_rate = pm.Exponential("final_rate", lam=0.2)
    transition_midpoint = pm.Uniform("transition_midpoint", lower=1, upper=30)
    transition_speed = pm.Exponential("transition_speed", lam=1)

    sigmoid_values = pm.math.sigmoid(
        transition_speed * (days - transition_midpoint)
    )
    daily_rate = pm.Deterministic(
        "daily_rate",
        initial_rate + (final_rate - initial_rate) * sigmoid_values,
    )
    pm.Poisson("daily_counts", mu=daily_rate, observed=daily_observations)

    posterior_3 = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=2,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
    )

parameter_names_3 = [
    "initial_rate", "final_rate", "transition_midpoint", "transition_speed"
]
print(az.summary(posterior_3, var_names=parameter_names_3, hdi_prob=0.95))
print("Număr divergente:", int(posterior_3.sample_stats["diverging"].sum()))

# 3.b
az.plot_posterior(
    posterior_3,
    var_names=["transition_midpoint", "transition_speed"],
    hdi_prob=0.95,
)
plt.tight_layout()
plt.savefig(output_directory / "ex3_tau_k_HDI.png", dpi=160)
plt.show()

az.plot_trace(posterior_3, var_names=parameter_names_3)
plt.tight_layout()
plt.savefig(output_directory / "ex3_traces.png", dpi=160)
plt.show()

# 3.c
initial_rate_samples = posterior_3.posterior["initial_rate"].values.reshape(-1)
final_rate_samples = posterior_3.posterior["final_rate"].values.reshape(-1)
probability_increasing = np.mean(final_rate_samples > initial_rate_samples)

posterior_daily_rate = posterior_3.posterior["daily_rate"].values.reshape(-1, 30)
rate_median = np.median(posterior_daily_rate, axis=0)
rate_lower, rate_upper = np.quantile(
    posterior_daily_rate, [0.025, 0.975], axis=0
)

print(f"P(lambda_2 > lambda_1 | date): {probability_increasing:.4f}")
plt.figure(figsize=(11, 5))
plt.scatter(days, daily_observations, color="black", label="Observații")
plt.plot(days, rate_median, label="Mediana ratei estimate")
plt.fill_between(
    days, rate_lower, rate_upper, alpha=0.2,
    label="Interval central de 95% pentru rată",
)
plt.xlabel("Zi")
plt.ylabel("Număr de solicitări / rată medie")
plt.title("Ex. 3: traiectoria ratei și observațiile")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory / "ex3_rata_vs_observatii.png", dpi=160)
plt.show()

# 3.d
with change_model:
    predictive_3 = pm.sample_posterior_predictive(
        posterior_3,
        var_names=["daily_counts"],
        random_seed=42,
        return_inferencedata=True,
    )

replicated_counts = (
    predictive_3.posterior_predictive["daily_counts"].values.reshape(-1, 30)
)
predictive_median = np.median(replicated_counts, axis=0)
predictive_lower, predictive_upper = np.quantile(
    replicated_counts, [0.025, 0.975], axis=0
)

fraction_in_predictive_band = np.mean(
    (daily_observations >= predictive_lower)
    & (daily_observations <= predictive_upper)
)
print(f"Proporția zilelor în banda predictivă de 95%: {fraction_in_predictive_band:.3f}")

plt.figure(figsize=(11, 5))
plt.scatter(days, daily_observations, color="black", label="Observații")
plt.plot(days, predictive_median, label="Mediana predictivă")
plt.fill_between(
    days,
    predictive_lower,
    predictive_upper,
    alpha=0.25,
    label="Interval predictiv central de 95%",
)
plt.xlabel("Zi")
plt.ylabel("Număr de solicitări")
plt.title("Ex. 3: verificare predictivă a posteriori")
plt.legend()
plt.tight_layout()
plt.savefig(output_directory / "ex3_posterior_predictive.png", dpi=160)
plt.show()