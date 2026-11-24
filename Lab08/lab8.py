import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

valori_y = [0, 5, 10]
valori_theta = [0.2, 0.5]

rezultate = {}

for y in valori_y:
    for theta in valori_theta:

        with pm.Model() as model:

            n = pm.Poisson("n", mu=10)

            pm.Binomial(
                "y_obs",
                n=n,
                p=theta,
                observed=y
            )

            idata = pm.sample(
                draws=4000,
                tune=1000,
                chains=2,
                cores=1,
                step=pm.Metropolis(),
                random_seed=42
            )

        rezultate[(y, theta)] = idata

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for row_index, y in enumerate(valori_y):
    for column_index, theta in enumerate(valori_theta):

        az.plot_posterior(
            rezultate[(y, theta)],
            var_names=["n"],
            ax=axes[row_index, column_index]
        )

        axes[row_index, column_index].set_title(
            f"Y = {y}, theta = {theta}"
        )

plt.tight_layout()
plt.show()