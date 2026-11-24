import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

valori_y = [0, 5, 10]
valori_theta = [0.2, 0.5]

rezultate = {}

for y in valori_y:
    for theta in valori_theta:
        with pm.Model() as model:

            n = pm.Poisson("n", mu=10, initval=max(10, y))

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
                random_seed=42,
                progressbar=False
            )

        rezultate[(y, theta)] = idata

fig, axes = plt.subplots(
    len(valori_y),
    len(valori_theta),
    figsize=(12, 10)
)

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
plt.savefig("posterior_n.png", dpi=150)
plt.show()

for (y, theta), idata in rezultate.items():

    esantion = idata.posterior["n"].values.flatten()

    print(
        f"Y={y:2d} theta={theta} "
        f"media={esantion.mean():.2f} "
        f"std={esantion.std():.2f} "
        f"min={esantion.min()}"
    )