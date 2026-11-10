import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

with pm.Model() as model:
    mu = pm.Normal("mu", mu=58, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)

    observations = pm.Normal(
        "observations",
        mu=mu,
        sigma=sigma,
        observed=data
    )

    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        random_seed=42
    )

print(az.summary(trace, var_names=["mu", "sigma"], hdi_prob=0.95))

az.plot_posterior(trace, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.show()