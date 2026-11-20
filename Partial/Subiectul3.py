import pymc as pm
import arviz as az

# A
with pm.Model() as coin_model:
    theta = pm.Beta("theta", alpha=2, beta=8)
    k = pm.Binomial("k", n=20, p=theta, observed=5)
    idata = pm.sample(draws=1000, tune=1000,
    chains=4, random_seed=41)
print(az.summary(idata, var_names=["theta"]))

pm.sample_posterior_predictive(idata, model=coin_model)

# C

with pm.Model() as coin_model:
    theta = pm.Beta("theta", alpha=2, beta=8)
    k = pm.Binomial("k", n=15, p=theta, observed=3)
    idata = pm.sample(draws=1000, tune=1000,
    chains=4, random_seed=41)
print(az.summary(idata, var_names=["theta"]))

pm.sample_posterior_predictive(idata, model=coin_model)