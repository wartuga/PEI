import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import arviz as az
import bridgestan as bs
import asyncio
import utils
import time
import functools

jax.config.update("jax_enable_x64", True)

# These paths are what they are because this example lives in a subfolder
# of the BridgeStan repository. If you're running this on your own, you
# will most likely want to delete the next line (to have BridgeStan
# download its sources for you) and change the paths on the following two
bs.set_bridgestan_path("..")

workdir = pathlib.Path(__file__).parent.parent.resolve()

# -----------------------------------

start_time = time.time()

count_data = np.loadtxt("txtdata.csv")
count_data = count_data.astype(int).tolist()
n_count_data = len(count_data)
count_sum = sum(count_data)
data = {'N': n_count_data, 'messages_count': count_data}

data_len = len(data.keys())
count = 0

with open("messages_data.json", mode="wt") as f:
    f.write("{\n")
    for k, v in data.items():
        if count != data_len - 1:
            f.write(" " + "\"" + k + "\"" + " : " + str(v) + ",\n")
        else:
            f.write(" " + "\"" + k + "\"" + " : " + str(v) + "\n")
        count += 1
    f.write("}")

stan = str(workdir) + "/test_models/messages/messages_data.stan"
data = str(workdir) + "/test_models/messages/messages_data.data.json"

model = bs.StanModel(stan, data)

# === Define log-probability for NUTS ===
@jax.custom_vjp
def logprob_fn(q):
    """Return log density for a given parameter vector q."""
    q_np = np.array(q, dtype=np.float64)
    return jnp.array(model.log_density(q_np))

def logprob_fwd(q):
    q_np = np.array(q, dtype=np.float64)
    logp, grad = model.log_density_gradient(q_np)
    return jnp.array(logp), grad  # value + residuals for backward pass

def logprob_bwd(res, g):
    grad = res
    return (g * jnp.array(grad),)

logprob_fn.defvjp(logprob_fwd, logprob_bwd)

initial_position = jnp.zeros(model.param_num())
inverse_mass_matrix = jnp.ones(model.param_num())
step_size = 1

nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
state = nuts.init(initial_position)
key = jax.random.PRNGKey(0)

# === Run inference ===
@functools.partial(jax.jit, static_argnums=(2,))
def run_inference(rng_key, state, num_samples):
    def one_step(state, rng_key):
        state, _ = nuts.step(rng_key, state)
        return state, state
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, state, keys)
    return states

states = run_inference(key, state, 1000)
positions = np.array(states.position)

# === Convert to dict of parameter samples ===
param_names = model.param_names()
samples_dict = {name: positions[:, i] for i, name in enumerate(param_names)}

# === ArviZ summary ===
idata = az.from_dict(posterior=samples_dict)

asyncio.run(utils.get_nutpie_statistics(idata))

print("--- %s seconds ---" % (time.time() - start_time))

# ------------------------------------

#model = bs.StanModel(stan, data)

#print(f"This model's name is {model.name()}.")
#print(f"It has {model.param_num()} parameters.")

#x = np.random.random(model.param_unc_num())
#q = np.log(x / (1 - x))  # unconstrained scale
#lp, grad = model.log_density_gradient(q, jacobian=False)
#print(f"log_density and gradient of Bernoulli model: {(lp, grad)}")