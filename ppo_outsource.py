import os
import torch
import rlgym

from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ProbabilisticActor, ValueOperator, MLP
from torchrl.objectives.ppo import PPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict.nn import TensorDictModule  # ✅ fixed import


# ============================
# 1. Environment
# ============================
def make_env():
    # Give each env a unique pipe_id to avoid "All pipe instances are busy"
    base_env = rlgym.make(pipe_id=os.getpid())
    return GymWrapper(base_env)


env = make_env()

# Inspect observation/action specs
print("Observation spec:", env.observation_spec)
print("Action spec:", env.action_spec)

# Fix for CompositeSpec: pull out the "observation" key
obs_dim = env.observation_spec["observation"].shape[-1]   # should be 70
act_dim = env.action_spec.shape[-1]


# ============================
# 2. Networks
# ============================
# Actor base net
actor_net = MLP(
    in_features=obs_dim,
    out_features=2 * act_dim,  # mean + log_std for Gaussian
    depth=2,
    num_cells=256
)

# Wrap actor net so TorchRL knows it reads from "observation"
actor_module = TensorDictModule(
    module=actor_net,
    in_keys=["observation"],
    out_keys=["param"],   # outputs parameters for distribution
)

# Critic base net
critic_net = MLP(
    in_features=obs_dim,
    out_features=1,
    depth=2,
    num_cells=256
)

critic_module = TensorDictModule(
    module=critic_net,
    in_keys=["observation"],
    out_keys=["state_value"],
)


# ============================
# 3. Wrap with ProbabilisticActor & ValueOperator
# ============================
actor = ProbabilisticActor(
    module=actor_module,
    spec=env.action_spec,
    in_keys=["param"],
    distribution_class="Normal",
    distribution_kwargs={"scale": 0.1},  # initial std
    return_log_prob=True,
)

critic = ValueOperator(
    module=critic_module,
    in_keys=["observation"],
    out_keys=["state_value"],
)


# ============================
# 4. PPO Loss
# ============================
loss_module = PPOLoss(
    actor=actor,
    critic=critic,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    critic_coef=0.5,
)


# ============================
# 5. Collector + Replay Buffer
# ============================
collector = SyncDataCollector(
    create_env_fn=make_env,
    policy=actor,
    frames_per_batch=2048,
    total_frames=50_000,
)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(100_000)
)


# ============================
# 6. Optimizer
# ============================
optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)


# ============================
# 7. Training Loop
# ============================
print("🚀 Starting PPO training on Rocket League (rlgym)...")

for batch_idx, batch in enumerate(collector):

    # store rollouts
    replay_buffer.extend(batch)

    # sample minibatch
    data = replay_buffer.sample(64)

    # compute PPO loss
    loss_vals = loss_module(data)
    loss = loss_vals["loss_objective"]

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
    optimizer.step()

    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

print("✅ Training finished")
