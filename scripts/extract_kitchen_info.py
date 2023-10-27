from xskill.dataset.kitchen_mjl_lowdim_dataset import KitchenMjlLowdimDataset
from xskill.env.kitchen.v0 import KitchenAllV0
from xskill.utility.utils import read_json, write_json
import numpy as np

base_dev_dir = "your/path/to/xskill"
kd = KitchenMjlLowdimDataset(
    dataset_dir=f"{base_dev_dir}/xskill/datasets/kitchen/kitchen_demos_multitask"
)
task_completions = []
env = KitchenAllV0(use_abs_action=True)
for i in range(kd.replay_buffer.n_episodes):
    obs = env.reset()
    eps_data = kd.replay_buffer.get_episode(i)
    reset_pos = np.concatenate([eps_data["obs"][0, :9], eps_data["obs"][0, 9:30]])
    env.robot.reset(env, reset_pos, env.init_qvel[:].copy())
    for j in range(len(eps_data["action"])):
        _, _, _, info = env.step(eps_data["action"][j])
    task_completions.append(info["completed_tasks"])

task_completions_list = [list(arr) for arr in task_completions]

write_json(
    f"{base_dev_dir}/xskill/datasets/kitchen_dataset/task_completions.json",
    task_completions_list,
)

# test read
read_json(f"{base_dev_dir}/xskill/datasets/kitchen_dataset/task_completions.json")

eval_mask = np.zeros(len(task_completions_list), dtype=bool)
for i, d in enumerate(task_completions_list):
    if (
        "kettle" in d
        and "light switch" in d
        and "slide cabinet" in d
        and "microwave" in d
    ):
        eval_mask[i] = True

write_json(
    f"{base_dev_dir}/xskill/datasets/kitchen_dataset/eval_mask.json",
    eval_mask.tolist(),
)

train_mask = ~eval_mask
write_json(
    f"{base_dev_dir}/xskill/datasets/kitchen_dataset/train_mask.json",
    train_mask.tolist(),
)
