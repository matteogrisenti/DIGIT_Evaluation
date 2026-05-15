import pickle
with open("../datasets/digit-force/sphere/batch_1/dataset_slip_forces.pkl", "rb") as f:
    labels = pickle.load(f)
first_traj_key = list(labels['trajectories'].keys())[0]
print(f"Inside Trajectory '{first_traj_key}': {labels['trajectories'][first_traj_key].keys()}")