import ast
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.ConditionalNF import ConditionalNormalizingFlow
from src.MLEDefinitions import MLESampler
from src.SimulationData import LQRController, EnergyShapingController



np.random.seed(42)

if __name__ == '__main__':
    
    ANGLE_SWITCH_THRESHOLD_DEG = 18
    EPISODE_DONE_ANGLE_THRESHOLD_DEG = 0.5 # deg
    GRAVITY = 10.0

    env = gym.make("Pendulum-v1") #, render_mode = 'human')
    pendulum_params = {"mass": env.unwrapped.m,
                        "rod_length": env.unwrapped.l,
                        "gravity": GRAVITY,
                        "action_limits": (env.action_space.low, env.action_space.high),
                        'dt': env.unwrapped.dt}

    energy_controller = EnergyShapingController(**pendulum_params)
    lqr_controller = LQRController(**pendulum_params)

    state_dim = 2
    action_dim = 1

    mle_model = MLESampler(weight_files='Data/P(a|s).pth', input_dim=state_dim, output_dim=action_dim)
    nf_model = ConditionalNormalizingFlow(condition_dim=state_dim, n_flows=6, latent_dim=1)
    nf_model.load_state_dict(torch.load('Data/CNF.pth',weights_only=True))

    duration_episodes = []
    collected_data_cntrl = []
    collected_data_MLE = []
    collected_data_NF = []

    seed = np.random.randint(0, 1000)
    obs, _ = env.reset(seed=seed, options={'x_init': np.pi, 'y_init': 8.0})
    print(f'initial state: {obs}')
    done = False
    state = obs.squeeze().copy()
    upright_angle_buffer = []
    ctrl_type = None
    counter = 0

    while not done:
        angle = np.arctan2(obs[1], obs[0])
        pos_vel = np.array([angle, obs[2]]).squeeze()

        if abs(angle) < np.deg2rad(ANGLE_SWITCH_THRESHOLD_DEG):
            action_cntrl = lqr_controller.compute_control(pos_vel)
            ctrl_type = 'LQR'
        else:
            action_cntrl = energy_controller.get_action(pos_vel)
            ctrl_type = 'EnergyShaping'

        obs ,_ ,_ ,_, _ = env.step(action_cntrl)

        if abs(angle) < np.deg2rad(EPISODE_DONE_ANGLE_THRESHOLD_DEG):
            upright_angle_buffer.append(angle)
        if len(upright_angle_buffer) > 40:
            done = True

        collected_data_cntrl.append([action_cntrl.squeeze(),
                                        state.tolist(),
                                        ctrl_type])
        
        state = obs.squeeze().copy() # use .copy() for arrays because of the shared memory issues
        counter += 1

    # env.close()

    actions = np.array([data[0] for data in collected_data_cntrl])
    states_xy = np.array([data[1] for data in collected_data_cntrl])
    angles = np.array([np.arctan2(state[1], state[0]) for state in states_xy])
    vels = np.array([state[2] for state in states_xy])
    states = np.concatenate([angles.reshape(-1, 1), vels.reshape(-1, 1)], axis=1)

    print(f'action shape: {type(actions)},\nstate_xy shape: {type(states_xy)},\nstate_angle shape: {states.shape}')
    print(counter)
    lenght = counter
    
    obs, _ = env.reset(seed=seed, options={'x_init': np.pi, 'y_init': 8.0})
    print(f'initial state: {obs}')
    state = obs.squeeze().copy()
    while lenght:
        angle = np.arctan2(obs[1], obs[0])
        pos_vel = np.array([angle, obs[2]]).squeeze()
        model_input = torch.tensor(pos_vel, dtype=torch.float32)
        mle_model.model_input = model_input
        _, action_MLE, _ = mle_model.sample()
        action_mle = action_MLE.detach().numpy().reshape(-1, 1)
        clipped_mle = np.clip(action_mle, -2.0, 2.0)

        obs ,_ ,_ ,_, _ = env.step(clipped_mle)

        collected_data_MLE.append([clipped_mle.squeeze(),
                                        state.tolist()])
        
        state = obs.squeeze().copy() # use .copy() for arrays because of the shared memory issues
        lenght -= 1
    
    actions_mle = np.array([data[0] for data in collected_data_MLE])
    states_xy_mle = np.array([data[1] for data in collected_data_MLE])
    angles_mle = np.array([np.arctan2(state[1], state[0]) for state in states_xy_mle])
    vels_mle = np.array([state[2] for state in states_xy_mle])
    states_mle = np.concatenate([angles_mle.reshape(-1, 1), vels_mle.reshape(-1, 1)], axis=1)
    print(f'action shape: {actions_mle},\nstate_xy shape: {type(states_xy_mle)},\nstate_angle shape: {states_mle.shape}')
    print(counter)

    obs, _ = env.reset(seed=seed, options={'x_init': np.pi, 'y_init': 8.0})
    print(f'initial state: {obs}')
    state = obs.squeeze().copy()
    while counter:
        angle = np.arctan2(obs[1], obs[0])
        pos_vel = np.array([angle, obs[2]]).reshape(1,-1)
        model_input = torch.tensor(pos_vel, dtype=torch.float32)
        with torch.no_grad():
            base_mean, _ = nf_model.conditional_base(model_input)
            actions_pred,_ = nf_model.inverse(base_mean, model_input)
    
        actions_pred = actions_pred.cpu().numpy().reshape(-1,1)
        clipped_cnf = np.clip(actions_pred, -2, 2)

        obs ,_ ,_ ,_, _ = env.step(clipped_cnf)

        collected_data_NF.append([clipped_cnf.squeeze(),
                                        state.tolist()])
        
        state = obs.squeeze().copy() # use .copy() for arrays because of the shared memory issues
        counter -= 1
    
    actions_cnf = np.array([data[0] for data in collected_data_NF])
    states_xy_cnf = np.array([data[1] for data in collected_data_NF])
    angles_cnf = np.array([np.arctan2(state[1], state[0]) for state in states_xy_cnf])
    vels_cnf = np.array([state[2] for state in states_xy_cnf])
    states_cnf = np.concatenate([angles_cnf.reshape(-1, 1), vels_cnf.reshape(-1, 1)], axis=1)
    print(f'action shape: {actions_cnf},\nstate_xy shape: {type(states_xy_cnf)},\nstate_angle shape: {states_cnf.shape}')
    print(counter)

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
    # ax[0].plot(states[:, 0], label="Theta")
    # ax[1].plot(states[:, 1], label="Theta_dot")
    ax[0].plot(actions, label="Action", color='blue')
    ax[1].plot(actions_cnf, label="Predicted Action CNF")
    ax[2].plot(actions_mle, label="Predicted Action MLE", color='blue')
    ax[3].plot(actions, label="Action", color='blue')
    ax[3].plot(actions_cnf, label="Predicted Action CNF", color='red')
    ax[4].plot(actions, label="Action", color='blue')
    ax[4].plot(actions_mle, label="Predicted Action MLE", color='green')
    for i in range(5):
        ax[i].grid()
        ax[i].legend(loc='upper right')
        ax[i].set_ylabel("Torque [Nm]")

    ax[0].set_title(f"Performance")
    # plt.savefig(f'Data/Plots/{model_name}_comp.svg')
    plt.show()


