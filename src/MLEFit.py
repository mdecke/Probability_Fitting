import ast

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.MLEDefinitions import ContinuousActionNN, MLESampler, train_model_with_validation
from  src.DataProcessing import augment_data


FILE_PATH = 'Data/CSVs/data_5000.csv'
INPUT_TYPE = 'state'#'state_action' or 'state' or 'prev_state_action'
# SEED = 42


if __name__ == '__main__':
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    
    print('modifying data')
    data = pd.read_csv(FILE_PATH)
    data['states'] = data['states'].apply(lambda x: ast.literal_eval(x))
    augmented_data = augment_data(data)
    augmented_data['angle_state'] = augmented_data['states'].apply(lambda x: np.arctan2(x[1], x[0]))
    augmented_data['angle_vel'] = augmented_data['states'].apply(lambda x: x[2])
    augmented_data['prev_angle_state'] = augmented_data['prev_state'].apply(lambda x: np.arctan2(x[1], x[0]))
    augmented_data['prev_angle_vel'] = augmented_data['prev_state'].apply(lambda x: x[2])
    episodes = augmented_data['episode'].unique()
    print('data modified')
    

    if INPUT_TYPE == 'state':
        model_name = 'P(a|s)'
        input_dim = 2
    elif INPUT_TYPE == 'state_action':
        model_name = 'P(a|s,a-1)'
        input_dim = 3
    elif INPUT_TYPE == 'prev_state_action':
        model_name = 'P(a|s,a-1,s-1)'
        input_dim = 5
        
    model = ContinuousActionNN(input_dim, 1)
    model, history, val_episodes = train_model_with_validation(model,
                                                            data=augmented_data,
                                                            unique_episodes=episodes,
                                                            epochs=100,
                                                            nb_of_batches=32,
                                                            nb_trajectories=20,
                                                            model_name = model_name,
                                                            input_type=INPUT_TYPE)
    
    model.eval()

    val_df = data[data['episode'].isin(val_episodes)]

    # Randomly sample 20% of those rows (set random_state for reproducibility if desired).
    val_df_sampled = val_df.sample(frac=0.2, random_state=42)

    # Now extract only from the sampled dataframe
    val_th = val_df_sampled['angle_state'].to_numpy()
    val_th_d = val_df_sampled['angle_vel'].to_numpy()
    val_actions = val_df_sampled['actions'].to_numpy()

    val_prev_th = val_df_sampled['angle_state'].to_numpy()
    val_prev_th_d = val_df_sampled['angle_vel'].to_numpy()
    val_prev_actions = val_df_sampled['prev_action'].to_numpy()

    # Convert to arrays/tensors
    val_states = np.column_stack([val_th, val_th_d])
    val_states_tensor = torch.tensor(val_states, dtype=torch.float32)

    val_prev_states = np.column_stack([val_prev_th, val_prev_th_d])
    val_prev_states_tensor = torch.tensor(val_prev_states, dtype=torch.float32)

    val_prev_actions_tensor = torch.tensor(val_prev_actions.reshape(-1, 1), dtype=torch.float32)

    # Use whichever INPUT_TYPE logic you have
    if INPUT_TYPE == 'state':
        model_input = val_states_tensor
    elif INPUT_TYPE == 'state_action':
        model_input = torch.cat((val_states_tensor, val_prev_actions_tensor), dim=1)
    elif INPUT_TYPE == 'prev_state_action':
        model_input = torch.cat((val_states_tensor, val_prev_actions_tensor, val_prev_states_tensor), dim=1)

    # Predict
    with torch.no_grad():
        actions_pred, _ = model(model_input)

    actions_pred = actions_pred.numpy().flatten()

    # Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_states[:, 0], val_states[:, 1], val_actions, color='blue', label='True')
    ax.scatter(val_states[:, 0], val_states[:, 1], actions_pred, color='red', label='Pred')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Action')
    plt.legend()
    # plt.savefig(f'Data/Plots/{model_name}_fit.svg')
    plt.show()
    
    val_th = data[data['episode'].isin(val_episodes)]['angle_state'].to_numpy()
    val_th_d = data[data['episode'].isin(val_episodes)]['angle_vel'].to_numpy()
    val_states = np.array([val_th, val_th_d]).T
    val_states_tensor = torch.tensor(val_states, dtype=torch.float32)
    val_actions = data[data['episode'].isin(val_episodes)]['actions'].to_numpy()

    val_prev_th = data[data['episode'].isin(val_episodes)]['angle_state'].to_numpy()
    val_prev_th_d = data[data['episode'].isin(val_episodes)]['angle_vel'].to_numpy()
    val_prev_states = np.array([val_prev_th, val_prev_th_d]).T
    val_prev_states_tensor = torch.tensor(val_prev_states, dtype=torch.float32)
    val_prev_actions = data[data['episode'].isin(val_episodes)]['prev_action'].to_numpy()
    val_prev_actions_tensor = torch.tensor(val_prev_actions.reshape(-1,1), dtype=torch.float32)

    if INPUT_TYPE == 'state':
        model_input = val_states_tensor
    elif INPUT_TYPE == 'state_action':
        model_input = torch.cat((val_states_tensor, val_prev_actions_tensor), dim=1)
    elif INPUT_TYPE == 'prev_state_action':
        model_input = torch.cat((val_states_tensor, val_prev_actions_tensor, val_prev_states_tensor), dim=1)
            

    with torch.no_grad():
        actions_pred, _ = model(model_input)
    actions_pred = actions_pred.numpy().flatten()   
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_states[:, 0], val_states[:, 1], val_actions, color='blue')
    ax.scatter(val_states[:, 0], val_states[:, 1], actions_pred, color='red')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Action')
    # plt.savefig(f'Data/Plots/{model_name}_fit.svg')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL Loss')
    # plt.savefig(f'Data/Plots/{model_name}_Loss.svg')
    plt.show()
    


    #___________________________________________________ Test with Sampler _____________________________________________________

    sampler = MLESampler(weight_files=f'Data/Models/Noise/MLE/{model_name}.pth', input_dim=input_dim, output_dim=1)
    
    rdm_episode = np.random.choice(val_episodes)

    # rdm_episode = 0
    th = data[data['episode']==rdm_episode]['angle_state'].to_numpy()
    th_d = data[data['episode']==rdm_episode]['angle_vel'].to_numpy()
    states = np.array([th, th_d]).T
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions = data[data['episode']==rdm_episode]['actions'].to_numpy()

    prev_th = data[data['episode']==rdm_episode]['prev_angle_state'].to_numpy()
    prev_th_d = data[data['episode']==rdm_episode]['prev_angle_vel'].to_numpy()
    prev_states = np.array([prev_th, prev_th_d]).T
    prev_states_tensor = torch.tensor(prev_states, dtype=torch.float32)
    prev_actions = data[data['episode']==rdm_episode]['prev_action'].to_numpy()
    prev_actions_tensor = torch.tensor(prev_actions.reshape(-1,1), dtype=torch.float32)

        
    if INPUT_TYPE == 'state':
        sampler.model_input = states_tensor
    elif INPUT_TYPE == 'state_action':
        sampler.model_input = torch.cat((states_tensor, prev_actions_tensor), dim=1)
    elif INPUT_TYPE == 'prev_state_action':
        sampler.model_input = torch.cat((states_tensor, prev_actions_tensor, prev_states_tensor), dim=1)


    actions_pred, mu,sigma = sampler.sample()
    mu = mu.detach().numpy().squeeze()
    sigma = sigma.detach().numpy().squeeze()
    clipped_actions_pred = np.clip(actions_pred, -2, 2)
    diff = actions - mu   
    shifted_mu = mu[1:]
    shifted_actions = actions[:-1]
    shifted_diff = shifted_actions - shifted_mu 

    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8, 8))
    # ax[0].plot(states[:, 0], label="Theta")
    # ax[1].plot(states[:, 1], label="Theta_dot")
    ax[0].plot(actions, label="Action", color='blue')
    ax[1].plot(clipped_actions_pred, label="Predicted Action Sample")
    ax[2].plot(mu, label="Predicted Action (mean)")
    ax[3].plot(actions, label="Action", color='blue')
    ax[3].plot(mu, label="Predicted Action (mean)", color='red')
    ax[4].plot(diff, label="difference action & mu")
    ax[5].plot(shifted_diff, label="difference action[i-1] & mu[i]")
    for i in range(6):
        ax[i].grid()
        ax[i].legend(loc='upper right')
        ax[i].set_ylabel("Torque [Nm]")

    ax[0].set_title(f"Performance for Episode {rdm_episode}")
    # plt.savefig(f'Data/Plots/{model_name}_comp.svg')
    plt.show()

    upper_bound = mu[:] + 2*sigma[:]
    lower_bound = mu[:] - 2*sigma[:]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(mu, label="Predicted Action (mean)")
    ax.plot(actions[:-1], label="Action", color='blue')
    ax.fill_between(range(len(mu)), upper_bound, lower_bound, alpha=0.2, label='+/- 2 sigma')
    ax.set_title(f"Performance for Episode {rdm_episode}")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Torque [Nm]")
    ax.grid()
    ax.legend(loc='upper right')
    # plt.savefig(f'Data/Plots/{model_name}_fit_{rdm_episode}.svg')
    plt.show()

    