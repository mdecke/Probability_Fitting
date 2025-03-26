import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)



#_________________________________________________________ Single Gaussian MLE _________________________________________________________________

class ContinuousActionNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousActionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(32, action_dim)
        self.log_sigma_head = nn.Linear(32, action_dim)  

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        return mu, log_sigma

class MLESampler:
    def __init__(self, weight_files:str, input_dim:int, output_dim:int):
        self.weight_file = weight_files
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_input = None

    def get_input(self, obs:np.ndarray, prev_action:np.ndarray=None, prev_obs:np.ndarray=None):
        theta = np.arctan2(obs[0,1], obs[0,0])
        theta_dot = obs[0,2]
        new_state = np.array([theta, theta_dot]).reshape(1, -1)
        if prev_action == None and prev_obs == None:
            input_mle = new_state
        elif prev_action.all() != None and prev_obs == None:
            input_mle = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
        elif prev_action.all() != None and prev_obs.all() != None:
            theta_prev = np.arctan2(prev_obs[0,1], prev_obs[0,0])
            theta_dot_prev = prev_obs[0,2]
            new_prev_state = np.array([theta_prev, theta_dot_prev]).reshape(1, -1)
            intermed = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
            input_mle = np.concatenate((intermed, new_prev_state), axis=1)
        else:
            raise ValueError("Invalid input type. Must be 'state' or 'state_action' or 'prev_state_action'")
        
        self.model_input = torch.tensor(input_mle, dtype=torch.float32)
    
    def sample(self):
        self.model_NN = ContinuousActionNN(state_dim=self.input_dim, action_dim=self.output_dim)
        self.model_NN.load_state_dict(torch.load(self.weight_file, weights_only=True))
        self.model_NN.eval()
        if self.model_input is None:
            raise ValueError("Input not set. Please call get_input() method first.")
        # print(self.model_input)
        mu, log_sigma = self.model_NN(self.model_input)
        # print(mu, log_sigma)
        sigma = torch.exp(log_sigma) + 1e-5
        dist = torch.distributions.Normal(mu, sigma) # Add small value to avoid case where sigma is zero
        action = dist.sample()
        return action ,mu, sigma

def gaussian_nll_loss(mu, log_std, target):
    std = torch.exp(log_std)
    variance = std ** 2
    log_variance = 2 * log_std
    
    nll = 0.5 * (
        log_variance + 
        ((target - mu) ** 2) / variance + 
        torch.log(2 * torch.tensor(np.pi))
    )
    return nll.mean()

def train_model_with_validation(model, data, unique_episodes, epochs, nb_of_batches, nb_trajectories, model_name, input_type, device='cpu',seed=None):
    # Split episodes into train and validation
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    random_episode_order = np.random.permutation(unique_episodes)
    val_size = int(len(unique_episodes) * 0.2) # 20% validation
    train_episodes = random_episode_order[:-val_size]
    val_episodes = random_episode_order[-val_size:]
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5, 
                                                   verbose=True)
    
    # Initialize early stopping
    model_save_path = os.path.join('Data/', 
                                  f'{model_name}.pth')
    early_stopping = EarlyStopping(patience=10, path=model_save_path)
    
    model = model.to(device)
    
    def process_batch(episodes_subset):
        batch_loss = 0.0
        for _ in range(nb_trajectories):
            # Sample random episode
            trajectory_idx = np.random.choice(episodes_subset, size=1)
            
            # Extract states and actions
            th = data[data['episode'].isin(trajectory_idx)]['angle_state'].to_numpy()
            th_d = data[data['episode'].isin(trajectory_idx)]['angle_vel'].to_numpy()
            states = np.array([th, th_d]).T
            actions = data[data['episode'].isin(trajectory_idx)]['actions'].to_numpy()

            prev_th = data[data['episode'].isin(trajectory_idx)]['angle_state'].to_numpy()
            prev_th_d = data[data['episode'].isin(trajectory_idx)]['angle_vel'].to_numpy()
            prev_states = np.array([prev_th, prev_th_d]).T
            previous_actions = data[data['episode'].isin(trajectory_idx)]['prev_action'].to_numpy()

            # Convert to tensors
            batch_states = torch.tensor(states, dtype=torch.float32).to(device)
            batch_actions = torch.tensor(actions, dtype=torch.float32).reshape(-1, 1).to(device)

            if input_type == 'state':
                model_input = batch_states
            elif input_type == 'state_action':
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32).reshape(-1, 1)
                model_input = torch.cat((batch_states, batch_previous_actions), dim=1)
            elif input_type == 'prev_state_action':
                batch_previous_states = torch.tensor(prev_states, dtype=torch.float32)
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32).reshape(-1, 1)
                model_input = torch.cat((batch_states, batch_previous_actions, batch_previous_states), dim=1)
            else:
                raise ValueError("Invalid data type. Must be 'states' or 'states_action' or 'prev_states_action'")
            
            # Forward pass
            mu, log_std = model(model_input)
            loss = gaussian_nll_loss(mu, log_std, batch_actions)
            batch_loss += loss
            
        return batch_loss / nb_trajectories
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for idx_batch in range(nb_of_batches):
            optimizer.zero_grad()
            
            batch_loss = process_batch(train_episodes)
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(batch_loss.item())
            
            if idx_batch % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {idx_batch}/{nb_of_batches} | '
                      f'Loss: {batch_loss.item():.6f}')
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for _ in range(nb_of_batches // 4):  # Fewer validation batches
                val_batch_loss = process_batch(val_episodes)
                val_losses.append(val_batch_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Log metrics
        print(f'Epoch: {epoch+1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f}')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    
    return model, history, val_episodes


#_________________________________________________________ MLE for Gaussian Mixture _________________________________________________________________

class GMMLE(nn.Module):
    def __init__(self, state_dim, action_dim, num_components=5):
        super(GMMLE, self).__init__()
        self.num_components = num_components
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mixture_weights_head = nn.Linear(32, num_components)
        self.mu_head = nn.Linear(32, num_components*action_dim)
        self.log_sigma_head = nn.Linear(32, num_components*action_dim)  

    def forward(self, state):
        batch_size = state.size(0)

        x = self.fc(state)
        
        alpha = torch.softmax(self.mixture_weights_head(x), dim=-1)
        
        mu = self.mu_head(x)
        mu = mu.view(batch_size, self.num_components, -1)
        
        log_sigma = self.log_sigma_head(x)
        log_sigma = log_sigma.view(batch_size, self.num_components, -1)
        return alpha, mu, log_sigma
    
class GMMLESampler:
    def __init__(self, weight_files:str, input_dim:int, output_dim:int, num_components:int=3):
        self.weight_file = weight_files
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_components = num_components
        self.model_input = None

    def get_input(self, obs:np.ndarray, prev_action:np.ndarray=None, prev_obs:np.ndarray=None):
        theta = np.arctan2(obs[0,1], obs[0,0])
        theta_dot = obs[0,2]
        new_state = np.array([theta, theta_dot]).reshape(1, -1)
        if prev_action == None and prev_obs == None:
            input_mle = new_state
        elif prev_action.all() != None and prev_obs == None:
            input_mle = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
        elif prev_action.all() != None and prev_obs.all() != None:
            theta_prev = np.arctan2(prev_obs[0,1], prev_obs[0,0])
            theta_dot_prev = prev_obs[0,2]
            new_prev_state = np.array([theta_prev, theta_dot_prev]).reshape(1, -1)
            intermed = np.concatenate((new_state, prev_action.reshape(1,-1)), axis=1)
            input_mle = np.concatenate((intermed, new_prev_state), axis=1)
        else:
            raise ValueError("Invalid input type. Must be 'state' or 'state_action' or 'prev_state_action'")
        
        self.model_input = torch.tensor(input_mle, dtype=torch.float32)
    
    def sample(self):
        self.model_NN = GMMLE(state_dim=self.input_dim, action_dim=self.output_dim, num_components=self.num_components)
        self.model_NN.load_state_dict(torch.load(self.weight_file, weights_only=True))
        self.model_NN.eval()
        if self.model_input is None:
            raise ValueError("Input not set. Please call get_input() method first.")
        alpha, mu, log_sigma = self.model_NN(self.model_input)
        mu.squeeze_(-1)
        log_sigma.squeeze_(-1)
        k = torch.distributions.Categorical(alpha).sample()
        means = torch.diagonal(mu[:,k], offset=0)
        log_stds = torch.diagonal(log_sigma[:,k], offset=0)
        stds = torch.exp(log_stds) + 1e-5
        dist = torch.distributions.Normal(means, stds)
        action = dist.sample()
        return action , means, stds

def gaussian_mixture_nll_loss(alpha, mu, log_std, target):
    """
    alpha: [B, K]
    mu: [B, K, d]
    log_sigma: [B, K, d]
    target: [B, d]
    
    Returns the average negative log-likelihood across the batch.
    """
    B, K, d = mu.shape
    target = target.unsqueeze(1) 
    target = target.expand(-1, K, -1)

    std = torch.exp(log_std) + 1e-8
    variance = std ** 2
    log_variance = 2 * log_std

    log_gaussian = -0.5 * (log_variance + (target - mu)**2 / variance + torch.log(2 * torch.tensor(np.pi)))
    log_gaussian_d = torch.sum(log_gaussian, dim=2)
    
    log_alpha = torch.log(alpha + 1e-15)
    log_probs = torch.logsumexp(log_alpha + log_gaussian_d, dim=1)
    nll = -torch.mean(log_probs)

    return nll

def train_gmm_model_with_validation(model, data, unique_episodes, epochs, nb_of_batches, nb_trajectories, model_name, input_type, device='cpu', seed=None):
    # Split episodes into train and validation
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    random_epsiode_order = np.random.permutation(unique_episodes)
    val_size = int(len(unique_episodes) * 0.2) # 20% validation
    train_episodes = random_epsiode_order[:-val_size]
    val_episodes = random_epsiode_order[-val_size:]
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5, 
                                                   verbose=True)
    
    # Initialize early stopping
    model_save_path = os.path.join('Data/Models/Noise/GMM', 
                                  f'{model_name}.pth')
    early_stopping = EarlyStopping(patience=10, path=model_save_path)
    
    model = model.to(device)
    
    def process_batch(episodes_subset):
        batch_loss = 0.0
        for _ in range(nb_trajectories):
            # Sample random episode
            trajectory_idx = np.random.choice(episodes_subset, size=1)
            
            # Extract states and actions
            th = data[data['episode'].isin(trajectory_idx)]['angle_state'].to_numpy()
            th_d = data[data['episode'].isin(trajectory_idx)]['angle_vel'].to_numpy()
            states = np.array([th, th_d]).T
            actions = data[data['episode'].isin(trajectory_idx)]['actions'].to_numpy()

            prev_th = data[data['episode'].isin(trajectory_idx)]['angle_state'].to_numpy()
            prev_th_d = data[data['episode'].isin(trajectory_idx)]['angle_vel'].to_numpy()
            prev_states = np.array([prev_th, prev_th_d]).T
            previous_actions = data[data['episode'].isin(trajectory_idx)]['prev_action'].to_numpy()

            # Convert to tensors
            batch_states = torch.tensor(states, dtype=torch.float32).to(device)
            batch_actions = torch.tensor(actions, dtype=torch.float32).reshape(-1, 1).to(device)

            if input_type == 'state':
                model_input = batch_states
            elif input_type == 'state_action':
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32).reshape(-1, 1)
                model_input = torch.cat((batch_states, batch_previous_actions), dim=1)
            elif input_type == 'prev_state_action':
                batch_previous_states = torch.tensor(prev_states, dtype=torch.float32)
                batch_previous_actions = torch.tensor(previous_actions, dtype=torch.float32).reshape(-1, 1)
                model_input = torch.cat((batch_states, batch_previous_actions, batch_previous_states), dim=1)
            else:
                raise ValueError("Invalid data type. Must be 'states' or 'states_action' or 'prev_states_action'")
            
            # Forward pass
            gm_weights, mu, log_std = model(model_input)
            loss = gaussian_mixture_nll_loss(gm_weights, mu, log_std, batch_actions)
            batch_loss += loss
            
        return batch_loss / nb_trajectories
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for idx_batch in range(nb_of_batches):
            optimizer.zero_grad()
            
            batch_loss = process_batch(train_episodes)
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(batch_loss.item())
            
            if idx_batch % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {idx_batch}/{nb_of_batches} | '
                      f'Loss: {batch_loss.item():.6f}')
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for _ in range(nb_of_batches // 4):  # Fewer validation batches
                val_batch_loss = process_batch(val_episodes)
                val_losses.append(val_batch_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Log metrics
        print(f'Epoch: {epoch+1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f}')
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    
    return model, history, val_episodes