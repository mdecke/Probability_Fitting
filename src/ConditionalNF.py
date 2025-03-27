import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import pandas as pd
from src.DataProcessing import augment_data
from sklearn.model_selection import train_test_split
from src.MLEDefinitions import MLESampler

np.random.seed(42)

class ConditionalBase(nn.Module):
    def __init__(self, condition_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * latent_dim)  # outputs concatenated mean and log_std
        )
    
    def forward(self, condition):
        params = self.net(condition)
        latent_dim = params.shape[1] // 2
        mean = params[:, :latent_dim]
        log_std = params[:, latent_dim:]
        return mean, log_std
    

class ConditionalAffineLayer(nn.Module):
    def __init__(self, condition_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  
        )
    
    def forward(self, a, condition):
        params = self.net(condition)  
        s = params[:, 0:1]            # log-scale parameter
        t = params[:, 1:2]            # translation
        scale = torch.exp(s)          # ensure scale is positive
        # Forward transform: a -> latent z.
        z = (a - t) / scale
        log_det = -torch.log(scale).squeeze(1)
        return z, log_det
    
    def inverse(self, z, condition):
        params = self.net(condition)
        s = params[:, 0:1]
        t = params[:, 1:2]
        scale = torch.exp(s)
        # Inverse transform: latent z -> a.
        a = scale * z + t
        log_det = torch.log(scale).squeeze(1)
        return a, log_det
    
class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, condition_dim, n_flows, latent_dim=1):
        super().__init__()
        self.n_flows = n_flows
        self.layers = nn.ModuleList([ConditionalAffineLayer(condition_dim) for _ in range(n_flows)])
        self.conditional_base = ConditionalBase(condition_dim, latent_dim)
    
    def forward(self, a, condition):
        # Map action a to latent variable z.
        log_det_total = 0.0
        z = a
        for layer in self.layers:
            z, log_det = layer(z, condition)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z, condition):
        
        log_det_total = 0.0
        a = z
        for layer in reversed(self.layers):
            a, log_det = layer.inverse(a, condition)
            log_det_total += log_det
        return a, log_det_total
    
    def log_prob(self, a, condition):
        z, log_det = self.forward(a, condition)
        
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        base_dist = torch.distributions.Normal(base_mean, base_std)
        log_base = base_dist.log_prob(z).squeeze(1)
        return log_base + log_det
    
    def sample(self, num_samples, condition):
        # Sample latent variable from the conditional base.
        base_mean, base_log_std = self.conditional_base(condition)
        base_std = torch.exp(base_log_std)
        base_dist = torch.distributions.Normal(base_mean, base_std)
        z = base_dist.rsample()  # reparameterized sample; shape: (num_samples, latent_dim)
        a, _ = self.inverse(z, condition)
        return a
    
    
if __name__ == '__main__':
        
    DATA = 'Data/CSVs/data_5000.csv'

    print('modifying data')
    data = pd.read_csv(DATA)
    data['states'] = data['states'].apply(lambda x: ast.literal_eval(x))
    augmented_data = augment_data(data)
    augmented_data['angle_state'] = augmented_data['states'].apply(lambda x: np.arctan2(x[1], x[0]))
    augmented_data['angle_vel'] = augmented_data['states'].apply(lambda x: x[2])
    augmented_data['prev_angle_state'] = augmented_data['prev_state'].apply(lambda x: np.arctan2(x[1], x[0]))
    augmented_data['prev_angle_vel'] = augmented_data['prev_state'].apply(lambda x: x[2])
    episodes = augmented_data['episode'].unique()
    print('data modified')


    train_episodes, test_episodes = train_test_split(episodes, test_size=0.2)
    # print(train_episodes)
    train = augmented_data[augmented_data['episode'].isin(train_episodes)]
    test = augmented_data[augmented_data['episode'].isin(test_episodes)]
    # print(train.head())
    # print(test.head())


    actions_train = torch.tensor(train['actions'].to_numpy(), dtype=torch.float32)
    angles_train = torch.tensor(train['angle_state'].to_numpy(), dtype=torch.float32)
    vels_train = torch.tensor(train['angle_vel'].to_numpy(), dtype=torch.float32)

    actions_test = torch.tensor(test['actions'].to_numpy(), dtype=torch.float32)
    angles_test = torch.tensor(test['angle_state'].to_numpy(), dtype=torch.float32)
    vels_test = torch.tensor(test['angle_vel'].to_numpy(), dtype=torch.float32)

    s_train = torch.stack([angles_train, vels_train], dim=1)
    s_test = torch.stack([angles_test, vels_test], dim=1)

    a_train = actions_train.unsqueeze(1)
    a_val   = actions_test.unsqueeze(1)


    train_dataset = TensorDataset(s_train, a_train)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('data ready')


    # Create the model instance.
    n_flows = 6  # Using more flows for flexibility.
    model = ConditionalNormalizingFlow(condition_dim=2, n_flows=n_flows, latent_dim=1)

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30

    def train_flow(model, train_loader, optimizer, num_epochs):
        model.train()
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            for states_batch, actions_batch in train_loader:
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)
                optimizer.zero_grad()
                # Compute negative log likelihood.
                log_prob = model.log_prob(actions_batch, states_batch)
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * actions_batch.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        return losses

    losses = train_flow(model, train_loader, optimizer, num_epochs)

    # Plot training loss.
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


    torch.save(model.state_dict(), "Data/CNF.pth")
    
    model.eval()
    with torch.no_grad():
        # Move validation states to device.
        # For an affine flow, the conditional mean is given by the inverse mapping of z=0.
        base_mean, _ = model.conditional_base(s_test.to(device))
        actions_pred,_ = model.inverse(base_mean, s_test)
        actions_pred = actions_pred.cpu().numpy().flatten()
        val_states_np = s_test.cpu().numpy()
        val_actions_np = a_val.cpu().numpy().flatten()


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the true actions (blue)
    ax.scatter(val_states_np[:200, 0], val_states_np[:200, 1], val_actions_np[:200], color='blue', label='True Actions')
    # Plot the predicted actions (red)
    ax.scatter(val_states_np[:200, 0], val_states_np[:200, 1], actions_pred[:200], color='red', label='Predicted Actions')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Theta_dot')
    ax.set_zlabel('Action')
    ax.legend()
    plt.title("True Actions vs. Predicted Actions from Conditional Normalizing Flow")
    plt.show()

    rdm_episode = np.random.choice(test_episodes)


    th = data[data['episode']==rdm_episode]['angle_state'].to_numpy()
    th_d = data[data['episode']==rdm_episode]['angle_vel'].to_numpy()
    states = np.array([th, th_d]).T
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions = data[data['episode']==rdm_episode]['actions'].to_numpy()

    sampler = MLESampler(weight_files=f'Data/P(a|s).pth', input_dim=2, output_dim=1)
    sampler.model_input = states_tensor

    with torch.no_grad():
        base_mean, _ = model.conditional_base(states_tensor.to(device))
        actions_pred,_ = model.inverse(base_mean, states_tensor)
    
    actions_pred = actions_pred.cpu().numpy().flatten()
    clipped_actions_pred = np.clip(actions_pred, -2, 2)

    actions_pred_mle, _,_ = sampler.sample()
    clipped_actions_pred_mle = np.clip(actions_pred_mle, -2, 2)

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
    # ax[0].plot(states[:, 0], label="Theta")
    # ax[1].plot(states[:, 1], label="Theta_dot")
    ax[0].plot(actions, label="Action", color='blue')
    ax[1].plot(clipped_actions_pred, label="Predicted Action CNF")
    ax[2].plot(clipped_actions_pred_mle, label="Predicted Action MLE", color='blue')
    ax[3].plot(actions, label="Action", color='blue')
    ax[3].plot(clipped_actions_pred, label="Predicted Action (mean)", color='red')
    ax[4].plot(actions, label="Action", color='blue')
    ax[4].plot(actions_pred_mle, label="Predicted Action MLE", color='green')
    for i in range(5):
        ax[i].grid()
        ax[i].legend(loc='upper right')
        ax[i].set_ylabel("Torque [Nm]")

    ax[0].set_title(f"Performance for Episode {rdm_episode}")
    # plt.savefig(f'Data/Plots/{model_name}_comp.svg')
    plt.show()

