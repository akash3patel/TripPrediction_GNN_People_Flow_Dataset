#importing required libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

import os
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
np.random.seed(42)

import torch
import torch.nn as nn
import torch.optim as optim



import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split

@hydra.main(version_base=None, config_path="conf", config_name="config")
def gnn_training(cfg : DictConfig):
    #run = wandb.init(project="PeopleFlow", job_type="training")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    df  = pd.read_csv(cfg['data_file'])
    df = df.loc[:, df.columns != 'Departure_time']
    df.head()
    df_Zone = pd.read_csv(cfg['feature_file'])
    filtered_df = pd.read_csv(cfg['filtered_file'])
    df_Zone.rename(columns={"T000918002": "business_secondary"},inplace=True)
    df_Zone.rename(columns={"T000918006": "employees_secondary"},inplace=True)
    df_Zone.rename(columns={"T000918021": "business_tertiary"},inplace=True)
    df_Zone.rename(columns={"T000918025": "employees_tertiary"},inplace=True)
    df_Zone.rename(columns={"T000847001": "Night_Population"},inplace=True)

    # trip_type_array = [df['Trip_type'].values]
    # trip_type_array = np.unique(trip_type_array)
    # trip_type_array

    df_1 = filtered_df.merge(df_Zone, left_on='Origin', right_on='ZONE_ID')
    df_full = df_1.merge(df_Zone, left_on='Destination', right_on='ZONE_ID', suffixes=('_origin', '_destination'))
    df_full.head()
    #G.add_nodes_from([2, 3])
    import numpy as np

    # df_edges = df_full.drop_duplicates(subset=['Origin', 'Destination'])
    # df_edges.shape

    e1 = np.array(df_full["Origin"]).reshape((-1,1))
    e2 = np.array(df_full["Destination"]).reshape((-1,1))
    print (e1.shape, e2.shape)
    e = np.unique(np.vstack((e1,e2)).squeeze()).tolist()
    mapped_zone_id_vs_zone_id = {val: i for i, val in enumerate(e)}


    # identity_matrix = identity_matrix = np.eye(len(df_sample["Origin"]))
    # n = np.dot(identity_matrix, df_sample["Origin"])
    # print(n)

    print (len(mapped_zone_id_vs_zone_id))


    # In[12]:


    df_full['mapped_Origin'] = df_full['Origin'].apply(lambda x : mapped_zone_id_vs_zone_id[x])
    df_full['mapped_Destination'] = df_full['Destination'].apply(lambda x : mapped_zone_id_vs_zone_id[x])


    # In[13]:


    df_full


    # In[14]:


    ed1 = np.array(df_full["mapped_Origin"])
    ed2 = np.array(df_full["mapped_Destination"])

    ed = np.column_stack((ed1,ed2)).T
    ed.shape


    # In[15]:


    df_node = df_full.drop_duplicates(subset=['mapped_Origin','mapped_Destination'])
    a1 = np.array(df_node["mapped_Origin"])
    b1 = np.array(df_node["business_secondary_origin"])
    c1 = np.array(df_node["employees_secondary_origin"])
    d1 = np.array(df_node["business_tertiary_origin"])
    e1 = np.array(df_node["employees_tertiary_origin"])
    f1 = np.array(df_node["Night_Population_origin"])
    n = np.column_stack((b1,c1,d1,e1,f1))

    print(n)
    df_node
    n.shape


    from torch_geometric.data import Data
    edge_index = torch.tensor(ed, dtype=torch.long)
    x = torch.tensor((n), dtype=torch.float)


    data = Data(x=x, edge_index=edge_index)


    data.x.shape, data.edge_index.shape


    # Assume you have an adjacency matrix (edges), node features (features), and labels (labels)
    edges = torch.tensor(ed, dtype=torch.long).t()
    features = torch.randn(22735, 6)  # 4 nodes, each with 10 features
    labels = torch.randint(0, 2, (22735,), dtype=torch.long)  # Binary node classification labels

    # Calculate the number of nodes
    num_nodes = features.shape[0]
    num_nodes


    # In[24]:


    # Create the adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj_matrix[edges[0], edges[1]] = 1
    adj_matrix[edges[1], edges[0]] = 1


    # In[25]:


    # Set up train, validation, and test masks
    num_train = int(0.8 * num_nodes)
    num_val = int(0.1 * num_nodes)
    num_test = num_nodes - num_train - num_val


    # In[26]:


    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)


    # In[27]:


    # Set random nodes for training, validation, and testing
    perm = torch.randperm(num_nodes)
    train_mask[perm[:num_train]] = 1
    val_mask[perm[num_train:num_train + num_val]] = 1
    test_mask[perm[num_train + num_val:]] = 1


    # In[28]:


    # Create a simple graph neural network (GNN) model
    class SimpleGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleGNN, self).__init__()
            self.layer = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.out_layer = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, adj):
            x = self.layer(x)
            x = self.activation(x)
            x = torch.mm(adj, x)  # Assuming a simple graph convolution operation using the adjacency matrix
            x = self.out_layer(x)
            return x


    # In[29]:


    # Instantiate the model
    input_dim = features.shape[1]
    hidden_dim = 32
    output_dim = 2  # Binary classification task
    model = SimpleGNN(input_dim, hidden_dim, output_dim)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # In[31]:


    # Training loop
    import tqdm as tqdm
    losses = []
    num_epochs = 1000   
    epoch=0
    for tqdm in range(1000):
        optimizer.zero_grad()
        epoch = epoch + 1
        output = model(features, adj_matrix)  # Assume 'adj_matrix' is the adjacency matrix  
        loss = criterion(output[train_mask], labels[train_mask])
        print('training')
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # Print the loss during training
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')



    with torch.no_grad():
        model.eval()
        predictions = model(features, adj_matrix)
        predictions = predictions.argmax(dim=1)
        
        accuracy = (predictions[test_mask] == labels[test_mask]).float().mean().item()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")



    plt.plot(losses)

    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    # Save the plot as an image file (e.g., PNG)
    plt.savefig('loss_plot_99.png')



if __name__ == '__main__':
    gnn_training()







