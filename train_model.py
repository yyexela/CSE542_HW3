import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = torch.nn.MSELoss()

def train_single(num_epochs, num_batches,batch_size, model, optimizer, replay_buffer):
    for epoch in range(num_epochs):

        for i in range(num_batches):
            optimizer.zero_grad()
            t1_observations, t1_actions, _, t1_next_observations, _ = replay_buffer.sample(batch_size)
            oa_in = torch.cat([t1_observations, t1_actions], dim=-1)

            next_o_pred = model(oa_in)
            loss = loss_fn(next_o_pred, t1_next_observations)

            loss.backward()
            optimizer.step()


def train_model(model, replay_buffer, optimizer, num_epochs=500, batch_size=32):
    """
    Train a single model with supervised learning
    """
    idxs = np.array(range(len(replay_buffer)))
    num_batches = len(idxs) // batch_size
    if not isinstance(model, list):
        train_single(num_epochs, num_batches, batch_size, model, optimizer, replay_buffer)
    else:
        # START-Ensemble models
        batch_sizes = np.linspace(250,500,len(model), dtype=int)
        print("Batching")
        # Hint1: try different batch size for each model
        # hint2: check out how we define optimizer and model for ensemble models. During training, each model should have their individual optimizer and batch size to increase diversity.
        for i, _ in enumerate(model):
            train_single(num_epochs, num_batches, batch_sizes[i], model[i], optimizer[i], replay_buffer)
        # END
