"""Online learner for incremental CLM."""
import torch


class OnlineLearner:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def update(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model(batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss.backward()
        self.optimizer.step()
        return loss.item()
