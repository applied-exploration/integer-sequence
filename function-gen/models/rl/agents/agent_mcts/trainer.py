import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, Policy, learning_rate=0.1):

        self.step_model = Policy()

        value_criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.step_model.parameters(),
                                    lr=learning_rate)

        def train(obs, search_pis, returns):
            obs = torch.from_numpy(obs)
            search_pis = torch.from_numpy(search_pis)
            returns = torch.from_numpy(returns)

            optimizer.zero_grad()
            logits, policy, value = self.step_model(obs)

            logsoftmax = nn.LogSoftmax(dim=1)
            
            logits_to_logprobs = logsoftmax(logits)
            logits_search_pis_multiplied = -search_pis * logits_to_logprobs
            
            # print("===================")
            # print("search_pis")
            # print(search_pis)
            # print(torch.sum(search_pis, dim=1))
            # print("logits")
            # print(logits)
            # print("policy")
            # print(policy)
            # print(torch.sum(policy, dim=1))
            
            # print("logits_to_logprobs")
            # print(logits_to_logprobs)
            # print("logits_search_pis_multiplied")
            # print(logits_search_pis_multiplied)
            
            
            # cross_entropy_loss = nn.CrossEntropyLoss()
            mse_loss = nn.MSELoss()
            
            policy_loss = mse_loss(policy, search_pis)     
            # print(policy_loss)
            # policy_loss = cross_entropy_loss(policy, search_pis)     
            # policy_loss = 5*torch.mean(torch.sum(logits_search_pis_multiplied, dim=1)) ''' This was the original '''
            value_loss = value_criterion(value, returns)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            return value_loss.data.numpy(), policy_loss.data.numpy()

        self.train = train