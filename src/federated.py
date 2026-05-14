import torch
import copy

class FederatedServer:
    @staticmethod
    def aggregate_weights(local_models):
        """
        Implements FedAvg: Computes the mean of weights from all local agents.
        """
        global_model = copy.deepcopy(local_models[0])
        global_dict = global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.stack([
                m.state_dict()[key].float() for m in local_models
            ], 0).mean(0)
            
        global_model.load_state_dict(global_dict)
        return global_model