## Utils script is designed for easy use of saving and loading the models

import torch

# Saves the model
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Loads the model
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval() # Also provides Evaluation
    return model
