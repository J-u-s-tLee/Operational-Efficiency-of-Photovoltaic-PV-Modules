import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedFeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super(SharedFeedForwardNN, self).__init__()
        
        self.shared_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.shared_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.regression_output = nn.Linear(hidden_dim, 1)
        self.classification_output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):

        x = (self.shared_layer_1(x))
        x = torch.relu(self.shared_layer_2(x))
        x = self.dropout(x)
        
        reg_output = self.regression_output(x)
        
        class_output = self.classification_output(x)

        return reg_output, class_output
