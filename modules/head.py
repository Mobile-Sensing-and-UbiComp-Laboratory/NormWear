import torch
import torch.nn as nn

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high   
        # self.low, self.high = ranges        
    def forward(self, x):                    
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low

class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout=0.1, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)
        
    def forward(self, x):
        """
        x: [bs x nvars x num_patch x d_model]
        output: [bs x output_dim]
        """
        x = x[:,:,0,:]             # only consider the cls_token, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)         # 128 x 3072
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y.squeeze()


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x num_patch x d_model]
        output: [bs x n_classes]
        """
        x = x[:,:,0,:]             # only consider the cls_token, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        
        return y
    