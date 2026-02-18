import torch
import torch.nn as nn
import torch.nn.functional as F

class TetrisNet(nn.Module):
    def __init__(self, input_channels=4, num_actions=40, global_feature_dim=5):
        super(TetrisNet, self).__init__()
        
        # shared convolutional network
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        # Flattened size is 2 * 20 * 10 = 400
        self.policy_fc = nn.Linear(400 + global_feature_dim, num_actions)

        # value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        # Flattened size is 1 * 20 * 10 = 200
        self.value_fc1 = nn.Linear(200 + global_feature_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board_tensors, global_features):
        x = F.relu(self.bn1(self.conv1(board_tensors)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = torch.cat([p, global_features], dim=1) # Inject global stats
        p = self.policy_fc(p)
        # We return logits (Log-Softmax is better for training stability)
        p = F.log_softmax(p, dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten
        v = torch.cat([v, global_features], dim=1) # Inject global stats
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Output between -1 and 1
        
        return p, v

    def export_to_onnx(self, filepath="model.onnx"):
        """Exports the model for Rust/ONNXRuntime consumption."""
        self.eval()
        dummy_board = torch.randn(1, 4, 20, 10)
        dummy_global = torch.randn(1, 5)
        torch.onnx.export(self, (dummy_board, dummy_global), filepath, 
                         input_names=['board', 'global'],
                         output_names=['policy', 'value'],
                         dynamic_axes={'board': {0: 'batch_size'}, 'global': {0: 'batch_size'}})