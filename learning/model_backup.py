import torch
import torch.nn as nn
import torch.nn.functional as F

class CamScoreNet(nn.Module):
    def __init__(self):
        super(CamScoreNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size = (6,8,12), stride = 2)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(64, 64, kernel_size = (6,8,12), stride = 2)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        self.fc4 = nn.Linear(3456, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)

    def forward(self, x, y):
        out1 = F.relu(self.conv1(x))
        out2 = self.bn1(out1)
        out3 = self.pool1(out2)

        print (out3.shape)

        out4 = F.relu(self.conv2(out3))
        out5 = self.bn2(out4)
        out_x = self.pool2(out5)

        print (out_x.shape)

        out6 = F.relu(self.fc1(y))
        out7 = F.relu(self.fc2(out6))
        out_y = F.relu(self.fc3(out7))

        #print (out_x.shape)

        out_x = out_x.view(-1, 64*5*5*2)
        out_cat = torch.cat((out_x, out_y), 1)
        
        out7 = F.relu(self.fc4(out_cat))
        out8 = F.relu(self.fc5(out7))
        out9 = F.relu(self.fc6(out8))
        out_final = torch.sigmoid(self.fc7(out9))

        return out_final


if __name__ == '__main__':
    model = CamScoreNet()
    rand_input_x = torch.randn(4, 1, 101, 121, 86)
    rand_input_y = torch.randn(4, 7)
    output = model(rand_input_x, rand_input_y)
    print (output)

