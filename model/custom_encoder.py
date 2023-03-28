import torch
import torch.nn as nn
import d3rply

class CustomEncoder(nn.Module):
    def __init__(self, nb_layers=1, nb_hidden_units=64,use_gpu=True,feature_size,observation_shape):
        super(CustomEncoder, self).__init__()
        self.use_gpu = use_gpu
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_hidden_units
        self.embedding_dim = observation_shape[2]
        self.feature_size = feature_size
        self.rnn=nn.rnn(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers=self.nb_layers
        )
        self.conv1 = nn.Conv1d(self.nb_hidden_units,32,3,1,1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64, feature_size)

        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       
    def init_hidden(self):
        hidden_h = torch.randn(self.nb_layers,self.hidden_units)
        if self.use_gpu:
            hidden_h = hidden_h.cuda()
        return hidden_h
    def forward(self, x):
        hidden = self.init_hidden()
        x,hidden = self.rnn(x,hidden)
        x = x.contiguous()
        x = x.view(x.size(0),-1)
        x = x.permute(0,1)
        x = self.conv1(x)
        x = self.relu1(x)
        x,_ = torch.max(x,1)
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.sigmoid(x)
        return x

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size

class CustomEncoderWithAction(nn.Module):
    def __init__(self, nb_layers=1, nb_hidden_units=64,use_gpu=True,feature_size,observation_shape,action_size):
        super(CustomEncoder, self).__init__()
        self.use_gpu = use_gpu
        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_hidden_units
        self.embedding_dim = observation_shape[2]
        self.feature_size = feature_size
        self.rnn=nn.rnn(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers=self.nb_layers
        )
        self.conv1 = nn.Conv1d(self.nb_hidden_units,32,3,1,1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(64+action_size,64)
        self.fc2 = nn.Linear(64, feature_size)

        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
       
    def init_hidden(self):
        hidden_h = torch.randn(self.nb_layers,self.hidden_units)
        if self.use_gpu:
            hidden_h = hidden_h.cuda()
        return hidden_h
    def forward(self, x):
        hidden = self.init_hidden()
        x,hidden = self.rnn(x,hidden)
        x = x.contiguous()
        x = x.view(x.size(0),-1)
        x = x.permute(0,1)
        x = self.conv1(x)
        x = self.relu1(x)
        x,_ = torch.max(x,1)
        x = torch.cat([x,action],dim=1)
        x = torch.fc1(x)
        x = torch.fc2(x)
       # x = self.sigmoid(x)
        return x

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.feature_size

class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"  # this is necessary

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return CustomEncoder(feature_size=self.feature_size,observation_shape = observation_shape)

    def create_with_action(self, observation_shape, action_size, discrete_action):
        return CustomEncoderWithAction(feature_size=self.feature_size,observation_shape=observation_shape,action_size=action_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
