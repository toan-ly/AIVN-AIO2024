import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_hidden_layers = config.get('num_hidden_layers', 2)
        self.batch_norm_enabled = config.get('batch_norm', False)
        self.skip_connection_enabled = config.get('skip_connection', False)
        self.skip_interval = config.get('skip_interval', 2)

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        if self.batch_norm_enabled:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(self.num_hidden_layers)]
            )
            
        self.activation = self._get_activation_function(config.get('activation', 'sigmoid'))

        if config.get('custom_init', False):
            self._init_weights(config.get('std', 1.0))
        # self.layers = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_dim, output_dim)
        # )
    
    def _get_activation_function(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f'Activation function {activation} not supported')

    def _init_weights(self, std):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.activation(x)
        
        skip = x if self.skip_connection_enabled else None
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)

            if self.config.get('batch_norm', False):
                x = self.batch_norms[i](x)

            x = self.activation(x)

            if self.skip_connection_enabled and skip and (i + 1) % self.skip_interval == 0:
                x += skip
                skip = x
                
        x = self.output_layer(x)
        return x
