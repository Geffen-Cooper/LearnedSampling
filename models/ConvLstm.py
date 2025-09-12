'''
Code taken from https://github.com/teco-kit/ISWC22-HAR/blob/main/models/deepconvlstm.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1),padding=(1,0))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), padding=(1,0))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)

        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)

        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                 filter_scaling_factor,
                 nb_conv_blocks         = 2,
                 nb_filters             = 64,
                 dilation               = 1,
                 batch_norm             = False,
                 filter_width           = 5,
                 nb_layers_lstm         = 1,
                 drop_prob              = 0.5,
                 nb_units_lstm          = 128,
                 t_context              = None):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
        
        """
        super(DeepConvLSTM, self).__init__()
        self.nb_conv_blocks = nb_conv_blocks
        self.nb_filters     = int(filter_scaling_factor*nb_filters)
        self.dilation       = dilation
        self.batch_norm     = bool(batch_norm)
        self.filter_width   = filter_width
        self.nb_layers_lstm = nb_layers_lstm
        self.drop_prob      = drop_prob
        self.nb_units_lstm  = int(filter_scaling_factor*nb_units_lstm)
        
        
        self.nb_channels    = input_shape[3]
        self.nb_classes     = nb_classes

    
        self.conv_blocks = []

        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = input_shape[1]
            else:
                input_filters = self.nb_filters
    
            self.conv_blocks.append(ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))

        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        
        # define lstm layers
        self.lstm_layers = []
        for i in range(self.nb_layers_lstm):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm, batch_first =True))
            else:
                self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm, batch_first =True))
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define classifier
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

        self.num_sensors, self.window_size = t_context
        self.t_context_layer = nn.Linear(self.num_sensors,self.num_sensors*3*int(self.window_size))

    def forward(self, x, ages=None):
        # reshape data for convolutions
        # B,L,C = x.shape
        # x = x.view(B, 1, L, C)
        x = x.unsqueeze(1)

        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)

        if ages is not None:
            temporal_context = F.sigmoid(self.t_context_layer(ages))
            # (B,1) -> (B,3*6) -> (B,6,3) -> (B,C,6,3) -> (B,C,6,15)
            # contexts = []
            # ages = F.normalize(ages) # along dimension 1
            # for i,context_layer in enumerate(self.context_layers):
            #     out = F.sigmoid(context_layer(ages[:,i].unsqueeze(0).T))
            #     out = out.view(x.shape[0],x.shape[2],-1).unsqueeze(1).repeat(1,x.shape[1],1,1)
            #     contexts.append(out)
            # contexts = torch.cat(contexts,axis=3)
            # x = x * contexts
            temporal_context = temporal_context.view(x.shape[0],self.window_size,-1).unsqueeze(1).repeat(1,x.shape[1],1,1)
            x = x * temporal_context

        final_seq_len = x.shape[2]

        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)

        x = self.dropout(x)
        

        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            

        x = x[:, -1, :]
    
        x = self.fc(x)


        return x
    

if __name__ == "__main__":
    import yaml
    from LearnedSampling.experiments.train_classifier import get_args
    
    config_file = open('model_configs.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)["convlstm"]

    # synthetic HAR batch (batch size, window size, channels)
    args = get_args()
    args = vars(args)
    num_channels = 3*len(args['body_parts'])*len(args['sensors'])
    data_synthetic = torch.randn((args['batch_size'], args['window_size'], num_channels)).cuda()
    ages_synthetic = torch.randn((args['batch_size'], len(args['body_parts']))).cuda()

    # create HAR model
    model = DeepConvLSTM(input_shape=(1,1,args['window_size'],num_channels),nb_classes=len(args['activities']),**config,t_context=(len(args['body_parts']),args['window_size'])).cuda()
    model.eval()
    with torch.no_grad():
        logits = model(data_synthetic, ages_synthetic)
        print(f"\t input: {data_synthetic.shape} {data_synthetic.dtype}")
        # print(f"\t z: {z.shape} {z.dtype}")
        print(f"\t logits: {logits.shape} {logits.dtype}")

        print(f"Num Params:{sum(p.numel() for p in model.parameters())}")

    for name,param in model.named_parameters():
        print(name)