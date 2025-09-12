import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn as rnn_utils

class VRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=20, num_layers=1, dropout_prob=0.5, skip_options=4):
        super(VRNN, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer before classification
        self.fc = nn.Linear(hidden_size, output_size)

        # takes in hidden state, current time, remaining budget
        self.policy_head = nn.Linear(hidden_size+2, skip_options)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.skip_options = skip_options
        # 0 -> 1, 1 -> 2, 2 -> 4, 3 -> 8, 4 -> 16
        self.skip_map = {i: 2**i for i in range(skip_options)}


    # default forward used to train classifier
    def forward(self, x_padded, lengths):
        # normal classifier forward pass
        packed = rnn_utils.pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.rnn(packed)
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        out = torch.stack([out[i, l - 1] for i, l in enumerate(lengths)])  # Final timestep per sequence
        out = self.dropout(out)  # Apply dropout before classification

        return self.fc(out)
    
    # special forward used to train policy and finetune classifier
    def other_forward(self, x=None, h=None, c=None, policy_input=None, budget=None, lengths=None, mode=None):
        # used to get intermediate hidden states, assume batch=1
        # we typically use this to go one step forward
        if mode == "IterateForward":
            out, (h_n,c_n) = self.rnn(x, (h,c))
            return h_n,c_n
        # used to train the policy
        elif mode == "PolicyForward":
            return self.policy_head(policy_input)
        elif mode == "SimpleForward":
            out, (h_n,c_n) = self.rnn(x)
            do = self.dropout(h_n)  # Apply dropout before classification
            return self.fc(do)
        elif mode == "SkipForward":
            # for each sample in batch, roll out the policy and store trajectories
            preds = []
            for x_indv,length in zip(x,lengths):
                x_indv = x_indv[:length,:] # unpad it
                h_i = torch.zeros((1,1,self.hidden_size)) # batch size 1, one layer
                c_i = torch.zeros((1,1,self.hidden_size)) # batch size 1, one layer
                idx = 0
                seq_len = x_indv.shape[0]
                x_traj = []
                rem_budget = budget
                # print("==")
                delay = seq_len // budget
                skips = []
                # skip = 0
                while rem_budget > 0:
                    # policy forward pass, map this to a skip amount
                    p_in = torch.cat([h_i[0][0],torch.tensor([idx]),torch.tensor([rem_budget])])
                    skip_pred = self.policy_head(p_in.unsqueeze(0))
                    skip = self.skip_map[torch.argmax(skip_pred).item()]
                   
                    # skip = delay-1
                    # print(skip_pred)
                    skips.append(skip)
                    # observe input at this skipped value
                    idx += skip
                    # print(budget,idx-skip,skip)
                    if idx >= seq_len:
                        break
                    # add observed sample and time step to trajectory
                    x_new = torch.cat([torch.tensor([idx]),x_indv[idx,:]])
                    # print(x_new)
                    x_traj.append(x_new)

                    # update the hidden state with next observation
                    out, (h_i,c_i) = self.rnn(x_new.unsqueeze(0).unsqueeze(0), (h_i,c_i))
                    # print(rem_budget,idx)
                    rem_budget -= 1
                    # skip = self.skip_map[torch.argmax(skip_pred).item()]
                    # skip = delay
                # print(skips)
                # do = self.dropout(h_i)  # Apply dropout before classification
                preds.append(self.fc(h_i))
            preds = torch.cat(preds,dim=1)[0] # merge back into a batch
            # print(torch.argmax(preds,dim=1))
            # exit()
            
            return preds


if __name__ == "__main__":
    import yaml
    from LearnedSampling.experiments.train_classifier import get_args
    
    config_file = open('model_configs.yaml', mode='r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)["vrnn"]

    # synthetic HAR batch (batch size, window size, channels)
    args = get_args()
    args = vars(args)
    num_channels = 3
    print(num_channels)

    sequences = []
    lengths = []
    labels = []

    import random
    from torch.nn.utils.rnn import pad_sequence
    for _ in range(32):
        seq_len = random.randint(10, 20)
        sequence = torch.randn(seq_len,3)  # (n_i, 3)

        sequences.append(sequence)
        lengths.append(seq_len)

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True).cuda()  # (batch_size, max_len, 3)
    lengths = torch.tensor(lengths)#.cuda()

    # create HAR model
    # model = VRNN(input_shape=(1,1,args['window_size'],num_channels),nb_classes=len(args['activities']),**config,t_context=(len(args['body_parts']),args['window_size'])).cuda()
    model = VRNN(input_size=num_channels,**config, output_size=20, num_layers=1).cuda()
    model.eval()
    with torch.no_grad():
        print(f"\t input: {padded.shape} {padded.dtype}")
        logits = model(padded, lengths)
        
        # print(f"\t z: {z.shape} {z.dtype}")
        print(f"\t logits: {logits.shape} {logits.dtype}")

        print(f"Num Params:{sum(p.numel() for p in model.parameters())}")

    for name,param in model.named_parameters():
        print(name)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))