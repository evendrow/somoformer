import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# import torch_dct as dct
from utils.dct import get_dct_matrix

class AuxilliaryEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(AuxilliaryEncoder, self).__init__(encoder_layer=encoder_layer,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        aux_output = []
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            aux_output.append(output)

        if self.norm is not None:
            output = self.norm(output)

        return output, aux_output

class LearnedDoublePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, num_joints=39, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        
        self.learned_encoding = nn.Embedding(num_joints, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        num_joints = x.size(0)//num_people
        half = x.size(2)//2

        x[:,:,0:half*2:2] = x[:,:,0:half*2:2] + self.learned_encoding(torch.arange(num_joints).repeat(num_people).to(self.device)).unsqueeze(1)
        x[:,:,1:half*2:2] = x[:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).repeat_interleave(num_joints, dim=0).to(self.device)).unsqueeze(1)
        return self.dropout(x)

class SoMoFormer(nn.Module):
    def __init__(self, tok_dim, nhid=256, nhead=8, dim_feedfwd=1024, nlayers=6, dropout=0.1, activation='relu', output_scale=1, location_method="grid", grid_len=3, grid_emb_size=8, normalize_inputs=True, seq_len=30, dct_n=20, residual_connection=False, num_joints=13, three_d_eembedding=False, id_joint_emb_size=0, learned_embedding=True, device='cuda:0'):
        super(SoMoFormer, self).__init__()

        self.nhid = nhid
        self.output_scale = output_scale
        self.location_method = location_method
        self.normalize_inputs = normalize_inputs
        self.dct_n = dct_n
        self.residual_connection = residual_connection
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.three_d_eembedding = three_d_eembedding
        self.grid_len = grid_len
        self.device = device

        dct_m, idct_m = get_dct_matrix(seq_len)
        self.dct = torch.from_numpy(dct_m[:dct_n]).float().to(device)
        self.idct = torch.from_numpy(idct_m[:, :dct_n]).float().to(device)

        if location_method == "naive":
            self.fc_in = nn.Linear(dct_n, nhid - id_joint_emb_size)
            self.double_id_encoder = LearnedDoublePositionalEncoding(nhid, dropout, num_joints=num_joints*3, device=device)

            self.id_encoder = PositionalEncoding(64, dropout)
            self.joint_encoder = PositionalEncoding(64, dropout)

        elif location_method == "neck":
            # self.fc_in = nn.Linear(dct_n+3, nhid)
            # self.double_id_encoder = DoublePositionalEncoding(nhid, dropout, num_joints=num_joints*3, learned=learned_embedding, device=device)
            self.fc_in = nn.Linear(dct_n, nhid-3)
            self.double_id_encoder = LearnedDoublePositionalEncoding(nhid-3, dropout, num_joints=num_joints*3, device=device)
        
        elif location_method == "grid":
            self.x_embed = nn.Embedding(grid_len, grid_emb_size)
            self.y_embed = nn.Embedding(grid_len, grid_emb_size)
#             self.grid_embed = nn.Embedding(grid_len*grid_len, grid_emb_size)
            self.fc_in = nn.Linear(dct_n, nhid - grid_emb_size - id_joint_emb_size)
            self.double_id_encoder = LearnedDoublePositionalEncoding(nhid-grid_emb_size, dropout, num_joints=num_joints*3, device=device)
            #self.double_id_encoder = LearnedPositionalEncoding(nhid-grid_emb_size, dropout)
        
            self.id_encoder = PositionalEncoding(64, dropout)
            self.joint_encoder = PositionalEncoding(64, dropout)
        
#         self.double_id_encoder = PositionalEncoding(nhid-3, dropout)
        
        #self.fc_in = nn.Linear(tok_dim, nhid)
        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)
        self.fc_out = nn.Linear(nhid, dct_n)
#         self.fc_out_aux = nn.ModuleList([nn.Linear(nhid, dct_n) for _ in range(nlayers)])
        
        if three_d_eembedding:
            self.fc_in = nn.Linear(dct_n*3, nhid-3)
            self.fc_out = nn.Linear(nhid, dct_n*3)
#             self.fc_out_aux = nn.ModuleList([nn.Linear(nhid, dct_n*3) for _ in range(nlayers)])
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.transformer = AuxilliaryEncoder(encoder_layer, num_layers=nlayers)
        
        self.m = torch.Tensor([ 0.0000, -0.5622,  0.0000,  0.0000, -0.5695,  0.0000,  0.0000, -0.8992,
                 0.0000,  0.0000, -0.9009,  0.0000,  0.0000, -1.3000,  0.0000,  0.0000,
                -1.2956,  0.0000,  0.0000,  0.0017,  0.0000,  0.0000, -0.0681,  0.0000,
                0.0000, -0.0691,  0.0000,  0.0000, -0.2849,  0.0000,  0.0000, -0.2850,
                0.0000,  0.0000, -0.4265,  0.0000,  0.0000, -0.4109,  0.0000]).to(device)
        self.s = torch.Tensor([0.4289, 0.1349, 0.4289, 0.4286, 0.1340, 0.4286, 0.4425, 0.1974, 0.4425,
                0.4409, 0.1984, 0.4409, 0.4587, 0.2031, 0.4587, 0.4537, 0.2088, 0.4537,
                0.4200, 0.1238, 0.4200, 0.4410, 0.1253, 0.4410, 0.4381, 0.1261, 0.4381,
                0.4640, 0.1400, 0.4640, 0.4635, 0.1439, 0.4635, 0.4836, 0.2115, 0.4836,
                0.4835, 0.2315, 0.4835]).to(device) * 5.
        
        self.dct_m = torch.Tensor([-7.1789e-02,  4.1777e-02,  2.0464e-02,  4.6075e-03,  6.3129e-04,
             2.2184e-03,  2.2500e-03,  6.2809e-04,  1.6267e-04,  7.9969e-04,
             7.8669e-04,  1.4957e-04,  5.4564e-05,  4.1828e-04,  3.6941e-04,
             1.3337e-05,  2.7676e-05,  2.8088e-04,  2.1341e-04, -3.4786e-05,
             5.6285e-06,  1.8702e-04,  1.1101e-04, -6.3937e-05,  4.1189e-06,
             1.4035e-04,  4.3991e-05, -9.2744e-05,  4.7437e-07,  1.1380e-04]).to(device)[:dct_n]
        self.dct_s = torch.Tensor([0.5954, 0.5319, 0.2537, 0.0897, 0.0567, 0.0498, 0.0459, 0.0383, 0.0321,
            0.0262, 0.0189, 0.0091, 0.0072, 0.0085, 0.0073, 0.0042, 0.0036, 0.0050,
            0.0039, 0.0027, 0.0022, 0.0036, 0.0024, 0.0025, 0.0017, 0.0029, 0.0016,
            0.0024, 0.0015, 0.0026]).to(device)[:dct_n]
        

        if num_joints == 14:
            print('Using Posetrack DCT')
            self.dct_m = torch.Tensor([ 5.3243e-02, -6.0472e-03, -7.2297e-02, -4.9128e-03,  8.1936e-03,
                1.8048e-03, -8.1131e-03, -1.4365e-03,  9.1773e-04,  1.1104e-03,
                -3.1621e-03, -7.2587e-04,  4.7325e-04,  5.8135e-04, -3.9191e-04,
                3.4860e-04, -8.1090e-04, -3.2206e-04, -2.9175e-04, -1.5268e-04,
                -5.1605e-04, -5.2221e-04,  1.3108e-06,  1.0118e-03, -1.5232e-04,
                -8.4104e-04,  2.7592e-05,  7.2812e-04, -4.4474e-04, -1.5173e-04])
            self.dct_s = torch.Tensor([2.0158, 0.4596, 0.2984, 0.1957, 0.1623, 0.1263, 0.1122, 0.0936, 0.0856,
                0.0736, 0.0692, 0.0625, 0.0587, 0.0537, 0.0503, 0.0488, 0.0453, 0.0431,
                0.0427, 0.0407, 0.0391, 0.0375, 0.0370, 0.0357, 0.0369, 0.0347, 0.0352,
                0.0334, 0.0356, 0.0331])

#         self.output_scale = torch.nn.Parameter(torch.ones(1)*output_scale)

    
    def dct_forward(self, x):
        """
        Performs DCT transform.
        x: torch.Tensor of shape (F, ...)
        """
        tgt_dct = self.dct @ x.reshape(self.dct.shape[1], -1)               # (dct_in, B*N*J*K)
        tgt_dct = tgt_dct.reshape(-1, *x.shape[1:]).permute(2, 1, 0)        # (N*J*K, B, dct_in)
        
        tgt_dct = (tgt_dct - self.dct_m[:self.dct_n])/self.dct_s[:self.dct_n] # (N*J*K, B, dct_in)
        # tgt_dct = tgt_dct.sign() * tgt_dct.abs().pow(1/2)
        return tgt_dct
        
    def dct_backward(self, x):
        """
        x: torch.Tensor of shape # (K, B, F)
        """
        x = (x * self.dct_s[:self.dct_n]) + self.dct_m[:self.dct_n]
        # x = x.sign() * x.pow(2)

        x = x.transpose(0, 2)
        
        out_dct = self.idct @ x.flatten(1)           # (F, B*K)
        out_dct = out_dct.reshape(-1, *x.shape[1:])  # (F, B, K)
        return out_dct
    
    def forward(self, tgt, tgt_neck, padding_mask, metamask=False):
        """"
        tgt: torch.Tensor of shape (B, in_F, N, J, K)
        tgt_neck: torch.Tensor of shape (B, N, K), representing the position of the neck joint at the first frame
        padding_mask: torch.Tensor of shape (B, N), describing which people are valid or padded
        """
        
        B, in_F, NJ, K = tgt.shape
        F = self.seq_len
        J = self.num_joints
        out_F = F - in_F
        N = NJ // J
        
        num_keys = J if self.three_d_eembedding else J*K
        
        # Pad joints with copy of last pose
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]

        # Or pad with velocity:
        #         velocity = (in_joints_flat[-1:] - in_joints_flat[-2:-1]).repeat(out_F, 1, 1)
        #         new_joints_flat = in_joints_flat[-1:] + velocity*torch.arange(1,out_F+1).reshape(out_F, 1, 1).to(velocity)
        #         padded_joints_flat = torch.cat((in_joints_flat, new_joints_flat), dim=0)
        
        # # Create neck position, to concatenate with model input
        # necks_concat = tgt_neck.reshape(B, N, K).transpose(0, 1) # (N, B, K)
        # necks_concat = necks_concat - necks_concat.mean(0).unsqueeze(0)
        # necks_concat = necks_concat.repeat_interleave(J*K, dim=0) # (N*J*K, B, K)
        

        tgt = tgt.flatten(-2).transpose(0, 1) # (F, B, N*J*K)
        

        if self.normalize_inputs:
            tgt = (tgt-self.m.repeat(N))/self.s.repeat(N)

        if self.location_method == "naive":
            naive_pos_concat = tgt[0].permute(1, 0).unsqueeze(0)
        elif self.location_method == "grid":
            # necks_concat = tgt_neck.reshape(B, N, K).transpose(0, 1)[:, :, :2] # (N, B, 2) Getting the xy coords only
            # necks_concat = (necks_concat - necks_concat.mean(0))/necks_concat.std(0, unbiased=False)
            # pose_grid = torch.clamp(torch.floor(self.grid_len * (necks_concat + 1) / 2), 0, self.grid_len - 1)
            # grid_emb = self.x_embed(pose_grid[:, :, 0].long()) + self.y_embed(pose_grid[:, :, 1].long()) # (N, B, g_e)
            # grid_emb = grid_emb.repeat_interleave(J*K, dim=0) # (N*J*K, B, g_e)
            necks_concat = tgt_neck.reshape(B, N, K)[:, :, [0,2]] # (B, N, 2) Getting the xy coords only
            # necks_concat = (necks_concat - necks_concat.amin(0)) / (necks_concat.amax(0) - necks_concat.amin(0) + 1e-6) # (B, N, 2)
            necks_concat = (torch.clamp(necks_concat, -3, 3) + 3) / 6
            pose_grid = (necks_concat*self.grid_len).div(self.grid_len, rounding_mode='floor').long() # (B, N, 2)
#             pose_grid_idx = pose_grid[:,:,0]*self.grid_len + pose_grid[:,:,1] # (B, N)
#             grid_emb = self.grid_embed(pose_grid_idx) # (B, N, g_e)
            grid_emb = self.x_embed(pose_grid[:,:,0].long()) + self.y_embed(pose_grid[:,:,1].long())
            grid_emb = grid_emb.transpose(0, 1).repeat_interleave(J*K, dim=0) # (N*J*K, B, g_e)
            


        tgt_dct = self.dct_forward(tgt) # (N*J*K, B, dct_in)
#         if self.three_d_eembedding:
#             # (N*J*K, B, dct_in) -> (F, B, N*J*K) -> (F, B, N*J, K) -> (F, K, B, N*J) -> (F*K, B, N*J)
#             # (N*J*K, B, F) -> (N*J, K, B, F) -> (N*J, B, F, K) -> (N*J, B, F*K)
#             tgt_dct = tgt_dct.reshape(-1, K, B, F).permute(0, 2, 3, 1).reshape(-1, B, K*F)
# #             tgt_dct = tgt_dct.reshape(F, B, -1, K).permute(0, 3, 1, 2).reshape(F*K, B, -1) 
        
        if metamask:
            NJK = tgt_dct.shape[0]
            mjm_mask = np.ones((NJK,1))

            mjm_percent = 0.05
            #pb = np.random.random_sample()
            #mjm_percent *= pb
            #masked_num = int(pb * mjm_percent * NJK) # at most x% of the vertices could be masked
            #indices = np.random.choice(np.arange(NJK),replace=False,size=masked_num)
            #mjm_mask[indices,:] = 0.0
            mjm_mask = (torch.rand((NJK, B, 1)).float().to(self.device) > mjm_percent).float()
        
            meta_masks = mjm_mask #torch.from_numpy(mjm_mask).float().cuda()
            constant_tensor = torch.ones_like(tgt_dct).float().to(self.device)*1.0
            tgt_dct  = tgt_dct*meta_masks + tgt_dct*constant_tensor*(1-meta_masks)



        ############
        # Transformer
        ###########

        tgt = tgt_dct #(N*J*K, B, dct_in)

        
        
        # if self.location_method == "neck":
        #     #print(tgt_neck.shape)
        #     #print(tgt.shape)
        #     necks = tgt_neck.reshape(B, N, K).transpose(0, 1).repeat_interleave(J*K, dim=0) # (N*J*K, B, K)
        #     tgt = torch.cat((tgt_dct, necks), dim=2)

#         print(tgt.shape)
        tgt = self.fc_in(tgt) #  * self.scale #(N*J*K, B, h_dim-1)
        tgt = self.double_id_encoder(tgt, num_people=N)
        
        #id_emb = self.id_encoder(torch.zeros(N, B, 64).to(self.device)).repeat_interleave(J*K, dim=0) # (N*J*K, B, 64)
        #joint_emb = self.joint_encoder(torch.zeros(J*K, B, 64).to(self.device)).repeat((N, 1, 1)) # (N*J*K, B, 64)
        #tgt = torch.cat((tgt, id_emb, joint_emb), dim=-1)
        
        if self.location_method == "naive":
            pass
#             tgt = torch.cat((tgt, naive_pos_concat), dim=2) # (N*J*K, B, h_dim)
        elif self.location_method == "grid":
            tgt = torch.cat((tgt, grid_emb), dim=2) # (N*J*K, B, h_dim)
        elif self.location_method == "neck":
            necks = tgt_neck.reshape(B, N, K).transpose(0, 1).repeat_interleave(J*K, dim=0) # (N*J*K, B, K)
            tgt = torch.cat((tgt, necks), dim=2)



        tgt_padding_mask = padding_mask.repeat_interleave(num_keys, dim=1) #(B, N*J*K)
        out, aux_out = self.transformer(tgt, mask=None,
                                        src_key_padding_mask=tgt_padding_mask)

        out = self.fc_out(out)
        aux_out = [self.fc_out(aux) for aux in aux_out]

        ############
        # Residual connection
        ###########
        if self.residual_connection:
            out = out * self.output_scale + tgt_dct
            aux_out = [aux * self.output_scale + tgt_dct for aux in aux_out]

        #############
        # Conversion from dct to 3d coordinates
        #############
        if self.three_d_eembedding:
            # (N*J, B, F*K) -> (N*J, B, F, K) -> (N*J, K, B, F) -> (N*J*K, B, F)
            out = out.reshape(-1, B, F, K).permute(0, 3, 1, 2).reshape(-1, B, F)
            aux_out = [aux.reshape(-1, B, F, K).permute(0, 3, 1, 2).reshape(-1, B, F) for aux in aux_out]
            
        out = self.dct_backward(out)
        aux_out = [self.dct_backward(aux) for aux in aux_out]

        ############
        # Normalize + output reshaping
        ###########
        if self.normalize_inputs:
            out = (out*self.s.repeat(N))+self.m.repeat(N)
            aux_out = [(aux*self.s.repeat(N))+self.m.repeat(N) for aux in aux_out]
        
        out = out.transpose(0, 1).reshape(B, F, NJ, K)
        aux_out = [aux.transpose(0, 1).reshape(B, F, NJ, K) for aux in aux_out]
        
        return out, aux_out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        
        self.register_buffer('pe', pe)
#         self.register_parameter('pe', torch.nn.Parameter(pe))
#         self.pe = pe.to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
#         x = x + self.pe[:x.size(0)//num_people].repeat(num_people, 1, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        
        self.learned_encoding = nn.Embedding(max_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        num_joints = x.size(0)//num_people
#         half = x.size(2)//2
        
#         print(x.shape)
#         print(x[:,:,0:half*2:2].shape)
#         print(torch.arange(num_joints).repeat(num_people).shape)
#         print(self.learned_encoding(torch.arange(num_joints).repeat(num_people).to(self.device)).unsqueeze(1).shape)
#         x[:,:,0:half*2:2] = x[:,:,0:half*2:2] + self.learned_encoding(torch.arange(num_joints).repeat(num_people).to(self.device)).unsqueeze(1)
#         x[:,:,1:half*2:2] = x[:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).repeat_interleave(num_joints, dim=0).to(self.device)).unsqueeze(1)
        x = x + self.learned_encoding(torch.arange(x.size(0)).to(self.device)).unsqueeze(1)
        return self.dropout(x)
    


class CustomDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(CustomDecoder, self).__init__(decoder_layer=decoder_layer,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt

        aux_output = []

        for i, mod in enumerate(self.layers):
            output =  mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            aux_output.append(output)

        if self.norm is not None:
            output = self.norm(output)

        #aux_output = torch.stack(aux_output)
        return output, aux_output


def create_model(config, logger):
    num_kps = config["MODEL"]["num_kps"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers=config["MODEL"]["num_layers"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]

    if config["MODEL"]["type"] == "somoformer":
        logger.info("Creating bert model.")
        model = SoMoFormer(num_kps*3,
            nhid=nhid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers=nlayers,
            output_scale=config["MODEL"]["output_scale"],
            normalize_inputs=config["MODEL"]["normalize_inputs"],
            seq_len=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            dct_n=config["MODEL"]["dct_n"],
            residual_connection=config["MODEL"]["residual_connection"],
            num_joints=num_kps,
            location_method=config["MODEL"]["location_method"],
            grid_len=config["MODEL"]["grid_len"],
            grid_emb_size=config["MODEL"]["grid_emb_size"],
            learned_embedding=config["MODEL"]["learned_embedding"],
            device=config["DEVICE"]
        ).to(config["DEVICE"]).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model

