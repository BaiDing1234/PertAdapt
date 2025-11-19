import torch
from torch import nn


class LayerNorm(nn.Module):
    ## Adapted from gRNAde's code repository: https://github.com/chaitjo/geometric-rna-design/blob/main/src/layers.py#L430
    """
    For GVP-GNN
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class ProteinInverseFoldingPredictionHead(nn.Module):
    def __init__(
        self,
        c_in,
        node_h_dim,
        embed_dim,
        num_blocks=1,
        num_attn_heads=8,
        attn_dropout=0.1,
        add_pos_enc_seq=True,
        add_pos_enc_str=False,
    ):
        super().__init__()

        self.linear_in_seq = nn.Linear(c_in, embed_dim)
        self.linear_in_str = nn.Linear(node_h_dim, embed_dim)
        # self.linear_in_str = torch.nn.Sequential(    ### NOTE: we would need this to preserve geometric details as well.
        #     LayerNorm(node_h_dim),
        #     # GVP(node_h_dim, node_h_dim, activations=(None, None), vector_gate=True),
        #     GVP(node_h_dim, (embed_dim, 0), activations=(None, None)),
        # )
        # raise Exception('Maybe too many layers in the structure projection.')

        self.multihead_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim * 2,
                    num_heads=num_attn_heads,
                    batch_first=True,
                    dropout=attn_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.Dropout(attn_dropout),
                    nn.ELU(inplace=True),
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.Dropout(attn_dropout),
                    nn.ELU(inplace=True),
                    nn.Linear(embed_dim // 2, embed_dim),
                    nn.Dropout(attn_dropout),
                    nn.ELU(inplace=True),
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.Dropout(attn_dropout),
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(
            embed_dim * 2, 44
        )  ## index=0,1,2,3 ==> AGCU tokens ; index=4, ==> [MASK] token.

        self.add_pos_enc_seq = add_pos_enc_seq
        self.add_pos_enc_str = add_pos_enc_str
        self.num_attn_heads = num_attn_heads
        self.num_blocks = num_blocks
        self.node_h_dim = node_h_dim
        self.c_in = c_in
        self.embed_dim = embed_dim

    def forward(self, data):
        x_seq = data["lm_representation"]
        # x_str = (data['structure_encoding'], data['structure_encoding_vector'])
        x_str = data["structure_encoding"]
        x_seq = self.linear_in_seq(x_seq)  # NOTE: query (N x L x E_q)
        x_str = self.linear_in_str(x_str)  # NOTE: key (N x S x E_k)

        # print('x_seq.shape, x_str.shape:\t', x_seq.shape, x_str.shape)

        x_seq_input = x_seq
        x_str_input = x_str
        if self.add_pos_enc_seq:
            x_seq_input = torch.cat(
                [x_seq_input, torch.roll(x_str_input, shifts=1, dims=1)], dim=2
            )  ##TODO: RECHECK CODE.......
            x_seq_input = x_seq_input + data["positional_encoding"]
            # x_seq_input[:, 2:-2, :] = x_seq_input[:, 2:-2, :] + x_str_input[:, :-4, :]
            # concat instead
        if self.add_pos_enc_str:
            raise Exception("We are not using this for now.")
            x_str_input = x_str + data["positional_encoding"][..., : x_str.shape[2]]

        ##@ multihead cross-attention
        attn_mask = None
        # pad_mask = data['pad_mask'].float()
        # attn_mask = torch.bmm(pad_mask.unsqueeze(2), pad_mask.unsqueeze(1))                         ## TODO: should be (N x S x S)
        # attn_mask = attn_mask.unsqueeze(0).repeat((self.num_attn_heads, 1, 1, 1)).permute(1,0,2,3)  ## TODO: should be (N x num_head x S x S)
        # attn_mask = attn_mask[:, :, :, :-4] ## TODO: should be (N x num_head x L x S)
        # print(attn_mask.shape); print(attn_mask); exit()

        for b_idx, attn_layer in enumerate(self.multihead_attn):
            attn_output, attn_output_weights = attn_layer(
                query=x_seq_input,
                key=x_seq_input,
                value=x_seq_input,
                key_padding_mask=data["pad_mask"],
                need_weights=data["need_attn_weights"],
                attn_mask=attn_mask,
            )

            ##@ bottleneck ffn
            attn_output = self.bottleneck[b_idx](attn_output)

            ##@ residual connection
            if b_idx == 0:
                x_seq_input = attn_output + x_seq_input
            else:
                x_seq_input = attn_output + x_seq_input

        x = self.linear_out(x_seq_input)

        x = x[:, :, :21]

        # print(x.shape, x_seq.shape, x_str.shape); exit()

        return x


class MultiheadAttnMDLMHead(nn.Module):
    def __init__(
        self,
        c_in,
        embed_dim,
        num_blocks=1,
        num_attn_heads=8,
        attn_dropout=0.1,
        add_pos_enc_seq=True,
    ):
        super().__init__()

        self.linear_in_seq = nn.Linear(c_in, embed_dim * 2)

        self.multihead_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim * 2,
                    num_heads=num_attn_heads,
                    batch_first=True,
                    dropout=attn_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(attn_dropout),
            nn.ELU(inplace=True),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Dropout(attn_dropout),
            nn.ELU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Dropout(attn_dropout),
            nn.ELU(inplace=True),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Dropout(attn_dropout),
        )
        self.linear_out = nn.Linear(
            embed_dim * 2, 5
        )  ## index=0,1,2,3 ==> AGCU tokens ; index=4, ==> [MASK] token.

        self.add_pos_enc_seq = add_pos_enc_seq
        self.num_attn_heads = num_attn_heads
        self.num_blocks = num_blocks
        self.c_in = c_in
        self.embed_dim = embed_dim

    def forward(self, data):
        x_seq = data["lm_representation"]
        x_seq = self.linear_in_seq(x_seq)  # NOTE: query (N x L x E_q)

        # print('x_seq.shape, x_str.shape:\t', x_seq.shape, x_str.shape)

        x_seq_input = x_seq
        if self.add_pos_enc_seq:
            x_seq_input = x_seq_input + data["positional_encoding"]

        ##@ multihead cross-attention
        attn_mask = None
        # pad_mask = data['pad_mask'].float()
        # attn_mask = torch.bmm(pad_mask.unsqueeze(2), pad_mask.unsqueeze(1))                         ## TODO: should be (N x S x S)
        # attn_mask = attn_mask.unsqueeze(0).repeat((self.num_attn_heads, 1, 1, 1)).permute(1,0,2,3)  ## TODO: should be (N x num_head x S x S)
        # attn_mask = attn_mask[:, :, :, :-4] ## TODO: should be (N x num_head x L x S)
        # print(attn_mask.shape); print(attn_mask); exit()

        for b_idx, attn_layer in enumerate(self.multihead_attn):
            attn_output, attn_output_weights = attn_layer(
                query=x_seq_input,
                key=x_seq_input,
                value=x_seq_input,
                key_padding_mask=data["pad_mask"],
                need_weights=data["need_attn_weights"],
                attn_mask=attn_mask,
            )

            ##@ bottleneck ffn
            attn_output = self.bottleneck(attn_output)

            ##@ residual connection
            if b_idx == 0:
                x_seq_input = attn_output + x_seq_input
            else:
                x_seq_input = attn_output + x_seq_input

        x = self.linear_out(x_seq_input)

        # print(x.shape, x_seq.shape, x_str.shape); exit()

        return x


class VanillaMDLMHead(nn.Module):
    def __init__(self, c_in, embed_dim, dropout=0.1, add_pos_enc_seq=True):
        super().__init__()

        self.add_pos_enc_seq = add_pos_enc_seq
        self.c_in = c_in
        self.embed_dim = embed_dim

        self.linear_in_seq = nn.Sequential(
            nn.Linear(c_in, embed_dim * 2),
        )

        self.pred_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
            nn.ELU(inplace=True),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Dropout(dropout),
            nn.ELU(inplace=True),
            nn.Linear(
                embed_dim // 2, 5
            ),  ## index=0,1,2,3 ==> AGCU tokens ; index=4, ==> [MASK] token.
        )

    def forward(self, data):
        x_seq = data["lm_representation"]
        x_seq = self.linear_in_seq(x_seq)
        # print('x_seq.shape, x_str.shape:\t', x_seq.shape, data['positional_encoding'].shape); exit()

        x_seq_input = x_seq
        if self.add_pos_enc_seq:
            x_seq_input = (
                x_seq_input + data["positional_encoding"][..., : x_seq_input.shape[2]]
            )

        ##@ predictor mlp
        x = self.pred_mlp(x_seq_input)

        # print(x.shape, x_seq.shape, x_str.shape); exit()

        return x
