import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse

class DASEN_v1(nn.Module):
    def __init__(self, input_shape, args):
        super(DASEN_v1, self).__init__()
        self.args = args
        self.transformer = SkillTransformer(args.token_dim, args.emb, args.heads, args.depth, args.emb, args.skill_num)
        self.q_self = nn.Linear(args.emb+args.emb, 6)
        self.skill_encoder = GumbelMLP(args.emb, args.skill_hidden, args.skill_num, args.gumbel_temperature, args.gumbel_hard)

    def init_hidden(self):
        return torch.zeros(1, self.args.emb).cuda()

    def init_skill(self):
        skill_init = torch.zeros(1, self.args.skill_num)
        skill_init[0, 0] = 1
        return skill_init.cuda()

    def forward(self, inputs, hidden_state, skill_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, skill_state, None)

        h = outputs[:, -2, :]

        s = outputs[:, -1, :]

        # no_op stop up down left right
        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], s), -1))

        q_enemies_list = []

        for i in range(task_enemy_num):
            q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], s), -1))
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        skill_state = self.skill_encoder(s)

        return q, h, skill_state

class DASEN_v2(nn.Module):
    def __init__(self, input_shape, args):
        super(DASEN_v2, self).__init__()
        self.args = args
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        self.q_self = nn.Linear(args.emb+args.skill_emb, 6)
        self.skill_decoder = nn.Linear(args.skill_num, args.skill_emb)
        self.skill_selector = MLP(args.emb+args.skill_emb, args.skill_hidden, args.skill_emb)
        self.skill_encoder = GumbelLayer(args.skill_emb, args.skill_num, args.gumbel_temperature, args.gumbel_hard)

    def init_hidden(self):
        return torch.zeros(1, self.args.emb).cuda()

    def init_skill(self):
        skill_init = torch.zeros(1, self.args.skill_num)
        skill_init[0, 0] = 1
        return skill_init.cuda()

    def forward(self, inputs, hidden_state, skill_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)

        h = outputs[:, -1:, :]

        skill_emb = self.skill_decoder(skill_state)
        s = self.skill_selector(torch.cat((h, skill_emb), -1)).squeeze()

        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], s), -1))

        q_enemies_list = []

        for i in range(task_enemy_num):
            q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], s), -1))
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        skill_state = self.skill_encoder(s)

        return q, h, skill_state

class DASEN_v3(nn.Module):
    def __init__(self, input_shape, args):
        super(DASEN_v3, self).__init__()
        self.args = args
        self.transformer = Transformer(args.token_dim, args.emb, args.heads, args.depth, args.emb)
        
        self.q_self = nn.Linear(args.emb+args.skill_num, 6)

        self.skill_embedding = nn.Linear(args.skill_num, args.skill_emb)
        
        self.skill_selector = GumbelMLP(args.emb+args.skill_emb, args.skill_hidden, args.skill_num, args.gumbel_temperature, args.gumbel_hard)

    def init_hidden(self):
        return torch.zeros(1, self.args.emb).cuda()
    
    def init_skill(self):
        skill_init = torch.zeros(1, self.args.skill_num)
        skill_init[0, 0] = 1
        return skill_init.cuda()

    def forward(self, inputs, hidden_state, phase_state, task_enemy_num, task_ally_num):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None)
        
        h = outputs[:, -1:, :]

        skill_emb = self.skill_embedding(phase_state)
        
        s = self.skill_selector(torch.cat((h, skill_emb), -1)).squeeze()

        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], s), -1))

        q_enemies_list = []

        for i in range(task_enemy_num):
            q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], s), -1))
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        max_skill = s.max(dim=-1)[1]
        dist_skill = torch.eye(s.shape[-1], device=s.device)[max_skill]

        return q, h, dist_skill

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask

# DASEN-v1
class SkillTransformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim, skill_num):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

        self.skill_decoder = nn.Linear(skill_num, emb)

    def forward(self, x, h, s, mask):
        s_emb = self.skill_decoder(s)

        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h, s_emb), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens

class Transformer(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):

        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GumbelLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gumbel_temperature=1.0, gumbel_hard=False):
        super(GumbelLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.tau = gumbel_temperature
        self.gumbel_hard = gumbel_hard
    
    def forward(self, x):
        x = self.fc1(x)
        y = F.gumbel_softmax(x, tau=self.tau, hard=self.gumbel_hard)
        
        return y

class GumbelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gumbel_temperature=1.0, gumbel_hard=False):
        super(GumbelMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tau = gumbel_temperature
        self.gumbel_hard = gumbel_hard
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        y = F.gumbel_softmax(x, tau=self.tau, hard=self.gumbel_hard)
        
        return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    parser.add_argument('--skill_num', default='4', type=int)
    args = parser.parse_args()
