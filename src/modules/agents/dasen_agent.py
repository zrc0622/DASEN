import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse


class PhaseUPDeT1(nn.Module):
    def __init__(self, input_shape, args):
        super(PhaseUPDeT1, self).__init__()
        self.args = args
        self.transformer = Transformer1(args.token_dim, args.emb, args.heads, args.depth, args.emb, args.phase_num, args.mask, args.mask_prob)
        self.q_self = nn.Linear(args.emb+args.phase_rep, 6)
        if(args.divide_Q): self.q_interaction = nn.Linear(args.emb+args.phase_rep, 1)
        self.p_next = GumbelSoftmaxNetwork(args.emb, args.phase_hidden, args.phase_num, args.temperature)

    def init_hidden(self):
        # 创建一个全0的隐藏状态
        return torch.zeros(1, self.args.emb).cuda()

    def init_phase(self):
        # 初始化阶段向量
        one_hot_phase = torch.zeros(1, self.args.phase_num)
        one_hot_phase[0, 0] = 1
        return one_hot_phase.cuda()

    def forward(self, inputs, hidden_state, phase_state, task_enemy_num, task_ally_num, test_mode):
        outputs, _ = self.transformer.forward(inputs, hidden_state, phase_state, None, if_train=not test_mode)
        # 倒数第二层是隐藏层
        h = outputs[:, -2, :]

        # 倒数第一层是阶段层
        p = outputs[:, -1, :]

        # 第一层输出不变动作 (no_op stop up down left right)
        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], p), -1))

        q_enemies_list = []

        # 1~i层为敌人（交互动作）
        if(self.args.divide_Q):
            for i in range(task_enemy_num):
                q_enemy = self.q_interaction(torch.cat((outputs[:, 1 + i, :], p), -1))
                q_enemies_list.append(q_enemy)
        else:
            for i in range(task_enemy_num):
                q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], p), -1))
                q_enemy_mean = torch.mean(q_enemy, 1, True)
                q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        p_n = self.p_next(p)

        # print((p_n>0.5).int())
        # # print(torch.round(p_next * 10) / 10.0)
        # print("-"*50)

        if(self.args.mixer == "pqmix"):
            return q, h, p_n, p
        else:
            return q, h, p_n

class PhaseUPDeT2(nn.Module):
    def __init__(self, input_shape, args):
        super(PhaseUPDeT2, self).__init__()
        self.args = args
        self.transformer = Transformer2(args.token_dim, args.emb, args.heads, args.depth, args.emb, args.mask, args.mask_prob)
        self.q_self = nn.Linear(args.emb+args.phase_rep, 6)
        if(args.divide_Q): self.q_interaction = nn.Linear(args.emb+args.phase_rep, 1)
        self.phase_embedding = nn.Linear(args.phase_num, args.phase_rep)
        self.phase_representation = MLP(args.emb+args.phase_rep, args.phase_hidden, args.phase_rep)
        self.phase_next = GumbelSoftmaxLayer(args.phase_rep, args.phase_num, args.temperature)
        self.p_rep = None

    def init_hidden(self):
        # 创建一个全0的隐藏状态
        return torch.zeros(1, self.args.emb).cuda()

    def init_phase(self):
        # 初始化阶段向量
        # print("-"*50)
        one_hot_phase = torch.zeros(1, self.args.phase_num)
        one_hot_phase[0, 0] = 1
        return one_hot_phase.cuda()

    def forward(self, inputs, hidden_state, phase_state, task_enemy_num, task_ally_num, test_mode):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None, if_train=not test_mode)

        # 倒数第一层是隐藏层
        h = outputs[:, -1:, :]

        # 计算当前阶段
        p_emb = self.phase_embedding(phase_state)
        p_rep = self.phase_representation(torch.cat((h, p_emb), -1)) # 当前阶段
        p_rep = p_rep.squeeze()

        # 第一层输出不变动作 (no_op stop up down left right)
        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], p_rep), -1))

        q_enemies_list = []

        # 第1~i层为敌人，输出交互动作
        if(self.args.divide_Q):
            for i in range(task_enemy_num):
                q_enemy = self.q_interaction(torch.cat((outputs[:, 1 + i, :], p_rep), -1))
                q_enemies_list.append(q_enemy)
        else:
            for i in range(task_enemy_num):
                q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], p_rep), -1))
                q_enemy_mean = torch.mean(q_enemy, 1, True)
                q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        p_next = self.phase_next(p_rep)
        self.p_rep = p_rep
        # print((p_next>0.5).int())
        # # print(torch.round(p_next * 10) / 10.0)
        # print("-"*50)

        if self.args.pqmix_v2:
            p_indices = torch.argmax(p_next, dim=-1) + 1
            return q, h, p_next, p_indices
        else:
            return q, h, p_next, p_rep # kl_1

class PhaseUPDeT3(nn.Module):
    def __init__(self, input_shape, args):
        super(PhaseUPDeT3, self).__init__()
        self.args = args
        self.transformer = Transformer2(args.token_dim, args.emb, args.heads, args.depth, args.emb, args.mask, args.mask_prob)
        self.q_self = nn.Linear(args.emb+args.phase_rep, 6)
        if(args.divide_Q): self.q_interaction = nn.Linear(args.emb+args.phase_rep, 1)

        self.p_rep = None
        p_e = self.generate_orthogonal_vectors(args.phase_num, args.phase_rep)
        self.p_e_w = nn.Parameter(p_e, requires_grad=False) # (6, 32)
        self.classify = DoubleMLP(args.emb+args.phase_rep, args.phase_hidden1, args.phase_hidden2, args.phase_num)

    def gram_schmidt(self, vectors):
        orthogonal_vectors = []
        for v in vectors:
            w = v.clone()
            for u in orthogonal_vectors:
                w -= (u @ w) * u
            w /= torch.norm(w)
            orthogonal_vectors.append(w)
        return torch.stack(orthogonal_vectors)

    def generate_orthogonal_vectors(self, n, m):
        # 生成n个维度为m的随机向量
        random_vectors = torch.randn(n, m)
        # 进行Gram-Schmidt正交化
        orthogonal_vectors = self.gram_schmidt(random_vectors)
        return orthogonal_vectors

    def init_hidden(self):
        # 创建一个全0的隐藏状态
        return torch.zeros(1, self.args.emb).cuda()

    def init_phase(self):
        # 初始化阶段向量
        phase = self.p_e_w[0, :]
        return phase.cuda()

    def forward(self, inputs, hidden_state, p_h, task_enemy_num, task_ally_num, test_mode):
        outputs, _ = self.transformer.forward(inputs, hidden_state, None, if_train=not test_mode)

        # 倒数第一层是隐藏层
        h = outputs[:, -1:, :]

        # 计算当前阶段
        p_n_logits = self.classify(torch.cat((h, p_h), -1)) # (32, 5, 6)
        p_n_probs = F.softmax(p_n_logits, dim=-1)
        p_rep = torch.matmul(p_n_probs, self.p_e_w) # (32, 5, 6) * (6, 32)
        p_rep = p_rep.squeeze()

        # 第一层输出不变动作 (no_op stop up down left right)
        q_basic_actions = self.q_self(torch.cat((outputs[:, 0, :], p_rep), -1))

        q_enemies_list = []

        # 第1~i层为敌人，输出交互动作
        if(self.args.divide_Q):
            for i in range(task_enemy_num):
                q_enemy = self.q_interaction(torch.cat((outputs[:, 1 + i, :], p_rep), -1))
                q_enemies_list.append(q_enemy)
        else:
            for i in range(task_enemy_num):
                q_enemy = self.q_self(torch.cat((outputs[:, 1 + i, :], p_rep), -1))
                q_enemy_mean = torch.mean(q_enemy, 1, True)
                q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze()

        # concat basic action Q with enemy attack Q
        q = torch.cat((q_basic_actions, q_enemies), 1)

        self.p_rep = p_rep
        # print((p_next>0.5).int())
        # # print(torch.round(p_next * 10) / 10.0)
        # print("-"*50)

        return q, h, p_rep, p_rep

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


class Transformer1(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim, phase_num, if_mask, mask_prob=0.2):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, if_mask=if_mask, mask_prob=mask_prob))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)
        self.phase_embedding = nn.Linear(phase_num, emb)

    def forward(self, x, h, p, mask, if_train):
        p_emb = self.phase_embedding(p) # p是阶段的one hot向量

        tokens = self.token_embedding(x) # (5, 11, 5)->(5, 11, 32): 5个agent、11个entity、5个最大观测
        tokens = torch.cat((tokens, h, p_emb), 1) # (5, 13, 32)

        b, t, e = tokens.size()

        x, mask, if_train = self.tblocks((tokens, mask, if_train))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens) # (5, 13, 32)

        return x, tokens

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

        self.skill_embedding = nn.Linear(skill_num, emb)

    def forward(self, x, h, s, mask):
        s_emb = self.skill_embedding(s)

        tokens = self.token_embedding(x)
        tokens = torch.cat((tokens, h, s_emb), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens

class Transformer2(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim, if_mask, mask_prob):
        super().__init__()

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, if_mask=if_mask, mask_prob=mask_prob))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask, if_train):

        tokens = self.token_embedding(x) # (5, 11, 5)->(5, 11, 32): 5个agent、11个entity、5个最大观测
        tokens = torch.cat((tokens, h), 1) # (5, 12, 32)

        b, t, e = tokens.size()

        x, mask, if_train = self.tblocks((tokens, mask, if_train))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens) # (5, 12, 32)

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

class GumbelSoftmaxLayer(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        super(GumbelSoftmaxLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.temperature = temperature

    def forward(self, x):
        logits = self.fc(x)
        return self.gumbel_softmax(logits, self.temperature)

    def gumbel_softmax(self, logits, temperature):
        gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel(0,1)
        gumbels = (logits + gumbels) / temperature
        y_soft = F.softmax(gumbels, dim=-1)
        return y_soft

class GumbelSoftmaxNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=1.0):
        super(GumbelSoftmaxNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gumbel_softmax_layer = GumbelSoftmaxLayer(hidden_dim, output_dim, temperature)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.gumbel_softmax_layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DoubleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DoubleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--token_dim', default='5', type=int)
    parser.add_argument('--emb', default='32', type=int)
    parser.add_argument('--heads', default='3', type=int)
    parser.add_argument('--depth', default='2', type=int)
    parser.add_argument('--ally_num', default='5', type=int)
    parser.add_argument('--enemy_num', default='5', type=int)
    parser.add_argument('--episode', default='20', type=int)
    parser.add_argument('--phase_num', default='6', type=int)
    args = parser.parse_args()


    # testing the agent
    agent = PhaseUPDeT1(None, args).cuda()
    hidden_state = agent.init_hidden().cuda().expand(args.ally_num, 1, -1)
    tensor = torch.rand(args.ally_num, args.ally_num+args.enemy_num, args.token_dim).cuda()
    q_list = []
    for _ in range(args.episode):
        q, hidden_state = agent.forward(tensor, hidden_state, args.ally_num, args.enemy_num)
        q_list.append(q)