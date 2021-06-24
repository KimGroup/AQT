import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


    
# BUILD THE NETWORK

class Transformer(nn.Module):
    """
    A standard Transformer architecture. Base for this and many 
    other models.
    """
    def __init__(self, decoder, tgt_embed, generator, Nq, Na):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.Nq = Nq
        self.Na = Na
        
    def forward(self, tgt, tgt_mask):
        "Take in and process masked target sequences."
        return self.decoder(self.tgt_embed(tgt), tgt_mask)

    def p(self, a_vec, ret_tensor=False):
        outcome = list(a_vec)
        for nq in range(self.Nq):
            outcome[nq] += 3
        trg = torch.tensor([[1] + outcome + [2]])
        trg_mask = Batch.make_std_mask(trg, 0)
        out=self.forward(trg, trg_mask)
        log_p=self.generator(out)
        p_tensor = torch.exp(log_p)
        
        p = 1.
        
        for nq in range(self.Nq):
            p *= p_tensor[0,nq,outcome[nq]].item()

        if ret_tensor:
            return (p, p_tensor)
        else:
            return p
    
    def generate_next(self, a_vec):

        a_vec = list(a_vec)
        
        for a_ind in range(len(a_vec)):
            a_vec[a_ind] += 3
        trg = torch.tensor([[1] + a_vec ] )
        trg_mask = Batch.make_std_mask(trg, 0)
        out=self.forward(trg, trg_mask)
        log_p=self.generator(out)
        p_tensor = torch.exp(log_p)

        p_vec = p_tensor[0,-1].detach().numpy()[3:]
        
        p_vec = np.array(p_vec)/sum(p_vec) # RENORMALIZE DUE TO NUMERICAL ERRORS
        next_a = np.random.choice(np.arange(len(p_vec)), size=None, p=p_vec)

        return next_a

    def samples(self, Ns):
        Nq = self.Nq
        outcomes = np.zeros((Ns, Nq), dtype=int)
        for ns in range(Ns):
            outcome = []
            for nq in range(Nq):
                o = self.generate_next(outcome)
                outcome = outcome + [o]
            outcomes[ns] = np.array(outcome)

        return outcomes
        
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
# BASIC BUILDING BLOCKS    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
# / BUILDING BLOCKS

# DECODER
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)
    
# / DECODER

# MASK FOR SEQUENTIAL GENERATIVE MODELLING
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
# / MASK


# ATTENTION MODULE

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
# / ATTENTION


# FEED FORWARD MODULE
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
# / FEED FORWARD

# EMBEDDING
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
# / EMBEDDING
    
# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # GHZ state: Permutation Invariant
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
# / POSITIONAL





# DATA BATCHING

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, trg=None, pad=0):
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    
# / BATCHING





# OPTIMIZATION: changes learning rate. increases linearly for n=warmup steps, then decays as sqrt(step)
# class NoamOpt:
#     "Optim wrapper that implements rate."
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0
        
#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
        
#     def rate(self, step = None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         # return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
#         return self.factor*0.01
        
# def get_std_opt(model):
#     return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
#             torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# / OPTIMIZATION

# LOSS FUNCTION
def LossFunction(x, y):
    # print(torch.tensor(y, dtype=torch.long))
    # loss_NLL = nn.NLLLoss(size_average=False)(x,torch.tensor(y,dtype=torch.long))
    loss_KL = nn.KLDivLoss(size_average=False)(x,y)
    loss_L1 = 0.*nn.L1Loss(size_average=False)(x,y)
    loss_L2 = 0.*nn.MSELoss(size_average=False)(x,y)

    return loss_KL+loss_L1+loss_L2
# / LOSS FUNCTION


# LABEL SMOOTHING ? 
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = LossFunction
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size(0) > 0:
            print('LabelSmoothing: A wild Padding Character has appeared!')
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
# / LABEL SMOOTHING


# COMPUTE LOSS AND BACK PROPAGATE
    
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item() * norm
    
    
# / LOSS, BACKPROP


def data_to_torch(data):
    # Padding = 0
    # Start-of-line character = 1
    # End-of-line character = 2
    # Tokens = {3, ...}

    data = np.array(data)
    Ns = len(data)
    Nq = len(data[0])
    data_np = np.zeros((Ns, Nq+2),dtype=int)

    for ns in range(Ns):
        data_np[ns, 0] = 1
        data_np[ns, -1] = 2
        data_np[ns,1:-1] = data[ns]+3

    np.random.shuffle(data_np)
    return torch.from_numpy(data_np)

def data_gen(data, batch_size):
    Ns = len(data)
    
    # batch_size = int(n_data/nbatches)
    Nbatch = int(Ns / batch_size)
    
    # Data batching
    for nb in range(Nbatch):
        data_tgt = data[nb*batch_size:min((nb+1)*batch_size, Ns)]
        
        tgt = Variable(data_tgt, requires_grad=False)
        
        yield Batch(tgt, 0)
    # / batch


# MAKE MODEL. SET HYPERPARAMETERS.
    
def make_model(Nq, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab), Nq, tgt_vocab-3)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
# / MAKE MODEL

# RUN ONE EPOCH
    
def run_epoch(data_iter, model, loss_compute, verbose=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.trg, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens.item())

        ntokens = batch.ntokens.item()
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            if verbose:
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %(i, loss / ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# / ONE EPOCH


def InitializeModel(Nq, Nlayer=2, dmodel=128, Nh=4, Na=4, dropout=0.):
    V = Na+3
    # Initialize Model
    model = make_model(Nq=Nq, tgt_vocab=V, N=Nlayer, d_model=dmodel,d_ff=4*dmodel,h=Nh,dropout=dropout)

    return model

def TrainModel(model, train_data_np, test_data_np, device, smoothing=0.0, lr=0.001, batch_size=100, Nep=20):

    V = model.Na+3

    # Train Model
    loss = np.zeros((2, Nep))

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Nep, eta_min=0.)

    train_data = data_to_torch(train_data_np).to(device)
    test_data = data_to_torch(test_data_np).to(device)

    for epoch in range(Nep):
        
        model.train()
        loss[0, epoch] = run_epoch(
            data_gen(train_data, batch_size),
            model, 
            SimpleLossCompute(model.generator, criterion, optimizer),
            verbose=False)
        model.eval()
        loss[1, epoch] = run_epoch(
            data_gen(test_data, batch_size),
            model, 
            SimpleLossCompute(model.generator, criterion, None),
            verbose=False)
        print(epoch+1,':',loss[1, epoch])
        scheduler.step()

    train_data.to('cpu')
    test_data.to('cpu')
    return model, loss

