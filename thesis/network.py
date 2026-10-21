import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from .common import atomic_torch, rng_state, restore_rng, atomic_json
from .data import batches

class DLSTM(nn.Module):
    def __init__(self,n_features,widths,dropout=0.1):
        super().__init__()
        if len(widths)!=4:
            raise ValueError('Expected four hidden widths.')
        
        self.recurrent=nn.ModuleList()
        n=n_features
        for h in widths[:3]:
            self.recurrent.append(nn.LSTM(n,h,batch_first=True))
            n=h

        self.drop=nn.Dropout(dropout)
        self.hidden=nn.Linear(n,widths[3])
        self.output=nn.Linear(widths[3],2)

    def forward(self,x,lengths):
        p=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        for layer in self.recurrent:
            p,_=layer(p)
            p=PackedSequence(self.drop(torch.relu(p.data)),p.batch_sizes,p.sorted_indices,p.unsorted_indices)
        
        z,_=pad_packed_sequence(p,batch_first=True)
        last=z[torch.arange(len(lengths),device=z.device),lengths.to(z.device)-1]
        return self.output(torch.relu(self.hidden(last)))
    
    def l1(self):
        return sum(p.abs().sum() for name,p in self.recurrent[0].named_parameters() if 'weight' in name)

def make_model(c,device):
    return DLSTM(18,c['hidden_widths'],c['dropout']).to(device)

def vector(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float32)

@torch.no_grad()
def assign(model,v):
    offset=0
    for p in model.parameters():
        n=p.numel()
        part=np.array(v[offset:offset+n],copy=True).reshape(tuple(p.shape))
        p.copy_(torch.from_numpy(part).to(p.device))
        offset+=n
    
    if offset!=len(v): 
        raise ValueError('Parameter vector dimension mismatch.')

def predict(model,xs,ys,c,device):
    model.eval()
    out=[]
    with torch.inference_mode():
        for x,l,y in batches(xs,ys,c['batch_size']):
            out.append(torch.softmax(model(x.to(device),l),dim=1)[:,1].cpu().numpy())
    
    p=np.concatenate(out)
    if not np.isfinite(p).all(): 
        raise FloatingPointError('Nonfinite predictions.')
    
    return p

def fit_baseline(model,xs,ys,c,device,path):
    path.mkdir(parents=True,exist_ok=True)
    checkpoint=path/'training_state.pt'
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    start=0
    history=[]
    if checkpoint.exists():
        s=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v):
                    st[k]=v.to(device)
        start=s['epoch']
        history=s['history']
        restore_rng(s['rng'])
    
    for epoch in range(start,c['epochs']):
        model.train()
        loss_sum=0.
        for x,l,y in batches(xs,ys,c['batch_size'],shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits=model(x.to(device),l)
            loss=nn.functional.cross_entropy(logits,y.to(device))+c['l1_coefficient']*model.l1()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
            opt.step()
            loss_sum+=float(loss.detach())*len(y)

        history.append(
            dict(
                epoch=epoch+1,
                training_loss=loss_sum/len(ys)
            )
        )
        
        atomic_torch(checkpoint,
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                epoch=epoch+1,
                history=history,
                rng=rng_state()
            )
        )

        atomic_json(path/'training_history.json',history)
        print(f'epoch {epoch+1}/{c["epochs"]} loss={history[-1]["training_loss"]:.6f}',flush=True)

    return model

class Fitness:
    def __init__(self,model,xs,ys,c,device):
        self.model=model
        self.xs=xs
        self.ys=np.asarray(ys)
        self.c=c
        self.device=device

    def __call__(self,v):
        assign(self.model,v)

        try:
            p=predict(self.model,self.xs,self.ys,self.c,self.device)
            if self.c['fitness']=='half_rmse':
                data_loss=0.5*np.sqrt(np.mean((p-self.ys)**2))
            elif self.c['fitness']=='cross_entropy':
                p=np.clip(p,1e-7,1-1e-7)
                data_loss=-np.mean(self.ys*np.log(p)+(1-self.ys)*np.log(1-p))
            else:
                raise ValueError('Unknown fitness')
            
            loss=float(data_loss)+self.c['l1_coefficient']*float(self.model.l1().detach())
           
            return loss if np.isfinite(loss) else 1e30
        except FloatingPointError:
            return 1e30

def parameter_count(widths,n_features=18):
    n=n_features
    total=0
    for h in widths[:3]:
        total+=4*h*(n+h+2)
        n=h
    return total+n*widths[3]+widths[3]+widths[3]*2+2
