"""DLSTM, a shared classification objective, and resumable Adam warm training."""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from .common import atomic_torch, rng_state, restore_rng, atomic_json, CumulativeTimer
from .data import batches


class DLSTM(nn.Module):
    def __init__(self, n_features, widths, dropout=0.1):
        super().__init__()
        if len(widths) != 4 or any(h < 1 for h in widths):
            raise ValueError('Expected four positive hidden widths.')
        self.recurrent = nn.ModuleList()
        n = n_features
        for h in widths[:3]:
            self.recurrent.append(nn.LSTM(n, h, batch_first=True))
            n = h
        self.drop = nn.Dropout(dropout)
        self.hidden = nn.Linear(n, widths[3])
        self.output = nn.Linear(widths[3], 2)

    def forward(self, x, lengths):
        p = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for layer in self.recurrent:
            p, _ = layer(p)
            p = PackedSequence(self.drop(torch.relu(p.data)), p.batch_sizes, p.sorted_indices, p.unsorted_indices)
        z, _ = pad_packed_sequence(p, batch_first=True)
        last = z[torch.arange(len(lengths), device=z.device), lengths.to(z.device) - 1]
        return self.output(torch.relu(self.hidden(last)))

    def l1(self):
        return sum(p.abs().sum() for name, p in self.recurrent[0].named_parameters() if 'weight' in name)


def make_model(c, device):
    return DLSTM(18, c['hidden_widths'], c['dropout']).to(device)


def vector(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float32)


@torch.no_grad()
def assign(model, v):
    if len(v) != sum(p.numel() for p in model.parameters()):
        raise ValueError('Parameter vector dimension mismatch.')
    offset = 0
    for p in model.parameters():
        n = p.numel()
        part = np.array(v[offset:offset + n], copy=True).reshape(tuple(p.shape))
        p.copy_(torch.from_numpy(part).to(p.device))
        offset += n


def data_statistic(logits, y, c):
    """Additive statistic: squared errors for RMSE, negative log likelihood for CE."""
    if c['fitness'] == 'half_rmse':
        p = torch.softmax(logits, dim=1)[:, 1]
        return (p - y.to(p.dtype)).square().sum()
    if c['fitness'] == 'cross_entropy':
        return nn.functional.cross_entropy(logits, y, reduction='sum')
    raise ValueError('Unknown fitness: ' + str(c['fitness']))


def objective(model, logits, y, c):
    """Same mathematical objective for Adam and all six metaheuristics."""
    value = data_statistic(logits, y, c) / len(y)
    if c['fitness'] == 'half_rmse':
        value = 0.5 * torch.sqrt(value.clamp_min(c.get('rmse_epsilon', 1e-12)))
    l1_sum = model.l1()
    penalty = c['l1_coefficient'] * l1_sum
    return value + penalty, value, penalty


def predict(model, xs, ys, c, device):
    model.eval()
    out = []
    with torch.inference_mode():
        for x, lengths, y in batches(xs, ys, c['batch_size']):
            out.append(torch.softmax(model(x.to(device), lengths), dim=1)[:, 1].cpu().numpy())
    p = np.concatenate(out)
    if not np.isfinite(p).all():
        raise FloatingPointError('Nonfinite predictions.')
    return p


def fit_baseline(model, xs, ys, c, device, path, progress=None):
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / 'training_state.pt'
    start, history, state = 0, [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state.get('fitness') != c['fitness']:
            raise ValueError('Warm-training objective changed. Use a new output_root.')
        model.load_state_dict(state['model'])
        start, history = state['epoch'], state['history']
        restore_rng(state['rng'])
        if start >= c['epochs']:
            model.zero_grad(set_to_none=True)
            return model
    opt = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    if state is not None:
        opt.load_state_dict(state['optimizer'])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and k != 'step':
                    st[k] = v.to(device)
    del state
    every = max(1, int(c.get('training_log_every_batches', 10)))
    with CumulativeTimer(path / 'training_timing.json') as timer:
        for epoch in range(start, c['epochs']):
            model.train()
            loss_sum = data_sum = penalty_sum = 0.0
            total_batches = math.ceil(len(ys) / c['batch_size'])
            epoch_start = time.perf_counter()
            for batch, (x, lengths, y) in enumerate(batches(xs, ys, c['batch_size'], shuffle=True), 1):
                opt.zero_grad(set_to_none=True)
                loss, data, penalty = objective(model, model(x.to(device), lengths), y.to(device), c)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                opt.step()
                loss_sum += float(loss.detach()) * len(y)
                data_sum += float(data.detach()) * len(y)
                penalty_sum += float(penalty.detach()) * len(y)
                if batch == 1 or batch % every == 0 or batch == total_batches:
                    timer.checkpoint()
                    if progress:
                        progress()
                    print(f'Adam epoch={epoch + 1}/{c["epochs"]} batch={batch}/{total_batches} '
                          f'loss={loss_sum / min(batch * c["batch_size"], len(ys)):.6g}', flush=True)
            history.append(dict(
                epoch=epoch + 1, training_loss=loss_sum / len(ys),
                mean_batch_data_loss=data_sum / len(ys), mean_batch_l1_penalty=penalty_sum / len(ys),
                epoch_training_seconds=time.perf_counter() - epoch_start, objective=c['fitness'],
            ))
            atomic_torch(checkpoint, dict(
                model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1,
                history=history, rng=rng_state(), fitness=c['fitness'],
            ))
            atomic_json(path / 'training_history.json', history)
            timer.checkpoint()
            if progress:
                progress()
            print(f'epoch {epoch + 1}/{c["epochs"]} loss={history[-1]["training_loss"]:.6g}', flush=True)
    model.zero_grad(set_to_none=True)
    return model


class Fitness:
    def __init__(self, model, xs, ys, c, device):
        self.model, self.xs, self.ys = model, xs, np.asarray(ys)
        self.c, self.device = c, device
        self.last_components = {}

    def __call__(self, v):
        assign(self.model, v)
        self.model.eval()
        self.last_components = {}
        with torch.inference_mode():
            statistic = 0.0
            for x, lengths, y in batches(self.xs, self.ys, self.c['batch_size']):
                logits = self.model(x.to(self.device), lengths)
                statistic += float(data_statistic(logits, y.to(self.device), self.c))
            data_loss = statistic / len(self.ys)
            if self.c['fitness'] == 'half_rmse':
                data_loss = 0.5 * np.sqrt(max(data_loss, self.c.get('rmse_epsilon', 1e-12)))
            l1_sum = float(self.model.l1())
        penalty = self.c['l1_coefficient'] * l1_sum
        loss = float(data_loss + penalty)
        if not np.isfinite(loss):
            self.last_components = {'nonfinite': True}
            return 1e30
        self.last_components = dict(data_loss=float(data_loss), l1_sum=l1_sum, l1_penalty=penalty)
        return loss


def parameter_count(widths, n_features=18):
    n, total = n_features, 0
    for h in widths[:3]:
        total += 4 * h * (n + h + 2)
        n = h
    return total + n * widths[3] + widths[3] + widths[3] * 2 + 2
