"""Optional deterministic CPU training for the jointly optimized H/A/D residual."""
import numpy as np
from possession import joint_temporal as joint


def make_network():
    """Lazy Torch import keeps the production inference installation lightweight."""
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.odd_input = nn.Linear(len(joint.CONTROL_NAMES), joint.HIDDEN, bias=False)
            self.even_input = nn.Linear(len(joint.CONTEXT_NAMES), joint.HIDDEN)
            self.odd_convs = nn.ModuleList([nn.Conv1d(joint.HIDDEN, joint.HIDDEN, 3, padding=d, dilation=d, bias=False) for d in joint.DILATIONS])
            self.even_convs = nn.ModuleList([nn.Conv1d(joint.HIDDEN, joint.HIDDEN, 3, padding=d, dilation=d) for d in joint.DILATIONS])
            self.home_output = nn.Linear(joint.HIDDEN, 1, bias=False)
            self.dead_output = nn.Linear(joint.HIDDEN, 1)
            nn.init.zeros_(self.home_output.weight); nn.init.zeros_(self.dead_output.weight); nn.init.zeros_(self.dead_output.bias)

        def forward(self, odd, even, valid):
            mask = valid[:, None, :]
            e = torch.tanh(self.even_input(even)).transpose(1, 2) * mask
            o = torch.tanh(self.odd_input(odd)).transpose(1, 2) * (1 + .25 * e) * mask
            for ec, oc in zip(self.even_convs, self.odd_convs):
                e = torch.tanh(e + ec(e)) * mask
                o = torch.tanh(o + oc(o)) * (1 + .25 * e) * mask
            h = self.home_output(o.transpose(1, 2)).squeeze(-1)
            d = self.dead_output(e.transpose(1, 2)).squeeze(-1)
            return joint.RESIDUAL_LIMIT * torch.tanh(torch.stack((h, d), dim=-1) / joint.RESIDUAL_LIMIT)
    return Network()


def export_model(net, control_scale, context_mean, context_scale, *, metadata=None, training=None):
    value = lambda tensor: tensor.detach().cpu().numpy().tolist()
    model = {"schema_version": 1, "kind": "equivariant_joint_temporal_residual", "sample_hz": 5,
             "dilations": list(joint.DILATIONS), "residual_limit": joint.RESIDUAL_LIMIT,
             "control_names": list(joint.CONTROL_NAMES), "context_names": list(joint.CONTEXT_NAMES),
             "control_scale": np.asarray(control_scale).tolist(), "context_mean": np.asarray(context_mean).tolist(),
             "context_scale": np.asarray(context_scale).tolist(), "odd_input": value(net.odd_input.weight),
             "even_input": value(net.even_input.weight), "even_input_bias": value(net.even_input.bias),
             "home_output": value(net.home_output.weight), "dead_output": value(net.dead_output.weight),
             "dead_output_bias": value(net.dead_output.bias), "metadata": metadata or {}, "training": training or {}}
    for i, (ec, oc) in enumerate(zip(net.even_convs, net.odd_convs)):
        model.update({f"odd_conv_{i}": value(oc.weight), f"even_conv_{i}": value(ec.weight), f"even_conv_bias_{i}": value(ec.bias)})
    return joint.validate_model(model)


def fit(sequences, *, seed=10527, epochs=8, metadata=None):
    """Each sequence is one half at 5 Hz: X, Z, y, w, base_home, base_dead.

    Positive weights must include live and dead examples. X references and zero
    weight rows do not affect scaling or the objective. Inputs remain observation
    context inside each half. The caller is responsible for whole-match splits.
    """
    import torch
    from torch.nn import functional as F
    if not sequences or type(epochs) is not int or epochs < 1 or type(seed) is not int:
        raise ValueError("Need sequences, an integer seed and positive epochs")
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    total = 0.; odd_square = np.zeros(len(joint.CONTROL_NAMES)); ctx_sum = np.zeros(len(joint.CONTEXT_NAMES)); ctx_square = ctx_sum.copy()
    rows = live_rows = dead_rows = 0
    class_rows = {label: 0 for label in "HAD"}
    checked = []
    for item in sequences:
        x, z = joint.observations(item["X"], item["Z"])
        y, w = np.asarray(item["y"]), np.asarray(item["w"], dtype=float)
        bh, bd = np.asarray(item["base_home"], dtype=float), np.asarray(item["base_dead"], dtype=float)
        if (y.shape != (len(x),) or w.shape != y.shape or bh.shape != y.shape or bd.shape != y.shape
                or not np.isin(y, ["H", "A", "D", "X"]).all() or not np.isfinite(w).all() or (w < 0).any()
                or not np.isfinite(bh).all() or not np.isfinite(bd).all()):
            raise ValueError("Invalid joint training labels, weights or baseline logits")
        good = (y != "X") & (w > 0); context = joint.context_features(x, z)
        odd_square += np.sum(x[good] ** 2 * w[good, None], axis=0)
        ctx_sum += np.sum(context[good] * w[good, None], axis=0)
        ctx_square += np.sum(context[good] ** 2 * w[good, None], axis=0); total += w[good].sum()
        rows += int(good.sum()); live_rows += int((good & (y != "D")).sum()); dead_rows += int((good & (y == "D")).sum())
        for label in class_rows:
            class_rows[label] += int((good & (y == label)).sum())
        checked.append((x, z, y, w * good, bh, bd))
    if total <= 0 or not all(class_rows.values()):
        raise ValueError("Joint training requires positive-weight H, A and D examples")
    control_scale = np.sqrt(odd_square / total); control_scale[control_scale < 1e-8] = 1
    context_mean = ctx_sum / total
    context_scale = np.sqrt(np.maximum(0, ctx_square / total - context_mean ** 2)); context_scale[context_scale < 1e-8] = 1
    net = make_network(); optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    scaling = {"control_scale": control_scale, "context_mean": context_mean, "context_scale": context_scale}
    prepared = []; windows = []; width = 128; halo = sum(joint.DILATIONS)
    for i, (x, z, y, w, bh, bd) in enumerate(checked):
        odd, even = joint.normalized_inputs(scaling, x, z)
        prepared.append((odd.astype(np.float32), even.astype(np.float32), (y == "H").astype(np.float32),
                         (y == "D").astype(np.float32), w.astype(np.float32), bh.astype(np.float32), bd.astype(np.float32)))
        windows.extend((i, start) for start in range(0, len(y), width) if w[start:start + width].sum() > 0)
    rng = np.random.default_rng(seed); losses = []
    for epoch in range(epochs):
        order = rng.permutation(len(windows)); loss_sum = weight_sum = 0.
        for at in range(0, len(order), 16):
            batch = [windows[k] for k in order[at:at + 16]]; size = len(batch)
            bx = np.zeros((size, width + 2 * halo, len(joint.CONTROL_NAMES)), dtype=np.float32)
            bz = np.zeros((size, width + 2 * halo, len(joint.CONTEXT_NAMES)), dtype=np.float32)
            bm = np.zeros((size, width + 2 * halo), dtype=np.float32)
            yh = np.zeros((size, width), dtype=np.float32); yd = yh.copy(); bw = yh.copy(); bh = yh.copy(); bd = yh.copy()
            for j, (i, start) in enumerate(batch):
                x, z, h, d, w, home, death = prepared[i]; stop = min(start + width, len(x))
                left, right = max(0, start - halo), min(len(x), start + width + halo)
                region = slice(left - start + halo, right - start + halo)
                bx[j, region] = x[left:right]; bz[j, region] = z[left:right]; bm[j, region] = 1
                for target, source in ((yh, h), (yd, d), (bw, w), (bh, home), (bd, death)):
                    target[j, :stop - start] = source[start:stop]
            optimizer.zero_grad(set_to_none=True)
            adjustment = net(torch.from_numpy(bx), torch.from_numpy(bz), torch.from_numpy(bm))[:, halo:halo + width]
            death = torch.from_numpy(yd); weight = torch.from_numpy(bw)
            dead_loss = F.binary_cross_entropy_with_logits(adjustment[:, :, 1] + torch.from_numpy(bd), death, reduction="none")
            home_loss = F.binary_cross_entropy_with_logits(adjustment[:, :, 0] + torch.from_numpy(bh), torch.from_numpy(yh), reduction="none")
            loss = (weight * (dead_loss + (1 - death) * home_loss)).sum() / weight.sum()
            if not torch.isfinite(loss): raise ValueError("Joint training produced a nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            loss_sum += float(loss.detach()) * float(weight.sum()); weight_sum += float(weight.sum())
        losses.append(loss_sum / weight_sum)
        print(f"  joint epoch {epoch + 1}/{epochs}: weighted H/A/D loss {losses[-1]:.6f}", flush=True)
    return export_model(net, control_scale, context_mean, context_scale, metadata=metadata, training={
        "seed": seed, "epochs": epochs, "positive_weight_rows": rows, "live_rows": live_rows, "dead_rows": dead_rows,
        "class_rows": class_rows,
        "losses": losses, "objective": "three-class H/A/D negative log likelihood, factorized as D and H conditional on live",
        "optimizer": "AdamW lr=.002 weight_decay=.01 clip_grad_norm=5", "cpu_threads": 1, "torch_version": torch.__version__})
