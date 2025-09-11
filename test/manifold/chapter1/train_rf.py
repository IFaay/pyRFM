import torch
import re
from collections import defaultdict


def restore_plain_state_dict(state_dict, eps=1e-12):
    """
    将带有 `parametrizations.weight.original*` 的权重还原为普通 `...weight`。
    假设采用的是 PyTorch 的 weight_norm，默认 dim=0（与官方默认一致）。
    对于 Linear/Conv/Embedding 都可用：对 v 在 dims = (1..N-1) 上做 L2 范数。
    """
    # 收集所有层的 (g, v) 指针
    buckets = defaultdict(dict)
    pat = re.compile(r"^(.*)\.parametrizations\.weight\.original([01])$")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            prefix, idx = m.group(1), m.group(2)
            buckets[prefix][idx] = k  # 记录 original0 / original1 的完整 key

    new_sd = {}

    # 先复制原来就已是“普通”的参数（例如 final_layer.weight/bias、任意 bias）
    for k, v in state_dict.items():
        if ".parametrizations.weight.original" in k:
            continue  # 稍后还原后会以 ...weight 的新 key 放进去
        new_sd[k] = v

    # 对每个需要还原的权重执行 weight_norm 逆变换
    for prefix, pair in buckets.items():
        if "0" not in pair or "1" not in pair:
            raise ValueError(f"{prefix} 缺少 original0/1，无法还原。")

        g = state_dict[pair["0"]].clone()
        v = state_dict[pair["1"]].clone()

        # 计算 ||v||，对除第 0 维以外的全部维度做范数（等价于 weight_norm 的默认 dim=0）
        if v.dim() == 1:
            # 例如某些特殊情况：把 1D 当作 (out,) —— 这时范数就是绝对值
            v_norm = v.abs() + eps
            scale = g / v_norm
            w = v * scale
        else:
            reduce_dims = tuple(range(1, v.dim()))
            v_norm = v.norm(dim=reduce_dims, keepdim=True) + eps
            # g 形状通常是 (out,)；需要 reshape 成 (out, 1, 1, ...) 才能广播
            shape = [g.shape[0]] + [1] * (v.dim() - 1)
            scale = g.view(*shape) / v_norm
            w = v * scale

        # 把还原后的权重写回为 `<prefix>.weight`
        new_sd[f"{prefix}.weight"] = w

    return new_sd


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_in = "checkpoints/tanh-tanh/sdf_last.pt"
    path_out = "checkpoints/tanh-tanh/sdf_last_plain.pt"

    ckpt = torch.load(path_in, map_location=device)
    sd = ckpt.get("model_state", ckpt)  # 兼容：有的 ckpt 外层包了 "model_state"

    plain_sd = restore_plain_state_dict(sd)

    # 如果原来在 ckpt["model_state"] 里，就回填；否则直接保存 plain_sd
    if "model_state" in ckpt:
        ckpt["model_state"] = plain_sd
        torch.save(ckpt, path_out)
    else:
        torch.save(plain_sd, path_out)

    # 简单核对一下你展示的三层
    print("Restored keys (sample):")
    for k in [
        "input_layer.0.weight", "input_layer.0.bias",
        "hidden_layer.0.weight", "hidden_layer.0.bias",
        "final_layer.weight",
    ]:
        if k in plain_sd:
            print("  OK:", k, tuple(plain_sd[k].shape))
        else:
            print("  MISSING:", k)
