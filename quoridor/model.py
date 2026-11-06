import torch
import torch.nn as nn
import torch.nn.functional as F

N_PLANES = 8
N_PAWN_ACTIONS = 12


def global_pool(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, pooled: bool) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.pool_fc = nn.Linear(2 * ch, ch) if pooled else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.bn1(x)))
        if self.pool_fc is not None:
            y = y + self.pool_fc(global_pool(y))[:, :, None, None]
        y = self.conv2(F.relu(self.bn2(y)))
        return x + y


class QuoridorNet(nn.Module):
    def __init__(
        self, blocks: int = 6, channels: int = 64, head_channels: int = 32
    ) -> None:
        super().__init__()
        self.config = {
            "blocks": blocks,
            "channels": channels,
            "head_channels": head_channels,
        }
        self.stem = nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(
            *[ResBlock(channels, pooled=i % 3 == 2) for i in range(blocks)]
        )
        self.trunk_bn = nn.BatchNorm2d(channels)
        self.policy_conv = nn.Conv2d(channels, head_channels, 1)
        self.wall_logits = nn.Conv2d(head_channels, 2, 1)
        self.pawn_logits = nn.Linear(2 * head_channels, N_PAWN_ACTIONS)
        self.value_conv = nn.Conv2d(channels, head_channels, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(2 * head_channels, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.moves_left_fc = nn.Sequential(
            nn.Linear(2 * head_channels, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.trunk_bn(self.blocks(self.stem(x))))
        p = F.relu(self.policy_conv(h))
        walls = self.wall_logits(p)[:, :, :-1, :-1].flatten(1)
        logits = torch.cat([self.pawn_logits(global_pool(p)), walls], dim=1)
        v = global_pool(F.relu(self.value_conv(h)))
        value = torch.tanh(self.value_fc(v)).squeeze(1)
        moves_left = torch.sigmoid(self.moves_left_fc(v)).squeeze(1)
        return logits, value, moves_left


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(mask == 0, -1e9)


def save(net: QuoridorNet, path: str, extra: dict) -> None:
    torch.save({"config": net.config, "state": net.state_dict(), **extra}, path)


def load(path: str, device: str = "cpu") -> tuple[QuoridorNet, dict]:
    ckpt = torch.load(path, map_location=device)
    net = QuoridorNet(**ckpt["config"]).to(device)
    missing, unexpected = net.load_state_dict(ckpt["state"], strict=False)
    if unexpected or any(not k.startswith("moves_left_fc.") for k in missing):
        raise RuntimeError(
            f"checkpoint mismatch: missing {missing}, unexpected {unexpected}"
        )
    net.eval()
    return net, ckpt
