from enum import Enum, auto


class StageStatus(Enum):
    # ─── Pre-training ────────────────────────────────────────────────────────────
    WARMUP = auto()  # collecting statistics only

    # ─── Main training states ───────────────────────────────────────────────────
    IMPROVING = auto()  # loss is falling steadily (green)

    PLATEAU = auto()  # loss curve is flat & calm (orange)
    FLUCTUATING = auto()  # flat on average but noisy (orange)
    DIVERGING = auto()  # loss rising (red)

    # ─── Terminal ───────────────────────────────────────────────────────────────
    STOP = auto()  # signal caller to exit current stage / early stop
