from enum import Enum


class Status(Enum):
    OBSERVING = 0
    SELL = 1
    BUY = 2
    HOLDING_BUY = 3
    HOLDING_SELL = 4


class Stop_status(Enum):
    LOSS = 0
    PROFIT = 1


class market_status(Enum):
    SHORT = 0
    LONG = 1
    NONE = 2