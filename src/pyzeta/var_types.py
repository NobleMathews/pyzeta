from enum import Enum
from my_client.types import KindKind
from my_client.types.kind import Call as KindKindCall, Put as KindKindPut, Future as KindKindFuture, Uninitialized as KindKindUninitialized

class OrderType(Enum):
    LIMIT = "limit"
    POSTONLY = "postonly"
    FILLORKILL = "fillorkill"

class Side(Enum):
    BID = "bid"
    ASK = "ask"

class Kind(Enum):
    UNINITIALIZED = "uninitialized"
    CALL = "call"
    PUT = "put"
    FUTURE = "future"

class MarginType(Enum):
    Initial = "initial"
    Maintenance = "maintenance"

class ProgramAccountType(Enum):
    MARGINACCOUNT = "MarginAccount"
    SPREADACCOUNT = "SpreadAccount"

class MovementType(Enum):
    LOCK = "lock"
    UNLOCK = "unlock"

class MarginAccountType(Enum):
    NORMAL = 0,
    MARKET_MAKER = 1

class MarginRequirement:
    def __init__(self, init_long, init_short, maintenance_lng, maintenance_shrt):
        self.initial_long = init_long
        self.initial_short = init_short
        self.maintenance_long = maintenance_lng
        self.maintenance_short = maintenance_shrt

class MarginAccount:
    def __init__(self, authority, nonce, balance, force_cancel_flag, open_orders_nonce, series_expiry, product_ledgers, product_ledgers_padding, rebalance_amount, asset, account_type, padding):
        self.authority = authority
        self.nonce = nonce
        self.balance = balance
        self.force_cancel_flag = force_cancel_flag
        self.open_orders_nonce = open_orders_nonce
        self.series_expiry = series_expiry
        self.product_ledgers = product_ledgers
        self.product_ledgers_padding = self.product_ledgers_padding
        self.rebalance_amount = rebalance_amount
        self.asset = asset
        self.account_type = account_type
        self.padding = padding
    
def is_market_maker(margin_account: MarginAccount):
    return from_program_margin_account_type(margin_account.account_type) == MarginAccountType.MARKET_MAKER

def from_program_margin_account_type(account_type):
    if account_type == { "normal": {} }:
        return MarginAccountType.NORMAL
    if account_type == { "market_maker": {} }:
        return MarginAccountType.MARKET_MAKER
    raise Exception("Invalid margin account type")

def to_product_kind(kind) -> Kind:
    if type(kind) == KindKindCall: return Kind.CALL
    if type(kind) == KindKindPut: return Kind.PUT
    if type(kind) == KindKindFuture: return Kind.FUTURE

    raise Exception("Invalid product type")