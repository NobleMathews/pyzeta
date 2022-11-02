import enum
from enum import Enum
from pathlib import Path

# import sys
# sys.path.append("../")
from solana.publickey import PublicKey


class Asset(enum.Enum):
    SOL = 0
    BTC = 1
    ETH = 2
    UNDEFINED = 255


def assetToName(asset):
    if asset == Asset.SOL:
        return "SOL"
    elif asset == Asset.BTC:
        return "BTC"
    elif asset == Asset.ETH:
        return "ETH"
    elif asset == Asset.UNDEFINED:
        return "UNDEFINED"
    else:
        return "ERROR"  # Invalid asset type


def nameToAsset(name):
    if name == "SOL":
        return Asset.SOL
    elif name == "BTC":
        return Asset.BTC
    elif name == "ETH":
        return Asset.ETH
    elif name == "UNDEFINED":
        return Asset.UNDEFINED
    else:
        return "ERROR"  # Invalid input name


def allAssets():
    retAssets = []
    for asset in Asset:
        if isinstance(asset.value, int):
            retAssets.append(nameToAsset(asset.name))
    return retAssets


def isValidType(asset):
    return assetToName(asset) != "ERROR"


def isValidStr(assetStr):
    return nameToAsset(assetStr) != "ERROR"


def getAssetMint(asset):
    if assetToName(asset) != "ERROR":
        return MINTS[assetToName(asset)]
    return -1


def toProgramAsset(asset):
    if asset == Asset.SOL: return {"sol": {}}
    if asset == Asset.BTC: return {"btc": {}}
    if asset == Asset.ETH: return {"eth": {}}
    return "ERROR"


def fromProgramAsset(obj):
    if obj == {"sol": {}}: return Asset.SOL
    if obj == {"btc": {}}: return Asset.BTC
    if obj == {"eth": {}}: return Asset.ETH
    return "ERROR"


class Network(Enum):
    LOCALNET = "localnet"
    DEVNET = "devnet"
    MAINNET = "mainnet"


def toNetwork(networkStr):
    if networkStr == "localnet": return Network.LOCALNET
    if networkStr == "devnet": return Network.DEVNET
    if networkStr == "mainnet": return Network.MAINNET


# MINTS = {"SOL": PublicKey("So11111111111111111111111111111111111111112")}
MINTS = {
    Asset.SOL: PublicKey("So11111111111111111111111111111111111111112"),
    Asset.BTC: PublicKey("qfnqNqs3nCAHjnyCgLRDbBtq4p2MtHZxw8YjSyYhPoL"),
    Asset.ETH: PublicKey("FeGn77dhg1KXRRFeSwwMiykZnZPw5JXW6naf2aQgZDQf")
}
UNDERLYINGS = [MINTS[Asset.SOL], MINTS[Asset.BTC], MINTS[Asset.ETH]]

DEX_PID = {
    Network.LOCALNET: PublicKey("5CmWtUihvSrJpaUrpJ3H1jUa9DRjYz4v2xs6c3EgQWMf"),
    Network.DEVNET: PublicKey("5CmWtUihvSrJpaUrpJ3H1jUa9DRjYz4v2xs6c3EgQWMf"),
    Network.MAINNET: PublicKey("zDEXqXEG7gAyxb1Kg9mK5fPnUdENCGKzWrM21RMdWRq"),
}

MAX_SETTLE_AND_CLOSE_PER_TX = 4
MAX_CANCELS_PER_TX = 4
MAX_GREEK_UPDATES_PER_TX = 20
MAX_SETTLEMENT_ACCOUNTS = 20
MAX_REBALANCE_ACCOUNTS = 20
MAX_SETTLE_ACCOUNTS = 5
MAX_ZETA_GROUPS = 20
MAX_MARGIN_AND_SPREAD_ACCOUNTS = 20
MARKET_INDEX_LIMIT = 18

# 3 accounts per set * 9 = 27 + 2 = 29 accounts.
CLEAN_MARKET_LIMIT = 9
CRANK_ACCOUNT_LIMIT = 12
MAX_MARKETS_TO_FETCH = 50

# This is the most we can load per iteration without
# hitting the rate limit.
MARKET_LOAD_LIMIT = 12

MARGIN_ACCOUNT_ASSET_OFFSET = 5764
SPREAD_ACCOUNT_ASSET_OFFSET = 2305

DEFAULT_ORDERBOOK_DEPTH = 5
MAX_ORDER_TAG_LENGTH = 4

PYTH_PRICE_FEEDS = {
    Network.LOCALNET: {
        "SOL/USD": PublicKey("2pRCJksgaoKRMqBfa7NTdd6tLYe9wbDFGCcCCZ6si3F7"),
    },
    Network.DEVNET: {
        "SOL/USD": PublicKey("J83w4HKfqxwcq3BEMMkPFSppX3gqekLyLJBexebFVkix"),
    },
    Network.MAINNET: {
        "SOL/USD": PublicKey("H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG"),
    },
}

USDC_MINT_ADDRESS = {
    Network.LOCALNET: PublicKey("6PEh8n3p7BbCTykufbq1nSJYAZvUp6gSwEANAs1ZhsCX"),
    Network.DEVNET: PublicKey("6PEh8n3p7BbCTykufbq1nSJYAZvUp6gSwEANAs1ZhsCX"),
    Network.MAINNET: PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
}

CLUSTER_URLS = {
    "localnet": "http://127.0.0.1:8899",
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}

# These are fixed and shouldn't change in the future.
NUM_STRIKES = 11
PRODUCTS_PER_EXPIRY = NUM_STRIKES * 2 + 1  # +1 for the future.
ACTIVE_EXPIRIES = 2
ACTIVE_MARKETS = ACTIVE_EXPIRIES * PRODUCTS_PER_EXPIRY
TOTAL_EXPIRIES = 6
TOTAL_MARKETS = PRODUCTS_PER_EXPIRY * TOTAL_EXPIRIES

DEFAULT_EXCHANGE_POLL_INTERVAL = 30
DEFAULT_MARKET_POLL_INTERVAL = 5
DEFAULT_CLIENT_POLL_INTERVAL = 20
DEFAULT_CLIENT_TIMER_INTERVAL = 1
UPDATING_STATE_LIMIT_SECONDS = 10

# Numbers represented in BN are generally fixed point integers with precision of 6.
PLATFORM_PRECISION = 6
PRICING_PRECISION = 12
MARGIN_PRECISION = 8
POSITION_PRECISION = 3

DEFAULT_ORDER_TAG = "SDK"

MAX_POSITION_MOVEMENTS = 10
BPS_DENOMINATOR = 10000

# Path to IDL
IDL_PATH = Path('../idl/zeta.json')
