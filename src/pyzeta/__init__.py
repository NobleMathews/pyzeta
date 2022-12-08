from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import typing
from dataclasses import field, dataclass
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Union, Optional, List, Dict, ClassVar

import solana.rpc.api
import solana.rpc.websocket_api
from anchorpy import Idl, Program, Provider, Wallet
from loguru import logger
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.websocket_api import connect
from solana.system_program import SYS_PROGRAM_ID
from solana.sysvar import SYSVAR_CLOCK_PUBKEY
from solana.sysvar import SYSVAR_RENT_PUBKEY
from solana.transaction import Transaction
from solana.transaction import TransactionInstruction, AccountMeta
from spl.token.constants import TOKEN_PROGRAM_ID

import pyzeta.instructions
import pyzeta.types

Exchange_network = "devnet"
Exchange_program_id = PublicKey("BG3oRikW8d16YjUEmX3ZxHm9SiJzrGtMhsSR8aCw1Cd7")


class Side(Enum):
    BID = "bid"
    ASK = "ask"


class OrderType(Enum):
    LIMIT = "limit"
    POSTONLY = "postonly"
    FILLORKILL = "FillOrKill"


class InitializeOpenOrdersAccounts(typing.TypedDict):
    state: PublicKey
    zeta_group: PublicKey
    dex_program: PublicKey
    system_program: PublicKey
    open_orders: PublicKey
    margin_account: PublicKey
    authority: PublicKey
    payer: PublicKey
    market: PublicKey
    serum_authority: PublicKey
    open_orders_map: PublicKey
    rent: PublicKey


def initialize_open_orders(
        accounts: InitializeOpenOrdersAccounts,
) -> TransactionInstruction:
    keys: list[AccountMeta] = [
        AccountMeta(pubkey=accounts["state"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["zeta_group"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["dex_program"], is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=accounts["system_program"], is_signer=False, is_writable=False
        ),
        AccountMeta(pubkey=accounts["open_orders"], is_signer=False, is_writable=True),
        AccountMeta(
            pubkey=accounts["margin_account"], is_signer=False, is_writable=True
        ),
        AccountMeta(pubkey=accounts["authority"], is_signer=True, is_writable=False),
        AccountMeta(pubkey=accounts["payer"], is_signer=True, is_writable=True),
        AccountMeta(pubkey=accounts["market"], is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=accounts["serum_authority"], is_signer=False, is_writable=False
        ),
        AccountMeta(
            pubkey=accounts["open_orders_map"], is_signer=False, is_writable=True
        ),
        AccountMeta(pubkey=accounts["rent"], is_signer=False, is_writable=False),
    ]
    identifier = b"7\xea\x10Rd*~\xc0"
    encoded_args = b""
    data = identifier + encoded_args
    return TransactionInstruction(keys, Exchange_program_id, data)


PublicKeyOrStr = Union[PublicKey, str]
_MAGIC = 0xA1B2C3D4
_VERSION_1 = 1
_VERSION_2 = 2
_SUPPORTED_VERSIONS = {_VERSION_1, _VERSION_2}
_ACCOUNT_HEADER_BYTES = 16  # magic + version + type + size, u32 * 4
_NULL_KEY_BYTES = b'\x00' * PublicKey.LENGTH
MAX_SLOT_DIFFERENCE = 25


class PythPriceStatus(Enum):
    UNKNOWN = 0
    TRADING = 1
    HALTED = 2
    AUCTION = 3


class PythPriceType(Enum):
    UNKNOWN = 0
    PRICE = 1


# Join exponential moving average for EMA price and EMA confidence
class EmaType(Enum):
    UNKNOWN = 0
    EMA_PRICE_VALUE = 1
    EMA_PRICE_NUMERATOR = 2
    EMA_PRICE_DENOMINATOR = 3
    EMA_CONFIDENCE_VALUE = 4
    EMA_CONFIDENCE_NUMERATOR = 5
    EMA_CONFIDENCE_DENOMINATOR = 6


current_dir = os.path.dirname(__file__)

MINTS = {
    "SOL": PublicKey("So11111111111111111111111111111111111111112"),
    "BTC": PublicKey("qfnqNqs3nCAHjnyCgLRDbBtq4p2MtHZxw8YjSyYhPoL"),
    "ETH": PublicKey("FeGn77dhg1KXRRFeSwwMiykZnZPw5JXW6naf2aQgZDQf"),
}

DEX_PID = {
    "localnet": PublicKey("5CmWtUihvSrJpaUrpJ3H1jUa9DRjYz4v2xs6c3EgQWMf"),
    "devnet": PublicKey("5CmWtUihvSrJpaUrpJ3H1jUa9DRjYz4v2xs6c3EgQWMf"),
    "mainnet": PublicKey("zDEXqXEG7gAyxb1Kg9mK5fPnUdENCGKzWrM21RMdWRq"),
}

MAX_SETTLE_AND_CLOSE_PER_TX = 4
MAX_CANCELS_PER_TX = 3
MAX_GREEK_UPDATES_PER_TX = 20
MAX_SETTLEMENT_ACCOUNTS = 20
MAX_REBALANCE_ACCOUNTS = 18
MAX_SETTLE_ACCOUNTS = 5
MAX_ZETA_GROUPS = 20
MAX_MARGIN_AND_SPREAD_ACCOUNTS = 20
MAX_SET_REFERRALS_REWARDS_ACCOUNTS = 12
MARKET_INDEX_LIMIT = 18
CLEAN_MARKET_LIMIT = 9
CRANK_ACCOUNT_LIMIT = 12
MAX_MARKETS_TO_FETCH = 50

MARKET_LOAD_LIMIT = 12

DEFAULT_ORDERBOOK_DEPTH = 5
MAX_ORDER_TAG_LENGTH = 4

MARGIN_ACCOUNT_ASSET_OFFSET = 5764
SPREAD_ACCOUNT_ASSET_OFFSET = 2305


class Asset(Enum):
    SOL = 0
    BTC = 1
    ETH = 2
    UNDEFINED = 255


PYTH_PRICE_FEEDS = {
    "localnet": {
        "SOL": PublicKey("2pRCJksgaoKRMqBfa7NTdd6tLYe9wbDFGCcCCZ6si3F7"),
        "BTC": PublicKey("9WD5hzrwEtwbYyZ34BRnrSS11TzD7PTMyszKV5Ur4JxJ"),
        "ETH": PublicKey("FkUZhotvECPTBEXXzxBPjnJu6vPiQmptKyUDSXapBgHJ"),
    },
    "devnet": {
        "SOL": PublicKey("J83w4HKfqxwcq3BEMMkPFSppX3gqekLyLJBexebFVkix"),
        "BTC": PublicKey("HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J"),
        "ETH": PublicKey("EdVCmQ9FSPcVe5YySXDPCRmc8aDQLKJ9xvYBMZPie1Vw"),
    },
    "mainnet": {
        "SOL": PublicKey("H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG"),
        "BTC": PublicKey("GVXRSBjFk6e6J3NbVPXohDJetcTjaeeuykUpbQF8UoMU"),
        "ETH": PublicKey("JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB"),
    },
}

USDC_MINT_ADDRESS = {
    "localnet": PublicKey("6PEh8n3p7BbCTykufbq1nSJYAZvUp6gSwEANAs1ZhsCX"),
    "devnet": PublicKey("6PEh8n3p7BbCTykufbq1nSJYAZvUp6gSwEANAs1ZhsCX"),
    "mainnet": PublicKey("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
}

CLUSTER_URLS = {
    "localnet": "http://127.0.0.1:8899",
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
}

NUM_STRIKES = 11
PRODUCTS_PER_EXPIRY = NUM_STRIKES * 2 + 1
SERIES_FUTURE_INDEX = PRODUCTS_PER_EXPIRY - 1
ACTIVE_EXPIRIES = 2
ACTIVE_MARKETS = ACTIVE_EXPIRIES * PRODUCTS_PER_EXPIRY
TOTAL_EXPIRIES = 6
TOTAL_MARKETS = PRODUCTS_PER_EXPIRY * TOTAL_EXPIRIES

DEFAULT_EXCHANGE_POLL_INTERVAL = 30
DEFAULT_MARKET_POLL_INTERVAL = 5
DEFAULT_CLIENT_POLL_INTERVAL = 20
DEFAULT_CLIENT_TIMER_INTERVAL = 1
UPDATING_STATE_LIMIT_SECONDS = 10

VOLATILITY_POINTS = 5

PLATFORM_PRECISION = 6
PRICING_PRECISION = 12
MARGIN_PRECISION = 8
POSITION_PRECISION = 3

DEFAULT_ORDER_TAG = "SDK"

MAX_POSITION_MOVEMENTS = 10
BPS_DENOMINATOR = 10_000

BID_ORDERS_INDEX = 0
ASK_ORDERS_INDEX = 1

MAX_TOTAL_SPREAD_ACCOUNT_CONTRACTS = 100_000_000


def get_address(encode_str: str, program_id: PublicKey, mint: PublicKey = None):
    if mint is None:
        return PublicKey.find_program_address(
            [
                encode_str.encode()
            ],
            program_id
        )
    else:
        return PublicKey.find_program_address(
            [
                encode_str.encode(),
                bytes(mint)
            ],
            program_id
        )


class SubExchange:
    @property
    def zeta_group(self):
        return self._zeta_group

    @property
    def zeta_group_address(self):
        return self._zeta_group_address

    def __init__(self, asset: Asset, program_id: PublicKey, program: Program):
        self._zeta_group = None
        self.program = program
        self._asset = asset
        underlying_mint = MINTS[asset.name]
        (zetaGroup, _) = get_address(
            "zeta-group",
            program_id,
            underlying_mint
        )
        self._zeta_group_address = zetaGroup

        (greeks, _) = get_address(
            "greeks",
            program_id,
            self._zeta_group_address
        )

        self._greeksAddress = greeks

        (vaultAddress, _) = get_address(
            "vault",
            program_id,
            self._zeta_group_address
        )

        (insuranceVaultAddress, _) = get_address(
            "zeta-insurance-vault",
            program_id,
            self._zeta_group_address
        )

        (socializedLossAccount, _) = get_address(
            "socialized-loss",
            program_id,
            self._zeta_group_address
        )

        self._vaultAddress = vaultAddress
        self._insuranceVaultAddress = insuranceVaultAddress
        self._socializedLossAccountAddress = socializedLossAccount

    async def load(self, asset: Asset):
        logger.info("Loading {} SubExchange", asset.name)
        await self.update_zeta_group()
        # TODO Subexchange loading [1]
        # self.markets = await ZetaGroupMarkets.load(this.asset, opts, 0)

    async def update_zeta_group(self):
        self._zeta_group = await self.program.account


@dataclass
class PythPriceInfo:
    """
    Contains price information.
    Attributes:
        raw_price (int): the raw price
        raw_confidence_interval (int): the raw confidence interval
        price (int): the price
        confidence_interval (int): the price confidence interval
        price_status (PythPriceStatus): the price status
        pub_slot (int): the slot time this price information was published
        exponent (int): the power-of-10 order of the price
    """

    LENGTH: ClassVar[int] = 32

    raw_price: int
    raw_confidence_interval: int
    price_status: PythPriceStatus
    pub_slot: int
    exponent: int

    price: float = field(init=False)
    confidence_interval: float = field(init=False)

    def __post_init__(self):
        self.price = self.raw_price * (10 ** self.exponent)
        self.confidence_interval = self.raw_confidence_interval * \
                                   (10 ** self.exponent)

    @staticmethod
    def deserialise(buffer: bytes, offset: int = 0, *, exponent: int):
        """
        Deserialise the data in the given buffer into a PythPriceInfo object.
        Structure:
            price (i64)
            confidence interval of price (u64)
            status (u32 PythPriceStatus)
            corporate action (u32, currently unused)
            slot (u64)
        """
        # _ is corporate_action
        price, confidence_interval, price_status, _, pub_slot = struct.unpack_from(
            "<qQIIQ", buffer, offset)
        return PythPriceInfo(price, confidence_interval, PythPriceStatus(price_status), pub_slot, exponent)

    def __str__(self) -> str:
        return f"PythPriceInfo status {self.price_status} price {self.price}"

    def __repr__(self) -> str:
        return str(self)


def _read_public_key_or_none(buffer: bytes, offset: int = 0) -> Optional[PublicKey]:
    buffer = buffer[offset:offset + PublicKey.LENGTH]
    if buffer == _NULL_KEY_BYTES:
        return None
    return PublicKey(buffer)


@dataclass
class PythPriceComponent:
    """
    Represents a price component. This is the individual prices each
    publisher sends in addition to their aggregate.
    Attributes:
        publisher_key (SolanaPublicKey): the public key of the publisher
        last_aggregate_price_info (PythPriceInfo): the price information from this
            publisher used in the last aggregate price
        latest_price_info (PythPriceInfo): the latest price information from this
            publisher
        exponent (int): the power-of-10 order for all the raw price information
            in this price component
    """

    LENGTH: ClassVar[int] = PublicKey.LENGTH + 2 * PythPriceInfo.LENGTH

    publisher_key: PublicKey
    last_aggregate_price_info: PythPriceInfo
    latest_price_info: PythPriceInfo
    exponent: int

    @staticmethod
    def deserialise(buffer: bytes, offset: int = 0, *, exponent: int):
        """
        Deserialise the data in the given buffer into a PythPriceComponent object.
        Structure:
            key of quoter (char[32])
            contributing price to last aggregate (PythPriceInfo)
            latest contributing price (PythPriceInfo)
        """
        key = _read_public_key_or_none(buffer, offset)
        if key is None:
            return None
        offset += PublicKey.LENGTH
        last_aggregate_price = PythPriceInfo.deserialise(
            buffer, offset, exponent=exponent)
        offset += PythPriceInfo.LENGTH
        latest_price = PythPriceInfo.deserialise(buffer, offset, exponent=exponent)
        return PythPriceComponent(key, last_aggregate_price, latest_price, exponent)


class RiskCalculator:
    def __init__(self, assets: list[Asset]):
        self._margin_requirements = {}
        for asset in assets:
            self._margin_requirements[asset.name] = ACTIVE_MARKETS


class PriceAccount:
    def __init__(self, product):
        self.product = product
        self.price_type = PythPriceType.UNKNOWN
        self.exponent: Optional[int] = None
        self.num_components: int = 0
        self.last_slot: int = 0
        self.valid_slot: int = 0
        self.product_account_key: Optional[PublicKey] = None
        self.next_price_account_key: Optional[PublicKey] = None
        self.aggregate_price_info: Optional[PythPriceInfo] = None
        self.price_components: List[PythPriceComponent] = []
        self.derivations: Dict[EmaType, int] = {}
        self.timestamp: int = 0  # unix timestamp in seconds
        self.min_publishers: Optional[int] = None
        self.prev_timestamp: int = 0  # unix timestamp in seconds

    def update_from(self, buffer: bytes, *, version: int, offset: int = 0) -> None:
        """
        Update the data in this price account from the given buffer.
        Structure:
            price type (u32 PythPriceType)
            exponent (i32)
            number of component prices (u32)
                (? unclear if this is supposed to match the number of
                    PythPriceComponents below)
            unused (u32)
            currently accumulating price slot (u64)
            slot of current aggregate price (u64)
            derivations (u64[6] - array index corresponds to (DeriveType - 1) - v2 only)
            unused derivation values and minimum publishers (u64[2], i32[2], )
            product account key (char[32])
            next price account key (char[32])
            account key of quoter who computed last aggregate price (char[32])
            aggregate price info (PythPriceInfo)
            price components (PythPriceComponent[up to 16 (v1) / up to 32 (v2)])
        """
        if version == _VERSION_2:
            price_type, exponent, num_components = struct.unpack_from("<IiI", buffer, offset)
            offset += 16  # struct.calcsize("IiII") (last I is the number of quoters that make up the aggregate)
            last_slot, valid_slot = struct.unpack_from("<QQ", buffer, offset)
            offset += 16  # struct.calcsize("QQ")
            derivations = list(struct.unpack_from("<6q", buffer, offset))
            self.derivations = dict((type_, derivations[type_.value - 1]) for type_ in
                                    [EmaType.EMA_CONFIDENCE_VALUE, EmaType.EMA_PRICE_VALUE])
            offset += 48  # struct.calcsize("6q")
            # drv[2-4]_ fields are currently unused
            timestamp, min_publishers = struct.unpack_from("<qB", buffer, offset)
            offset += 16  # struct.calcsize("qBbhi") ("bhi" is drv_2, drv_3, drv_4)
            product_account_key_bytes, next_price_account_key_bytes = struct.unpack_from("32s32s", buffer, offset)
            offset += 88  # struct.calcsize("32s32sQqQ") ("QqQ" is prev_slot, prev_price, prev_conf)
            prev_timestamp = struct.unpack_from("<q", buffer, offset)[0]
            offset += 8  # struct.calcsize("q")
            self.timestamp = timestamp
            self.min_publishers = min_publishers
            self.prev_timestamp = prev_timestamp
        elif version == _VERSION_1:
            price_type, exponent, num_components, _, last_slot, valid_slot, product_account_key_bytes, next_price_account_key_bytes, aggregator_key_bytes = struct.unpack_from(
                "<IiIIQQ32s32s32s", buffer, offset)
            self.derivations = {}
            offset += 128  # struct.calcsize("<IiIIQQ32s32s32s")
        else:
            assert False

        # aggregate price info (PythPriceInfo)
        aggregate_price_info = PythPriceInfo.deserialise(
            buffer, offset, exponent=exponent)

        # price components (PythPriceComponent[up to 16 (v1) / up to 32 (v2)])
        price_components: List[PythPriceComponent] = []
        offset += PythPriceInfo.LENGTH
        buffer_len = len(buffer)
        while offset < buffer_len:
            component = PythPriceComponent.deserialise(
                buffer, offset, exponent=exponent)
            if not component:
                break
            price_components.append(component)
            offset += PythPriceComponent.LENGTH

        self.price_type = PythPriceType(price_type)
        self.exponent = exponent
        self.num_components = num_components
        self.last_slot = last_slot
        self.valid_slot = valid_slot
        self.product_account_key = PublicKey(product_account_key_bytes)
        self.next_price_account_key = _read_public_key_or_none(
            next_price_account_key_bytes)
        self.aggregate_price_info = aggregate_price_info
        self.price_components = price_components


class Exchange:
    def __init__(self, assets: list[Asset],
                 program_key: str,
                 network: str,
                 client: AsyncClient,
                 wallet=Wallet.dummy()):
        self._poll_interval = DEFAULT_EXCHANGE_POLL_INTERVAL
        self.clockSlot = None
        self.program_id = PublicKey(program_key)
        self._assets = assets
        self._network = network
        self._provider = Provider(client, wallet)
        self._sub_exchanges = {}
        idl_path = os.path.join(current_dir, "idl/zeta.json")
        with Path(idl_path).open() as f:
            raw_idl = json.load(f)
        idl = Idl.from_json(raw_idl)
        self._program = Program(idl, program_id=self.program_id, provider=self._provider)
        for asset in assets:
            self._sub_exchanges[asset.name] = SubExchange(asset, self.program_id, self._program)
        self._riskCalculator = RiskCalculator(assets)
        (mintAuthority, _) = get_address(
            "mint-auth",
            self.program_id
        )
        (state, _) = get_address(
            "state",
            self.program_id
        )
        (serumAuthority, _) = get_address(
            "serum",
            self.program_id
        )

        self._mintAuthority = mintAuthority
        self._stateAddress = state
        self._serumAuthority = serumAuthority
        self._usdcMintAddress = USDC_MINT_ADDRESS[network]
        self.clockTimestamp = None

        (treasuryWallet, _) = get_address(
            "zeta-treasury-wallet",
            self.program_id,
            self._usdcMintAddress
        )

        self._treasuryWalletAddress = treasuryWallet

        (referralsRewardsWallet, _) = get_address(
            "zeta-referrals-rewards-wallet",
            self.program_id,
            self._usdcMintAddress
        )

        self._referralsRewardsWalletAddress = referralsRewardsWallet

        self._lastPollTimestamp = 0

        self._oracle = Oracle(self, network, client)

    def set_clock_data(self, data: bytes):
        slot, _epochStartTimestamp, _epoch, _leaderScheduleEpoch, unix_timestamp = struct.unpack_from("<QqQQq", data, 0)
        self.clockTimestamp = unix_timestamp
        self.clockSlot = slot

    def subscribe_clock(self, callback):
        logger.info("Subscribing to clock")
        wss = CLUSTER_URLS[self._network].replace("https", "wss")
        ws_run = Thread(target=asyncio.run,
                        args=(generic_subscriber(wss, SYSVAR_CLOCK_PUBKEY, self.set_clock_data),))
        ws_run.start()
        if self.clockTimestamp and self.clockTimestamp > self._lastPollTimestamp + self._poll_interval:
            self._lastPollTimestamp = self.clockTimestamp
            # TODO handle subexchange polling [Deferred]

    async def load(self, callback):
        await self.subscribe_oracle(self._assets, callback)
        #
        # await Promise.all(
        #     self.assets.map(async(asset) => {
        #     await self.getSubExchange(asset).load(
        #         asset,
        #         self.program_id,
        #         self.network,
        #         self.opts,
        #         throttleMs,
        #         callback
        #     )
        # })
        # )
        #
        # await Promise.all(
        #     self.assets.map(async(asset) => {
        #     self._markets = self._markets.concat(self.getMarkets(asset))
        # })
        # )
        #
        # await self.updateState()
        self.subscribe_clock(None)

    def display_state(self):
        for asset, sub_exchange in self._sub_exchanges.items():
            ordered_indexes = [
                sub_exchange._zeta_group
            ]

    async def subscribe_oracle(self, assets, callback):
        await self._oracle.subscribe_price_feeds(assets, callback)


def _parse_header(buffer: bytes, offset: int = 0):
    if len(buffer) - offset < _ACCOUNT_HEADER_BYTES:
        raise ValueError("Pyth account data too short")

    # Pyth magic (u32) == MAGIC
    # version (u32) == VERSION_1 or 2
    # account type (u32)
    # account data size (u32)
    magic, version, type_, size = struct.unpack_from("<IIII", buffer, offset)

    if len(buffer) < size:
        raise ValueError(
            f"Pyth header says data is {size} bytes, but buffer only has {len(buffer)} bytes")

    if magic != _MAGIC:
        raise ValueError(
            f"Pyth account data header has wrong magic: expected {_MAGIC:08x}, got {magic:08x}")

    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Pyth account data has unsupported version {version}")

    return size, version


async def generic_subscriber(wss, address, callback, added_params=(), sub_ids=None, sub=None):
    if sub_ids is None:
        sub_ids = {}
    async with connect(wss) as websocket:
        websocket: solana.rpc.websocket_api.SolanaWsClientProtocol
        await websocket.account_subscribe(address, encoding="base64")
        first_resp = await websocket.recv()
        subscription_id = first_resp.result
        sub_ids[sub] = subscription_id
        msg: solana.rpc.websocket_api.AccountNotification
        async for msg in websocket:
            # if not subscriptions_alive:
            #     break
            data_base64, _data_format = msg.result.value.data
            data = base64.b64decode(data_base64)
            callback(data, *added_params)
        await websocket.logs_unsubscribe(subscription_id)
        # next_resp = await websocket.recv()
        # await websocket.logs_unsubscribe(subscription_id)


class Oracle:
    def __init__(self, exchange: Exchange, network: str, client: AsyncClient):
        self._callback = None
        self._network = network
        self._exchange = exchange
        self._client = client
        self.subscription_ids = {}
        self._data = {}
        self.callback = None
        self.subscriptions_alive = True

    async def subscribe_price_feeds(self, asset_list: list[Asset], callback):
        self._callback = callback
        # tasks = []
        for asset in asset_list:
            logger.info("Oracle subscribing to feed {}", asset.name)
            price_address = PYTH_PRICE_FEEDS[self._network][asset.name]
            wss = CLUSTER_URLS[self._network].replace("https", "wss")
            added_params = (asset, self)
            ws_run = Thread(target=asyncio.run,
                            args=(
                                generic_subscriber(wss, price_address, price_callback, added_params,
                                                   self.subscription_ids,
                                                   asset),))
            ws_run.start()
            # await self.run_forever(asset)
            # tasks.append(task)


def price_callback(data, asset, self: Oracle):
    exchange = self._exchange
    size, version = _parse_header(data, 0)
    price_acc = PriceAccount(asset.name)
    price_acc.update_from(data[:size], version=version, offset=_ACCOUNT_HEADER_BYTES)
    price_info = price_acc.aggregate_price_info
    oracle_data = {
        asset.name: asset.name,
        "price": price_info.price,
        "lastUpdatedTime": exchange.clockTimestamp,
        "lastUpdatedSlot": price_info.pub_slot
    }
    self._data[asset] = oracle_data
    if self.callback is not None:
        self.callback(asset, oracle_data)


def cli():
    # asyncio.run(example())
    print("Welcome to Zeta!")


async def get_open_orders(program_id, market, user_key):
    return PublicKey.find_program_address(
        [
            bytes("open-orders", "utf-8"),
            bytes(DEX_PID[Exchange_network]),
            bytes(market),
            bytes(user_key)
        ],
        program_id
    )


async def get_open_orders_map(program_id: PublicKey, open_orders: PublicKey):
    return PublicKey.find_program_address(
        [bytes(open_orders)],
        program_id
    )


async def initialize_open_orders_ix(asset, market, user_key, margin_account):
    open_orders_pda, _open_orders_nonce = await get_open_orders(
        Exchange_program_id,
        market,
        user_key
    )

    open_orders_map, _open_orders_map_nonce = await get_open_orders_map(
        Exchange_program_id,
        open_orders_pda
    )

    return [
        initialize_open_orders({
            "state": PublicKey("9VddCF6iEyZbjkCQ4g8VJpjEtuLsgmvRCc6LQwAvXigC"),
            # Exchange.state_address,
            "zeta_group": PublicKey("CcLF7qQbgRQqUDmQeEkTSP2UbX82N9G91THjV5uRGCMW"),
            # Exchange.get_zeta_group_address(asset),
            "dex_program": DEX_PID[Exchange_network],
            "system_program": SYS_PROGRAM_ID,
            "open_orders": open_orders_pda,
            "margin_account": margin_account,
            "authority": user_key,
            "payer": user_key,
            "market": market,
            "rent": SYSVAR_RENT_PUBKEY,
            "serum_authority": PublicKey("7m134nU79ezr3b7jYSehnhto94AvSP4yexa3Wdhxff99"),
            # Exchange._serum_authority,
            "open_orders_map": open_orders_map
        }),
        open_orders_pda
    ]


async def process_transaction(provider: Provider, tx: Transaction, signers=None, opts=None, use_ledger=False):
    # txSig = TransactionSignature()
    blockhash = await provider.connection.get_recent_blockhash()
    tx.recent_blockhash = blockhash["result"]["value"]["blockhash"]
    # Exchange.ledger_wallet.public_key if use_ledger else
    tx.fee_payer = provider.wallet.public_key
    if signers is None:
        signers = []

    for s in signers:
        if s is not None:
            tx.sign_partial(s)

        # if use_ledger:
        #     tx = await Exchange.ledger_wallet.sign_transaction(tx)
        # else:
    tx = await provider.wallet.sign_transaction(tx)

    try:
        tx_sig = await solana.rpc.api.Client.send_raw_transaction(tx.serialize(), opts)
        return tx_sig
    except:
        raise Exception("Error in process_transaction")


def to_program_side(side):
    if side == Side.BID:
        return {"Bid": {}}
    if side == Side.ASK:
        return {"Ask": {}}
    raise Exception("Invalid side")


def to_program_order_type(orderType):
    if orderType == OrderType.LIMIT:
        return {"Limit": {}}
    if orderType == OrderType.POSTONLY:
        return {"PostOnly": {}}
    if orderType == OrderType.FILLORKILL:
        return {"fillOrKill": {}}


def place_order_v3_ix(asset, market_index, price, size, side, order_type, client_order_id, tag, margin_account,
                      authority, open_orders, whitelist_trading_fees_account):
    if len(tag) > MAX_ORDER_TAG_LENGTH:
        raise Exception("Tag is too long! Max length = " + str(MAX_ORDER_TAG_LENGTH))

    # sub_exchange = Exchange.get_sub_exchange(asset)
    # market_data = sub_exchange.markets.markets[market_index]
    remaining_accounts = [
        {
            "pubkey": whitelist_trading_fees_account,
            "is_signer": False,
            "is_writable": False,
        },
    ] if whitelist_trading_fees_account is None else []

    return pyzeta.instructions.place_order_v3(
        {
            "price": price,
            "size": size,
            "side": pyzeta.types.side.from_decoded(to_program_side(side)),
            "order_type": pyzeta.types.order_type.from_decoded(to_program_order_type(order_type)),
            "client_order_id": None if client_order_id == 0 else client_order_id,
            "tag": "SDK"
        },
        {
            "state": PublicKey("9VddCF6iEyZbjkCQ4g8VJpjEtuLsgmvRCc6LQwAvXigC"),
            # Exchange.state_address,
            "zeta_group": PublicKey("CcLF7qQbgRQqUDmQeEkTSP2UbX82N9G91THjV5uRGCMW"),
            # "zeta_group": sub_exchange.zeta_group_address,
            "margin_account": margin_account,
            "authority": authority,
            "dex_program": DEX_PID[Exchange_network],
            "token_program": TOKEN_PROGRAM_ID,
            "serum_authority": PublicKey("7m134nU79ezr3b7jYSehnhto94AvSP4yexa3Wdhxff99"),
            # Exchange._serum_authority,
            "greeks": PublicKey("4WKZv9ptg7zc9WCdyK7y5UfJsr3ACPeBiqWXirDQrDhV"),
            # sub_exchange.zeta_group.greeks,
            "open_orders": open_orders,
            "rent": SYSVAR_RENT_PUBKEY,
            "market_accounts": {
                "asks": PublicKey("59eaBiRzude5h8vwx8SkPMWAGad7ag3g9pErbyiqV55F"),
                "bids": PublicKey("9dFLWFpWdXbpWcgfVRXqPzFBrYNqV9zs83Hd3QZyzGWF"),
                "coin_vault": PublicKey("Fck8esUdZ3Gku4Uy3rgFZyy8bqhaAL5iKPpCR69PMMBc"),
                "coin_wallet": PublicKey("Bb5yvLhfD7JFXdZUfhnXpUwgmCo4ZsAyrBBY6rZuVJaN"),
                "event_queue": PublicKey("2ExH97tjdj25Hi5JQHGx3jgMUnxhMjpkdyw3GvC7Kd4N"),
                "market": PublicKey("2YJaZTe5FGDbjbXmKLMw28L7ZhsgaxrFbGmN6Wdc42UN"),
                "order_payer_token_account": PublicKey(
                    "D4vzFUHHCkv6oN6gFyYzt9wjXjbPRmyXGzwioDo367Bz") if Side.BID else PublicKey(
                    "Bb5yvLhfD7JFXdZUfhnXpUwgmCo4ZsAyrBBY6rZuVJaN"),
                "pc_vault": PublicKey("46tATCNejhV5d62XTwzHcmUTaLL6ftoErojTygvCk4Vx"),
                "pc_wallet": PublicKey("D4vzFUHHCkv6oN6gFyYzt9wjXjbPRmyXGzwioDo367Bz"),
                "request_queue": PublicKey("6pfFXyNovSso6g57rojKD6d6RLSn1motbP4AevpEwR7L"),
            },
            #     {
            #     "market": market_data.serum_market.decoded.own_address,
            #     "request_queue": market_data.serum_market.decoded.request_queue,
            #     "event_queue": market_data.serum_market.decoded.event_queue,
            #     "bids": market_data.serum_market.decoded.bids,
            #     "asks": market_data.serum_market.decoded.asks,
            #     "coin_vault": market_data.serum_market.decoded.base_vault,
            #     "pc_vault": market_data.serum_market.decoded.quote_vault,
            #     "order_payer_token_account": market_data.quote_vault if Side.BID else market_data.base_vault,
            #     "coin_wallet": market_data.base_vault,
            #     "pc_wallet": market_data.quote_vault,
            # },

            "oracle": PublicKey("HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J"),
            # sub_exchange.zeta_group.oracle,
            "market_node": PublicKey("2oR2TGmRFVMWNCVSJ9zQGGm64PcY4LbwyCSNHjQ7VGUH"),
            # sub_exchange.greeks.node_keys[market_index]
            "market_mint": PublicKey(
                "HTq7MqyG5t9ifByHabjtDndbRtA67dQ8YovhiAbb69h5") if side == Side.BID else PublicKey(
                "C8m11wQ89ugMUDdm1gjgE7vsHF1nP9AN1x886qYHjXQK"),
            # market_data.serum_market.quote_mint_address
            # market_data.serum_market.base_mint_address,
            "mint_authority": PublicKey("278LYD3sfoYF9tbcYf2HatzGYGu8VuQ7SjpSFwAgGMY2"),
            # Exchange._mint_authority,
        },
    )


# class Client:
#     def __init__(self, connection, wallet: Wallet, opts):
#         self._provider = Provider(connection, wallet, opts)
#         self._subclients = {}
#         self._referral_account = None
#         self._referrer_account = None
#         self._referrer_account = None
#         self._whitelist_deposit_address = None
#         self._whitelist_trading_fees_address = None
#         self._usdc_account_address = None
#         self._referral_alias = None

# async def place_order(self, asset, market, price, size, side, type, client_order_id, tag):
#     ret = await self.get_sub_client(asset).place_order_v3(
#         self.market_identifier_to_public_key(asset, market),
#         price,
#         size,
#         side,
#         type,
#         client_order_id,
#         tag
#     )
#     return ret


# class SubClient:
#     def __init__(self, asset, parent):
#         self._asset = asset
#         # self._sub_exchange = Exchange.get_sub_exchange(self, asset)
#         self._open_orders_accounts = []
#         for i in range(len(self._sub_exchange.zeta_group.products)):
#             self._open_orders_accounts.append(PublicKey("11111111111111111111111111111111"))
#         self._parent = parent
#
#         self._margin_positions = []
#         self._spread_positions = []
#         self._orders = []
#         self._last_update_timestamp = 0
#         self._pending_update = False
#         self._margin_account = None
#         self._margin_account_address = None
#         self._spread_account = None
#         self._poll_interval = DEFAULT_CLIENT_POLL_INTERVAL
#         self._updating_state = False
#         self._updating_state_timestamp = None
#         self._spread_account_address = None
#         self._callback = None
#         self._connection = None
#         self._pending_update_slot = 0
_open_orders_accounts = [PublicKey("11111111111111111111111111111111")] * 46


# len(Exchange.zetaGroup.products)


async def place_order_v3(provider, market, price, size, side, order_type, client_order_id, tag, market_index):
    tx = Transaction()
    # market_index = self._sub_exchange.markets.get_market_index(market)
    open_orders_pda = None
    if _open_orders_accounts[market_index] == PublicKey("11111111111111111111111111111111"):
        print("User doesn't have open orders account for this asset. Initializing for market: " + str(market))
        init_ix, _open_orders_pda = await initialize_open_orders_ix(
            # self._asset,
            None,
            market,
            # TODO ! ASK what to set here
            PublicKey("2cBRcusmXXQeoMUte4DxqXEThM4cTrNPQ1VsBZwZ9kcM"),
            # self._parent.public_key(),
            PublicKey("D84vvCFFSQBfeiT89fyR9GP3hDEas7MWveZwwsiG93Kx"),
            # self._margin_account_address
        )
        open_orders_pda = _open_orders_pda
        tx.add(init_ix)
    else:
        open_orders_pda = _open_orders_accounts[market_index]

    order_ix = place_order_v3_ix(
        # self._asset,
        None,
        market_index,
        price,
        size,
        side,
        order_type,
        client_order_id,
        tag,
        PublicKey("D84vvCFFSQBfeiT89fyR9GP3hDEas7MWveZwwsiG93Kx"),
        # self._margin_account_address,
        PublicKey("2cBRcusmXXQeoMUte4DxqXEThM4cTrNPQ1VsBZwZ9kcM"),
        # self._parent.public_key(),
        open_orders_pda,
        # TODO ASK what is this
        None
        # self._parent.whitelist_trading_fees_address
    )

    tx.add(order_ix)

    txId = await process_transaction(provider, tx)
    _open_orders_accounts[market_index] = open_orders_pda
    return txId


async def place_order(provider, asset, market, price, size, side, type, client_order_id=0, tag="SDK"):
    ret = await place_order_v3(
        # self.market_identifier_to_public_key(asset, market),
        provider,
        PublicKey("2YJaZTe5FGDbjbXmKLMw28L7ZhsgaxrFbGmN6Wdc42UN"),
        price,
        size,
        side,
        type,
        client_order_id,
        tag,
        market
    )
    return ret
