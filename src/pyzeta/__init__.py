import asyncio
import json
import os
from pathlib import Path
import enum

import solana.rpc.websocket_api
from loguru import logger

from anchorpy import Idl, Program, Provider, Wallet
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.websocket_api import connect

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


class Asset(enum.Enum):
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
    def __init__(self, asset: Asset, program_id: PublicKey, program: Program):
        self.program = program
        self._asset = asset
        underlying_mint = MINTS[asset.name]
        (zetaGroup, _) = get_address(
            "zeta-group",
            program_id,
            underlying_mint
        )
        self._zetaGroupAddress = zetaGroup

        (greeks, _) = get_address(
            "greeks",
            program_id,
            self._zetaGroupAddress
        )

        self._greeksAddress = greeks

        (vaultAddress, _) = get_address(
            "vault",
            program_id,
            self._zetaGroupAddress
        )

        (insuranceVaultAddress, _) = get_address(
            "zeta-insurance-vault",
            program_id,
            self._zetaGroupAddress
        )

        (socializedLossAccount, _) = get_address(
            "socialized-loss",
            program_id,
            self._zetaGroupAddress
        )

        self._vaultAddress = vaultAddress
        self._insuranceVaultAddress = insuranceVaultAddress
        self._socializedLossAccountAddress = socializedLossAccount

    async def load(self, asset:Asset):
        logger.info("Loading {} SubExchange", asset.name)
        await self.update_zeta_group()

    async def update_zeta_group(self):
        self._zetaGroup = await self.program.account


class RiskCalculator:
    def __init__(self, assets: list[Asset]):
        self._margin_requirements = {}
        for asset in assets:
            self._margin_requirements[asset.name] = ACTIVE_MARKETS

class Oracle:
    def __init__(self, network: str, client: AsyncClient):
        self._network = network
        self._client = client
        self.subscription_ids = {}
        self._data = {}
        self.callback = None

    async def subscribe_price_feeds(self, asset_list: list[Asset], callback):
        self._callback = callback
        # tasks = []
        for asset in asset_list:
            logger.info("Oracle subscribing to feed {}", asset.name)
            price_address = PYTH_PRICE_FEEDS[self._network][asset.name]
            async with connect(CLUSTER_URLS[self._network].replace("https", "wss")) as websocket:
                websocket: solana.rpc.websocket_api.SolanaWsClientProtocol
                await websocket.account_subscribe(price_address)
                first_resp = await websocket.recv()
                subscription_id = first_resp.result
                next_resp = await websocket.recv()
                await websocket.logs_unsubscribe(subscription_id)
            # tasks.append(task)


class Exchange:
    def __init__(self, assets: list[Asset],
                 program_key: str,
                 network: str,
                 client: AsyncClient,
                 wallet= Wallet.dummy()):
        self.program_id = PublicKey(program_key)
        self._assets = assets
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
        
        self._oracle = Oracle(network, client)

    async def load(self):
        await self.subscribe_oracle(self._assets, None)
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
        # await self.subscribeClock(callback)

    def display_state(self):
        for asset, sub_exchange in self._sub_exchanges.items():
            ordered_indexes = [
                sub_exchange
            ]

    async def subscribe_oracle(self, assets, callback):
        await self._oracle.subscribe_price_feeds(assets, callback)


def cli():
    # asyncio.run(example())
    print("Welcome to Zeta!")
