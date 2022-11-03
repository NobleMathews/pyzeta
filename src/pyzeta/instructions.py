from __future__ import annotations

import typing

import borsh_construct as borsh
from solana.publickey import PublicKey
from solana.transaction import TransactionInstruction, AccountMeta

from pyzeta.types import side, order_type

PROGRAM_ID = PublicKey("BG3oRikW8d16YjUEmX3ZxHm9SiJzrGtMhsSR8aCw1Cd7")


class PlaceOrderV3Args(typing.TypedDict):
    price: int
    size: int
    side: side.SideKind
    order_type: order_type.OrderTypeKind
    client_order_id: typing.Optional[int]
    tag: typing.Optional[str]


layout = borsh.CStruct(
    "price" / borsh.U64,
    "size" / borsh.U64,
    "side" / side.layout,
    "order_type" / order_type.layout,
    "client_order_id" / borsh.Option(borsh.U64),
    "tag" / borsh.Option(borsh.String),
)


class PlaceOrderV3Accounts(typing.TypedDict):
    state: PublicKey
    zeta_group: PublicKey
    margin_account: PublicKey
    authority: PublicKey
    dex_program: PublicKey
    token_program: PublicKey
    serum_authority: PublicKey
    greeks: PublicKey
    open_orders: PublicKey
    rent: PublicKey
    market_accounts: MarketAccountsNested
    oracle: PublicKey
    market_node: PublicKey
    market_mint: PublicKey
    mint_authority: PublicKey


class MarketAccountsNested(typing.TypedDict):
    market: PublicKey
    request_queue: PublicKey
    event_queue: PublicKey
    bids: PublicKey
    asks: PublicKey
    order_payer_token_account: PublicKey
    coin_vault: PublicKey
    pc_vault: PublicKey
    coin_wallet: PublicKey
    pc_wallet: PublicKey


def place_order_v3(
        args: PlaceOrderV3Args, accounts: PlaceOrderV3Accounts
) -> TransactionInstruction:
    keys: list[AccountMeta] = [
        AccountMeta(pubkey=accounts["state"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["zeta_group"], is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=accounts["margin_account"], is_signer=False, is_writable=True
        ),
        AccountMeta(pubkey=accounts["authority"], is_signer=True, is_writable=False),
        AccountMeta(pubkey=accounts["dex_program"], is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=accounts["token_program"], is_signer=False, is_writable=False
        ),
        AccountMeta(
            pubkey=accounts["serum_authority"], is_signer=False, is_writable=False
        ),
        AccountMeta(pubkey=accounts["greeks"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["open_orders"], is_signer=False, is_writable=True),
        AccountMeta(pubkey=accounts["rent"], is_signer=False, is_writable=False),
        AccountMeta(
            pubkey=accounts["market_accounts"]["market"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["request_queue"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["event_queue"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["bids"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["asks"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["order_payer_token_account"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["coin_vault"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["pc_vault"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["coin_wallet"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["market_accounts"]["pc_wallet"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(pubkey=accounts["oracle"], is_signer=False, is_writable=False),
        AccountMeta(pubkey=accounts["market_node"], is_signer=False, is_writable=True),
        AccountMeta(pubkey=accounts["market_mint"], is_signer=False, is_writable=True),
        AccountMeta(
            pubkey=accounts["mint_authority"], is_signer=False, is_writable=False
        ),
    ]
    identifier = b"\x92]\x0e\xa7\x9f\x14\x06:"
    encoded_args = layout.build(
        {
            "price": args["price"],
            "size": args["size"],
            "side": args["side"].to_encodable(),
            "order_type": args["order_type"].to_encodable(),
            "client_order_id": args["client_order_id"],
            "tag": args["tag"],
        }
    )
    data = identifier + encoded_args
    return TransactionInstruction(keys, PROGRAM_ID, data)
