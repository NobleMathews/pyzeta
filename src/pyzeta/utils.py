import math
from decimal import Decimal
from typing import Dict

import my_client.accounts
import solana
import solana.rpc.api
from anchorpy import Provider, EventParser, Event
from solana.publickey import PublicKey
from solana.transaction import Transaction, TransactionSignature, TransactionInstruction
from spl.token._layouts import ACCOUNT_LAYOUT as TOKEN_ACCOUNT_LAYOUT
from spl.token.constants import TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID

import constants


def default_commitment() -> Dict:
    return {
        "skipPreflight": False,
        "preflightCommitment": "confirmed",
        "commitment": "confirmed",
    }


def get_vault(program_id: PublicKey, zeta_group: PublicKey):
    return PublicKey.find_program_address(
        [bytes("vault", "utf-8"), bytes(zeta_group)], program_id
    )


def get_serum_vault(program_id: PublicKey, zeta_group):
    """_summary_

    Parameters
    ----------
    program_id : _type_
            _description_
    zeta_group : _type_
            _description_
    """
    pass


def get_mint_authority(program_id: PublicKey):
    return PublicKey.find_program_address(
        [bytes("mint-auth", "utf-8")], program_id
    )


def get_serum_authority(program_id: PublicKey):
    return PublicKey.find_program_address(
        [bytes("serum", "utf-8")], program_id
    )


def get_zeta_group(
        program_id: PublicKey, mint: PublicKey
):
    """_summary_

    Parameters
    ----------
    program_id : solana.publickey.PublicKey
            _description_
    mint : solana.publickey.PublicKey
            _description_

    Returns
    -------
    _type_
            _description_
    """
    return PublicKey.find_program_address(
        [bytes("zeta-group", "utf-8"), mint], program_id
    )


def get_zeta_vault(program_id: PublicKey, mint: PublicKey):
    return PublicKey.find_program_address(
        [
            bytes("zeta-vault", "utf-8"),
            bytes(mint)
        ],
        program_id
    )


def get_zeta_insurance_vault(program_id: PublicKey, zeta_group_address: PublicKey):
    return PublicKey.find_program_address(
        [bytes("zeta-insurance-vault", "utf-8"), bytes(zeta_group_address)], program_id
    )


def get_socialized_loss_account(program_id: PublicKey, zeta_group_address: PublicKey):
    return PublicKey.find_program_address(
        [bytes("socialized-loss", "utf-8"), bytes(zeta_group_address)], program_id
    )


def get_user_whitelist_deposit_account():
    pass


def get_state(program_id):
    return PublicKey.find_program_address(
        [bytes("state", "utf-8")], program_id
    )


def get_market_uninitialized():
    pass


def get_underlying(program_id: PublicKey, underlying_index: int):
    return PublicKey.find_program_address(
        [bytes("underlying", "utf-8"), bytes([underlying_index])], program_id
    )


def convert_native_lot_size_to_decimal(amount):
    """
    Converts a native lot size where 1 unit = 0.001 lots to human readable decimal
    """
    return amount / math.pow(10, constants.POSITION_PRECISION)


def convert_decimal_to_native_integer(amount):
    return Decimal(amount)


def get_most_recent_expired_index():
    from exchange import Exchange
    if Exchange._markets.front_expiry_index - 1 < 0:
        return constants.ACTIVE_EXPIRIES - 1
    else:
        return Exchange._markets.front_expiry_index - 1


def display_state():
    from exchange import Exchange
    ordered_indexes = [
        Exchange._zeta_group.front_expiry_index,
        get_most_recent_expired_index()

    ]
    print("[EXCHANGE] Display market state...")


def get_margin_account(program_id: PublicKey, zeta_group: PublicKey, user_key: PublicKey):
    return PublicKey.find_program_address(
        [
            bytes("margin", "utf-8"),
            bytes(zeta_group),
            bytes(user_key),
        ],
        program_id,
    )


def get_associated_token_address(mint: PublicKey, owner: PublicKey):
    # TODO(J0): Figure out how to properly import Addresses below from SPL
    return PublicKey.find_program_address(
        [
            bytes(owner),
            bytes(PublicKey(TOKEN_PROGRAM_ID)),
            bytes(mint),
        ],
        PublicKey(ASSOCIATED_TOKEN_PROGRAM_ID),
    )[0]


def get_zeta_group(program_id: PublicKey, mint: PublicKey):
    print("getting zeta group")
    print(program_id)
    return PublicKey.find_program_address(
        [
            bytes("zeta-group", "utf-8"),
            bytes(mint),
        ],
        program_id
    )


def get_greeks(program_id: PublicKey, zeta_group: PublicKey):
    return PublicKey.find_program_address(
        [
            bytes("greeks", "utf-8"),
            bytes(zeta_group),
        ],
        program_id
    )


def get_underlying(program_id: PublicKey, underlying_index: int):
    return PublicKey.find_program_address(
        [
            bytes("underlying", "utf-8"),
            bytes([underlying_index]),
        ],
        program_id
    )


async def process_transaction(provider: Provider, tx: Transaction, signers=None, opts=None, use_ledger=False):
    from exchange import Exchange
    txSig = TransactionSignature()
    blockhash = await provider.connection.get_recent_blockhash()
    tx.recent_blockhash = blockhash
    tx.fee_payer = Exchange.ledger_wallet.public_key if use_ledger else provider.wallet.public_key
    if signers == None:
        signers = []

    for s in signers:
        if s != None:
            tx.sign_partial(s)

    if use_ledger:
        tx = await Exchange.ledger_wallet.sign_transaction(tx)
    else:
        tx = await provider.wallet.sign_transaction(tx)

    try:
        tx_sig = await solana.rpc.api.Client.send_raw_transaction(tx.serialize(), opts)
        return tx_sig
    except:
        raise Exception("Error in process_transaction")


def get_spread_account(program_id, zeta_group, user_key):
    return PublicKey.find_program_address(
        [
            bytes("spread", "utf-8"),
            bytes(zeta_group),
            bytes(user_key)
        ],
        program_id
    )


async def get_zeta_treasury_wallet(program_id, mint):
    return PublicKey.find_program_address(
        [
            bytes("zeta-treasury-wallet", "utf-8"),
            bytes(mint)
        ],
        program_id
    )


async def get_user_whitelist_deposit_account(program_id, user_key):
    return await PublicKey.find_program_address(
        [
            bytes("whitelist-deposit", "utf-8"),
            bytes(user_key)
        ],
        program_id
    )


async def get_user_whitelist_trading_fees_account(program_id, user_key):
    return await PublicKey.find_program_address(
        [
            bytes("whitelist-trading-fees", "utf-8"),
            bytes(user_key)
        ],
        program_id
    )


async def get_referrer_account_address(program_id, referrer):
    return await PublicKey.find_program_address(
        [
            bytes("referrer", "utf-8"),
            bytes(referrer)
        ],
        program_id
    )


async def fetch_referrer_alias_account(referrer, alias):
    if not referrer and not alias:
        return None

    referrer_aliases = await my_client.accounts.referrer_alias.ReferrerAlias.alias
    for i in range(len(referrer_aliases)):
        acc = referrer_aliases[i].account
        if (referrer and acc.referrer == referrer) or (alias and convert_buffer_to_trimmed_string(acc.alias) == alias):
            return acc
    return None


def convert_buffer_to_trimmed_string(buffer: list[int]):
    buffer_string = str(bytes(buffer))
    split_index = len(buffer_string)

    for index in range(len(buffer_string)):
        if buffer_string[index] == 0:
            split_index = index
            break
    return buffer_string[0:split_index]


async def get_referral_account_address(program_id, user):
    return await PublicKey.find_program_address(
        [
            bytes("referral", "utf-8"),
            bytes(user)
        ],
        program_id
    )


async def get_serum_vault_owner_and_nonce(market: PublicKey, dex_pid: PublicKey):
    nonce = 0
    while nonce < 255:
        try:
            vault_owner = await PublicKey.create_program_address(
                [bytes(market), nonce.to_bytes(8, "small")],
                dex_pid
            )
            return vault_owner, nonce
        except:
            nonce += 1
    raise Exception("Unable to find nonce")


def split_ixs_into_txs(ixs: TransactionInstruction, ixs_per_tx: int):
    txs: list[Transaction] = []
    i = 0
    while i < len(ixs):
        tx = Transaction()
        for _ in range(ixs_per_tx):
            tx.add(ixs[i])
            i += 1
        txs.append(tx)
    return txs


def split_ixs_into_tx(ixs: TransactionInstruction, ixs_per_tx: int):
    txs: list[Transaction] = []
    i = 0
    while i < len(ixs):
        tx = Transaction()
        for _ in range(ixs_per_tx):
            tx.add(ixs[i])
            i += 1
        txs.append(tx)
    return txs


async def create_open_orders_address(program_id: PublicKey, market: PublicKey, user_key: PublicKey, nonce: int):
    from exchange import Exchange
    return await PublicKey.create_program_address(
        [
            bytes("open-order", "utf-8"),
            bytes(constants.DEX_PID[Exchange.network]),
            bytes(market),
            bytes(user_key),
            bytes([nonce])
        ]
    )


async def simulate_transaction(provider: Provider, tx: Transaction):
    from exchange import Exchange
    response = None
    try:
        response = await provider.simulate(tx)
    except:
        raise Exception("First error in simulating transaction")

    if response == None:
        raise Exception("Unable to simulate transaction")

    logs = response.logs
    if not logs:
        raise Exception("Simulated logs not found")

    parser = EventParser(
        Exchange.program_id,
        Exchange.program.coder
    )

    events = []

    def cb(e: Event):
        events.append(e)

    parser.parse_logs(response.logs, cb)


async def get_open_orders(program_id, market, user_key):
    from exchange import Exchange
    return await PublicKey.find_program_address(
        [
            bytes("open-orders", "utf-8"),
            bytes(constants.DEX_PID[Exchange.network]),
            bytes(market),
            bytes(user_key)
        ],
        program_id
    )


async def get_open_orders_map(program_id: PublicKey, open_orders: PublicKey):
    return await PublicKey.find_program_address(
        [bytes(open_orders)],
        program_id
    )


async def get_referrer_alias_address(program_id: PublicKey, alias):
    return await PublicKey.find_program_address(
        [
            bytes("referrer-alias", "utf-8"),
            bytes(alias)
        ],
        program_id
    )


async def get_market_node(program_id: PublicKey, zeta_group: PublicKey, market_index: int):
    return await PublicKey.find_program_address(
        [
            bytes("market-node", "utf-8"),
            bytes(zeta_group),
            bytes([market_index])
        ],
        program_id
    )


async def get_market_indexes(program_id: PublicKey, zeta_group: PublicKey):
    return PublicKey.find_program_address(
        [
            bytes("market-indexes", "utf-8"),
            bytes(zeta_group)
        ],
        program_id
    )


def convert_native_integer_to_decimal(amount):
    return amount / (10 ** constants.PLATFORM_PRECISION)


async def get_token_account_info(provider: Provider, key: PublicKey):
    print("pubkey of token account: " + str(key))
    info = await provider.connection.get_account_info(key)
    if info == None:
        raise Exception("Token account " + str(key) + "doesn't exist")

    print("info keys")
    print(info.keys())
    print(str(info))
    print(str(info["result"]))
    print(info["result"].keys())
    print(info["result"]['value']['data'])
    print(info["result"]['value']['data'][0])

    print(len(info["result"]['value']['data'][0]))
    print(TOKEN_ACCOUNT_LAYOUT.sizeof())

    if len(info["result"]['value']['data'][0]) != TOKEN_ACCOUNT_LAYOUT.sizeof():
        raise Exception("Invalid account size")

    data = bytes.fromhex(info["result"]['value']['data'][0])
    account_info = TOKEN_ACCOUNT_LAYOUT.parse(data)

    account_info.address = key
    account_info.mint = PublicKey(account_info.mint)
    account_info.owner = PublicKey(account_info.owner)
    account_info.amount = account_info.amount

    if account_info.delegate_option == 0:
        account_info.delegate = None
        account_info.delegated_amount = 0
    else:
        account_info.delegate = PublicKey(account_info.delegate)
        account_info.delegated_amount = account_info.delegated_amount

    account_info.is_initialized = account_info.state != 0
    account_info.is_frozen = account_info.state == 2

    if account_info.is_native_option == 1:
        account_info.rent_exempt_reserve = account_info.is_native
        account_info.is_native = True
    else:
        account_info.rent_exempt_reserve = None
        account_info.is_native = False

    if account_info.close_authority_option == 0:
        account_info.close_authority = None
    else:
        account_info.close_authority = PublicKey(account_info.close_authority)

    return account_info
