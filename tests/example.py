import asyncio
import os

from anchorpy import Provider
from anchorpy import Wallet
from dotenv import load_dotenv
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient, Commitment

from pyzeta import CLUSTER_URLS, place_order, Asset, Side, OrderType

load_dotenv()

PRIVATE_KEY = os.getenv("private_key")
PROGRAM_ID = os.getenv("PROGRAM_ID")
SERVER_URL = os.getenv("SERVER_URL")


async def main():
    network = "devnet"
    keypair = Keypair.from_secret_key(PRIVATE_KEY.encode())
    wallet = Wallet(keypair)
    client = AsyncClient(CLUSTER_URLS[network], commitment=Commitment("confirmed"))
    await client.is_connected()
    response = await client.get_account_info(
        wallet.public_key,
        encoding="base64"
    )
    my_lamports = response['result']['value']
    # if my_lamports is None or my_lamports['lamports'] < 100000000:
    # await client.request_airdrop(wallet.public_key, 100000000)
    # response = requests.post(f"{SERVER_URL}/faucet/usdc", json={
    #     "key": str(wallet.public_key),
    #     "amount": 10_000
    # })
    # if response.status_code != 200:
    #     logger.error("Failed to get USDC")

    provider = Provider(client, wallet)

    await place_order(
        provider,
        Asset.BTC,
        10,
        100000,
        # Decimal(0.1),
        2000,
        # 2 / pow(10, 3),
        Side.BID,
        OrderType.LIMIT,
    )

    # exchange = Exchange(
    #     [Asset.SOL, Asset.BTC],
    #     PROGRAM_ID,
    #     network,
    #     client,
    #     None
    # )
    #
    # await exchange.load(None)
    #
    # exchange.display_state()
    await client.close()


asyncio.run(main())
