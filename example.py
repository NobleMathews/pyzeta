def place_order_v3_ix(asset, market_index, price, size, side, order_type, client_order_id, tag, margin_account,
                      authority, open_orders, whitelist_trading_fees_account):
    if len(tag) > constants.MAX_ORDER_TAG_LENGTH:
        raise Exception("Tag is too long! Max length = " + str(constants.MAX_ORDER_TAG_LENGTH))

    sub_exchange = Exchange.get_sub_exchange(asset)
    market_data = sub_exchange.markets.markets[market_index]
    remaining_accounts = [
        {
            "pubkey": whitelist_trading_fees_account,
            "is_signer": False,
            "is_writable": False,
        },
    ] if whitelist_trading_fees_account == None else []

    return place_order_v3(
        {
            "price": price,
            "size": size,
            "side": types.to_program_side(side),
            "order_type": types.to_program_order_type(order_type),
            "client_order_id": None if client_order_id == 0 else client_order_id,
        },
        {
            "state": Exchange.state_address,
            "zeta_group": sub_exchange.zeta_group_address,
            "margin_account": margin_account,
            "authority": authority,
            "dex_program": constants.DEX_PID[Exchange.network],
            "token_program": TOKEN_PROGRAM_ID,
            "serum_authority": Exchange._serum_authority,
            "greeks": sub_exchange.zeta_group.greeks,
            "open_orders": open_orders,
            "rent": SYSVAR_RENT_PUBKEY,
            "market_accounts": {
                "market": market_data.serum_market.decoded.own_address,
                "request_queue": market_data.serum_market.decoded.request_queue,
                "event_queue": market_data.serum_market.decoded.event_queue,
                "bids": market_data.serum_market.decoded.bids,
                "asks": market_data.serum_market.decoded.asks,
                "coin_vault": market_data.serum_market.decoded.base_vault,
                "pc_vault": market_data.serum_market.decoded.quote_vault,
                "order_payer_token_account": market_data.quote_vault if types.Side.BID else market_data.base_vault,
                "coin_wallet": market_data.base_vault,
                "pc_wallet": market_data.quote_vault,
            },
            "oracle": sub_exchange.zeta_group.oracle,
            "market_node": sub_exchange.greeks.node_keys[market_index],
            "market_mint": market_data.serum_market.quote_mint_address if side == types.Side.BID else market_data.serum_market.base_mint_address,
            "mint_authority": Exchange._mint_authority,
        }
    )
