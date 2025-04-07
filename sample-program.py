import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import jsonpickle

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


    
class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        MAX_POSITION = 20
        
        # load any prev data
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)
        else:
            trader_data = {
                "prev_prices": {}  # midprices from last round
            }
        
        
        """ 
        Performing the main function
        """
        
        # calculate bid/ask and mid
        for product in state.order_depths.keys():
            """ 
            Finding basic info such as bid/ask and calculating a mid price
            """
            
            # initializing
            best_bid = None
            best_ask = None
            orders: list[Order] = []
            if product not in trader_data["prev_prices"]:
                trader_data["prev_prices"][product] = []  # Initialize list if not present

            # gathering basic info
            order_depth: OrderDepth = state.order_depths[product]
            current_pos = state.position.get(product, 0)   
            
            # calculating the mid price
            
            # if there are any sell orders
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                
            # if there are any buy orders
            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
        
            # calculate a new midprice, else use old midprice
            if best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
            elif product in trader_data["prev_prices"]:
                mid_price = trader_data["prev_prices"][product][-1]

            
            """ 
            Strategy implementation
            
            Assign our fair price as the previous mid price
            Send a max buy order if it is cheaper than that
            Send a max sell order if it is more expensive than that
            
            """
            # checking to see if we know the fair price
            try:
                acceptable_price = trader_data["prev_prices"][product][-1]
                
                if best_ask and best_ask < acceptable_price:
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    buy_volume = min(-best_ask_volume, MAX_POSITION - current_pos)
                    if buy_volume > 0:
                        orders.append(Order(product, best_ask, buy_volume))
                        
                if best_bid and best_bid > acceptable_price:
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    sell_volume = min(best_bid_volume, current_pos + MAX_POSITION)
                    if sell_volume > 0:
                        orders.append(Order(product, best_bid, -sell_volume))
            
            # Skip the following trading logic for this product
            # Should happen on the first trading day
            except KeyError:
                pass
            except IndexError:
                pass
                
            """ 
            Tidying up
            """
            trader_data["prev_prices"][product].append(mid_price)
            result[product] = orders
            
        # encode prev data
        traderData = jsonpickle.encode(trader_data)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData