from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import json
import jsonpickle
from typing import Any, List



"""
GRID SEARCH param. Leave this equal to None to initialize it.

Wherever you want to perform a grid search, replace that with = grid search

ex.

parameter_beta = grid_search

Run the gridsearch.py with this file name and follow guide there
"""
grid_search = None


"""
Boilerplate for backtester
"""
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


""" Strategy Code starts here """

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    
    
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "clear_width": 0
    },
    Product.KELP: {
        "fair_value": 2000,
        "ewma_beta": 0, # found using grid search 0 - 1
        "clear_width": 0,
        "rolling_window": 5000
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.25, # found using grid search 0 - 1
        "clear_width": 0,
        "rolling_window": 5000
    }
}

    
    
class Trader:
    
    """
    Initialize any unique parameters we want about the products
    Including their limits
    """
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}
        
    
    def best_orders(
        self,
        product: str,
        order_depth: OrderDepth #product specific OrderDepth
    ) -> (float, float, int, int):

        """
        Helper function used to find the best bid/ask prices and volumes
        """      
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        return best_bid, best_ask, best_bid_amount, best_ask_amount
    
    
    def mm_orders(
            self,
            product: str,
            order_depth: OrderDepth
            ) -> (int, int, int, int):
        
        """
        Helper function used to find the widest bid/ask prices and volumes, the bot market maker
        """  
        
        if len(order_depth.buy_orders) != 0:
            mm_bid = min(order_depth.buy_orders.keys())
            mm_bid_amount = order_depth.buy_orders[mm_bid]
            
        if len(order_depth.sell_orders) != 0:
            mm_ask = max(order_depth.sell_orders.keys())
            mm_ask_amount = -1 * order_depth.sell_orders[mm_ask]
    
        return mm_bid, mm_ask, mm_bid_amount, mm_ask_amount
    
    
    def prev_mm_quote(
            self,
            product: str,
            traderObject: dict,
            order_depth: OrderDepth
            ) -> (int, int, int, int):
        
        prev_quote = traderObject.get("prev_mm_quote", {}).get(product)
        
        curr_bid, curr_ask, curr_bid_quantity, curr_ask_quantity = self.mm_orders(product, order_depth)
        traderObject["prev_mm_quote"][product] = [curr_bid, curr_ask, curr_bid_quantity, curr_ask_quantity]
        
        bid = prev_quote[0]
        ask = prev_quote[1]
        bid_quantity = prev_quote[2]
        ask_quantity = prev_quote[3]
        
        return bid, ask, bid_quantity, ask_quantity
    
    
    def ewma(
        self,
        product: str,
        traderObject: dict,
        mid: float,
        beta: float # exp weight parameter
        ) -> float:
        
        """
        Helper function used to calculate an exponentially weighted moving average
        """  
        
        prev_ewma = traderObject.get("ewma", {}).get(product)

        if prev_ewma is None:
            result = mid
        else:
            result = (1 - beta) * mid + beta * prev_ewma
            
        traderObject["ewma"][product] = result
    
        return result

    def ewma_volatility(
        self,
        product: str,
        traderObject: dict,
        mid: float,
        beta: float  # smoothing factor, e.g., 0.94
        ) -> float:
        
        """
        Calculates exponentially weighted volatility (standard deviation) of price levels.
        This version works without computing returns explicitly.
        """
        
        # Access or initialize tracking variables
        if "vol" not in traderObject:
            traderObject["vol"] = {}
        if product not in traderObject["vol"]:
            traderObject["vol"][product] = {
                "ewma_var": 0.0,
                "last_price": mid
            }
    
        vol_data = traderObject["vol"][product]
        last_price = vol_data["last_price"]
        prev_ewma_var = vol_data["ewma_var"]
    
        # Use squared difference between current and last price
        squared_diff = (mid - last_price) ** 2
    
        # Update EWMA variance
        ewma_var = beta * prev_ewma_var + (1 - beta) * squared_diff
    
        # Store updates
        traderObject["vol"][product]["ewma_var"] = ewma_var
        traderObject["vol"][product]["last_price"] = mid
    
        # Return standard deviation
        return ewma_var ** 0.5
        
    def take_orders(
            self,
            product:str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            ) -> (List[Order], int, int):
        
        """
        Primary function used to place directional orders based on market prices and a pre-determined fair value. Typically always the MM-mid
        """  
        
        orders: List[Order] = []
        buy_quantity = 0 
        sell_quantity = 0
        
        best_bid, best_ask, best_bid_quantity, best_ask_quantity = self.best_orders(product, order_depth)
        
        if best_bid and best_bid > fair_value:
            sell_volume = min(best_bid_quantity, position + position_limit)
            if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                sell_quantity = sell_volume
        
        if best_ask and best_ask < fair_value:
            buy_volume = min(best_ask_quantity, position_limit - position)
            if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume))
                buy_quantity = buy_volume
                    
        return orders, buy_quantity, sell_quantity #signed, passed onto make so they don't cancel each other
    
    
    def resin_make_orders(
        self,
        product: str,
        order_depth: OrderDepth,  # Product-specific OrderDepth
        fair_value: int,
        position: int,
        position_limit: int,
        default_edge: float,  # The distance from fair value for bids/asks
        default_order_size: int,
        take_buy_quantity: int, # existing buy and sell orders we have placed due to market taking to prevent cancellation
        take_sell_quantity: int
        ) -> (List[Order], int, int):

        """
        Primary function used to market make for RAINFOREST RESIN
        """  
        
        orders: List[Order] = []
    
        # Calculate bid and ask prices
        bid_price = fair_value - default_edge
        ask_price = fair_value + default_edge
        
        buy_quantity = 0
        sell_quantity = 0
    
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity # Maximum we can buy
        max_sell_capacity = position + position_limit  - take_sell_quantity # Maximum we can sell
    
        # Place bid order if does not exceed capacity
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)  # Default order size of 15
            orders.append(Order(product, bid_price, buy_quantity))  # Buy order
    
        # Place ask order if does not exceed capacity
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)  # Default order size of 15
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    
    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            clear_width: int,
            take_buy_quantity: int, # existing buy and sell orders we have placed due to market taking to prevent cancellation
            take_sell_quantity: int
            ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        
        effective_position = position + take_buy_quantity - take_sell_quantity

        # Fair value thresholds
        fair_for_bid = fair_value - clear_width
        fair_for_ask = fair_value + clear_width
    
        # Remaining quantities we are allowed to trade
        remaining_buy_qty = position_limit - (position + take_buy_quantity)
        remaining_sell_qty = position_limit + (position - take_sell_quantity)
    
        # Try to clear long positions by selling at better-than-fair prices
        if effective_position > 0 and remaining_sell_qty > 0:
            # Look for aggressive buyers (bid >= fair + width)
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if price >= fair_for_ask:
                    qty = min(abs(volume), effective_position, remaining_sell_qty)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        effective_position -= qty
                        take_sell_quantity += qty
                        remaining_sell_qty -= qty
    
        # Try to clear short positions by buying at better-than-fair prices
        if effective_position < 0 and remaining_buy_qty > 0:
            # Look for aggressive sellers (ask <= fair - width)
            for price, volume in sorted(order_depth.sell_orders.items()):
                if price <= fair_for_bid:
                    qty = min(abs(volume), abs(effective_position), remaining_buy_qty)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        effective_position += qty
                        take_buy_quantity += qty
                        remaining_buy_qty -= qty
    
        return orders, take_buy_quantity, take_sell_quantity
    
    
    def make_orders(self,
        product: str,
        order_depth: OrderDepth,  # Product-specific OrderDepth
        fair_value: int,
        position: int,
        position_limit: int,
        default_edge: float,  # The distance from fair value for bids/asks
        default_order_size: float,
        disregard_edge: float,
        join_edge: float,
        take_buy_quantity: int, # existing buy and sell orders we have placed due to market taking to prevent cancellation
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask_price = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask_price = best_ask_above_fair  # join
            else:
                ask_price = best_ask_above_fair - 1  # penny

        bid_price = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid_price = best_bid_below_fair
            else:
                bid_price = best_bid_below_fair + 1
                
        buy_quantity = 0
        sell_quantity = 0
    
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity # Maximum we can buy
        max_sell_capacity = position + position_limit  - take_sell_quantity # Maximum we can sell
    
        # Place bid order if does not exceed capacity
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)  # Default order size of 15
            orders.append(Order(product, bid_price, buy_quantity))  # Buy order
    
        # Place ask order if does not exceed capacity
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)  # Default order size of 15
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    def kelp_make_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            default_order_size: int,
            take_buy_quantity: int,
            take_sell_quantity: int,
            bias: float,
            state: TradingState
            ) -> (List[Order], int, int):

        orders: List[Order] = []
    
        EDGE = 1.5
        BIAS = bias
    
        # Start with base bid/ask
        bid_price = fair_value - EDGE
        ask_price = fair_value + EDGE
    
        # Bias based on inventory
        if position < 0:
            # Want to buy more → more aggressive bid
            bid_price += BIAS
        elif position > 0:
            # Want to sell more → more aggressive ask
            ask_price -= BIAS
    
        # Round to int
        bid_price = round(bid_price)
        ask_price = round(ask_price)
    
        # Inventory-aware capacity
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position + position_limit - take_sell_quantity
    
        buy_quantity = 0
        sell_quantity = 0
    
        # Place buy order
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
    
        # Place sell order
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    def ink_make_orders(self,
                        product: str,
                        order_depth: OrderDepth,
                        fair_value: int,
                        position: int,
                        position_limit: int,
                        default_order_size: int,
                        take_buy_quantity: int,
                        take_sell_quantity: int,
                        penny_amount: int = 1
                        ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        buy_quantity = 0
        sell_quantity = 0
        
        
        mm_bid, mm_ask, mm_bid_quantity, mm_ask_quantity = self.mm_orders(product, order_depth)
        
        bid_price = mm_bid + penny_amount
        ask_price = mm_ask - penny_amount
        
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position + position_limit - take_sell_quantity
        
        # Place buy order
        if max_buy_capacity > 0 and position < 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
    
        # Place sell order
        if max_sell_capacity > 0 and position > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)
            orders.append(Order(product, ask_price, -sell_quantity))
            
        if max_sell_capacity > 0 and max_buy_capacity > 0 and position == 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
            sell_quantity = min(default_order_size, max_sell_capacity)
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity

    
    """
    Main run function
    """
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        # load any prev data
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        # if first time, create it
        else:
            traderObject = {
                "prev_quote": {
                    Product.KELP: [],
                    Product.RAINFOREST_RESIN: [],
                    Product.SQUID_INK: []
                    },  # quotes from last tick
                "ewma": {
                    Product.KELP: None,
                    Product.RAINFOREST_RESIN: None,
                    Product.SQUID_INK: None
                    }
            }
        
        """ Rainforest Resin"""
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            # taking orders
            resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.take_orders(Product.RAINFOREST_RESIN,
                                                                      resin_order_depth,
                                                                      self.params[Product.RAINFOREST_RESIN]["fair_value"],
                                                                      resin_position,
                                                                      self.LIMIT[Product.RAINFOREST_RESIN])
            
            # making orders
            required_edge = 4
            default_order_size = 15
            
            resin_make_orders, _, _ = self.resin_make_orders(Product.RAINFOREST_RESIN, 
                                                       resin_order_depth, 
                                                       self.params[Product.RAINFOREST_RESIN]["fair_value"], 
                                                       resin_position, 
                                                       self.LIMIT[Product.RAINFOREST_RESIN], 
                                                       required_edge,
                                                       default_order_size, 
                                                       resin_take_buy_quantity, 
                                                       resin_take_sell_quantity)
            
            # clearing orders
            resin_clear_orders, _, _ = self.clear_orders(Product.RAINFOREST_RESIN,
                                                        resin_order_depth,
                                                        self.params[Product.RAINFOREST_RESIN]["fair_value"],
                                                        resin_position,
                                                        self.LIMIT[Product.RAINFOREST_RESIN],
                                                        self.params[Product.RAINFOREST_RESIN]["clear_width"],
                                                        resin_take_buy_quantity,
                                                        resin_take_sell_quantity)
            
            # adding orders together
            result[Product.RAINFOREST_RESIN] = (resin_take_orders + resin_make_orders + resin_clear_orders)
        
        """ Kelp """
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_order_depth = state.order_depths[Product.KELP]
            
            kelp_mm_bid, kelp_mm_ask, _, _ = self.mm_orders(Product.KELP, kelp_order_depth)
            kelp_mm_mid = (kelp_mm_bid + kelp_mm_ask) / 2
        
            kelp_bid, kelp_ask, _, _ = self.best_orders(Product.KELP, kelp_order_depth)    
                        
            kelp_fair_value = self.ewma(Product.KELP, traderObject, kelp_mm_mid, self.params[Product.KELP]["ewma_beta"])
            if kelp_fair_value:
                self.params[Product.KELP]["fair_value"] = kelp_fair_value
                
            # taking orders
            kelp_take_orders, kelp_take_buy_quantity, kelp_take_sell_quantity = self.take_orders(Product.KELP,
                                                                                                 kelp_order_depth,
                                                                                                 self.params[Product.KELP]["fair_value"],
                                                                                                 kelp_position,
                                                                                                 self.LIMIT[Product.KELP])
            
            # making orders
            """ New make orders method """
            kelp_make_orders, _, _ = self.kelp_make_orders(Product.KELP,
                                                      kelp_order_depth,
                                                      kelp_mm_mid,
                                                      position = kelp_position,
                                                      position_limit = 50,
                                                      default_order_size = 15,
                                                      take_buy_quantity= kelp_take_buy_quantity,
                                                      take_sell_quantity= kelp_take_sell_quantity,
                                                      bias = 1,
                                                      state = state)
            
            kelp_clear_orders, _, _ = self.clear_orders(Product.KELP,
                                                        kelp_order_depth,
                                                        kelp_mm_mid,
                                                        kelp_position,
                                                        self.LIMIT[Product.KELP],
                                                        self.params[Product.KELP]["clear_width"],
                                                        kelp_take_buy_quantity,
                                                        kelp_take_sell_quantity)
        
        result[Product.KELP] = (kelp_take_orders + kelp_make_orders + kelp_clear_orders)

        """ Squid Ink """
        
        if Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_order_depth = state.order_depths[Product.SQUID_INK]
            
            ink_mm_bid, ink_mm_ask, _, _ = self.mm_orders(Product.SQUID_INK, ink_order_depth)
            ink_mm_mid = (ink_mm_bid + ink_mm_ask) / 2
        
            ink_bid, ink_ask, _, _ = self.best_orders(Product.SQUID_INK, ink_order_depth)    
                        
            ink_fair_value = self.ewma(Product.SQUID_INK, traderObject, ink_mm_mid, self.params[Product.SQUID_INK]["ewma_beta"])
            if ink_fair_value:
                self.params[Product.SQUID_INK]["fair_value"] = ink_fair_value
                
            # taking orders
            ink_take_orders, ink_take_buy_quantity, ink_take_sell_quantity = self.take_orders(Product.SQUID_INK,
                                                                                                 ink_order_depth,
                                                                                                 ink_mm_mid,
                                                                                                 ink_position,
                                                                                                 self.LIMIT[Product.SQUID_INK])
            
            # one way market making
            ink_make_orders, _, _ = self.ink_make_orders(Product.SQUID_INK,
                                                         ink_order_depth,
                                                         ink_mm_mid,
                                                         ink_position,
                                                         self.LIMIT[Product.SQUID_INK],
                                                         default_order_size = 15,
                                                         take_buy_quantity= ink_take_buy_quantity,
                                                         take_sell_quantity= ink_take_sell_quantity)
            
            """ DISCONTINUED for now
            # making orders
            ink_make_orders2, _, _ = self.make_orders2(Product.SQUID_INK,
                                                      ink_order_depth,
                                                      ink_mm_mid,
                                                      position = ink_position,
                                                      position_limit = 50,
                                                      default_order_size = 5,
                                                      take_buy_quantity= ink_take_buy_quantity,
                                                      take_sell_quantity= ink_take_sell_quantity,
                                                      bias = 1,
                                                      state = state)
            """
            # clear orders, provides no benefit for now as the make orders clears positions
            ink_clear_orders, _, _ = self.clear_orders(Product.SQUID_INK,
                                                        ink_order_depth,
                                                        ink_mm_mid,
                                                        ink_position,
                                                        self.LIMIT[Product.SQUID_INK],
                                                        self.params[Product.SQUID_INK]["clear_width"],
                                                        ink_take_buy_quantity,
                                                        ink_take_sell_quantity)
        
        # making
        result[Product.SQUID_INK] = (ink_take_orders +  ink_clear_orders + ink_make_orders)

        
        """ 
        Tidying up
        """

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData