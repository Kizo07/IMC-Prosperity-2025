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
    
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000
    },
    Product.KELP: {
        "fair_value": None,
        "ewma_beta": 0.5 # found using grid search 0 - 1
    },
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
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}
        
    

    def best_orders(
        self,
        product: str,
        order_depth: OrderDepth #product specific OrderDepth
    ) -> (float, float, int, int):

        """
        Helper function used to find the best bid/ask prices and volumes
        """      
    
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
        
        return best_bid, best_ask, best_bid_amount, best_ask_amount
    
    
    
    def order_flow_imbalance(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: dict
    ) -> float:
        
        """
        Helper function used to find the order flow imbalance from current data
        and previous data (not well tested)
        """
        
        B_n, A_n, q_B_n, q_A_n = self.best_orders(product, order_depth)
        B_n_minus_1, A_n_minus_1, q_B_n_minus_1, q_A_n_minus_1 = traderObject["prev_quote"][product]
        
        
        I_B_increase = int(B_n >= B_n_minus_1)  # 1 if best bid increases, else 0
        I_B_decrease = int(B_n <= B_n_minus_1)  # 1 if best bid decreases, else 0
        I_A_increase = int(A_n >= A_n_minus_1)  # 1 if best ask increases, else 0
        I_A_decrease = int(A_n <= A_n_minus_1)  # 1 if best ask decreases, else 0

        # Calculate order flow imbalance using the provided formula
        e_n = (I_B_increase * q_B_n 
               - I_B_decrease * q_B_n_minus_1 
               - I_A_decrease * q_A_n 
               + I_A_increase * q_A_n_minus_1)

        return e_n

      
    def previous_midprice(
            self,
            product:str,
            traderObject: dict
            ) -> float:
        """
        Helper function used to calculate the previous tick's mid price
        """  
        
        try:
            prev_bid = traderObject["prev_quote"][product][0]
            prev_ask = traderObject["prev_quote"][product][1]
            return (prev_bid + prev_ask) / 2
        
        except IndexError:
            return None
    
    
    
    def full_book_weighted_mid_price(
            self,
            product:str,
            order_depth: OrderDepth
            ) -> float:
   
        """
        Computes the full book volume-weighted mid-price using *all* buy and sell orders.
        """
        
        weighted_sum = 0.0
        total_volume = 0
    
        # Buy side
        for price, volume in order_depth.buy_orders.items():
            weighted_sum += price * volume
            total_volume += volume
    
        # Sell side (convert negative volume to positive)
        for price, volume in order_depth.sell_orders.items():
            volume = abs(volume)
            weighted_sum += price * volume
            total_volume += volume
    
        if total_volume == 0:
            return None  # Can't compute weighted average if there's no volume
    
        return weighted_sum / total_volume

    
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
        
        if traderObject["ewma"][product] is None:
            ewma = mid
        else:
            ewma = (mid * (1 - beta) + beta * traderObject["ewma"][product])
            
        return ewma

    def take_orders(
            self,
            product:str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            ) -> (List[Order], int, int):
        
        """
        Primary function used to place directional orders based on market prices and a pre-determined fair value
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
                    
        return orders, sell_quantity, buy_quantity #signed, passed onto make so they don't cancel each other
    
    def resin_make_orders(
        self,
        product: str,
        order_depth: OrderDepth,  # Product-specific OrderDepth
        fair_value: int,
        position: int,
        position_limit: int,
        n: float,  # The distance from fair value for bids/asks
        threshold_ratio: float,  # The ratio of the max position limit for threshold
        default_order_size: int,
        take_sell_quantity: int, # existing buy and sell orders we have placed due to market taking to prevent cancellation
        take_buy_quantity: int
        ) -> List[Order]:

        """
        Primary function used to market make for RAINFOREST RESIN
        """  
        
        orders: List[Order] = []
    
        # Calculate bid and ask prices
        bid_price = fair_value - n
        ask_price = fair_value + n
    
        # Calculate threshold based on position limit
        threshold = position_limit * threshold_ratio
    
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity # Maximum we can buy
        max_sell_capacity = position + position_limit  - take_sell_quantity # Maximum we can sell
    
        # Place bid order if within threshold and does not exceed capacity
        if max_buy_capacity > 0 and position < threshold:
            buy_quantity = min(default_order_size, max_buy_capacity)  # Default order size of 15
            orders.append(Order(product, bid_price, buy_quantity))  # Buy order
    
        # Place ask order if within threshold and does not exceed capacity
        if max_sell_capacity > 0 and -position < threshold:
            sell_quantity = min(default_order_size, max_sell_capacity)  # Default order size of 15
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders
    
    
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
                    Product.RAINFOREST_RESIN: []
                    },  # quotes from last tick
                "ewma": {
                    Product.KELP: None,
                    Product.RAINFOREST_RESIN: None
                    }
            }
        
        """ Rainforest Resin"""
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            # taking orders
            resin_take_orders, resin_take_sell_quantity, resin_take_buy_quantity = self.take_orders(Product.RAINFOREST_RESIN,
                                                                      resin_order_depth,
                                                                      self.params[Product.RAINFOREST_RESIN]["fair_value"],
                                                                      resin_position,
                                                                      self.LIMIT[Product.RAINFOREST_RESIN])
            
            # making orders
            required_edge = 4
            soft_threshold_ratio = 1 # for starting to cancel out some of the order directions
            default_order_size = 15
            
            resin_make_orders = self.resin_make_orders(Product.RAINFOREST_RESIN, 
                                                       resin_order_depth, 
                                                       self.params[Product.RAINFOREST_RESIN]["fair_value"], 
                                                       resin_position, 
                                                       self.LIMIT[Product.RAINFOREST_RESIN], 
                                                       required_edge, 
                                                       soft_threshold_ratio, 
                                                       default_order_size, 
                                                       resin_take_sell_quantity, 
                                                       resin_take_buy_quantity)
            
            # adding orders together
            result[Product.RAINFOREST_RESIN] = (resin_take_orders + resin_make_orders)
        
        
        """ 
        Tidying up
        """

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData