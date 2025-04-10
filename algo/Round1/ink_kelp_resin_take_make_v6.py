from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import json
import jsonpickle
import numpy as np  # Add numpy import
from typing import Any, List
from collections import deque  # Add deque for efficient sliding window operations



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
        "rolling_window": 100,  # Reduced window size for efficiency
        "spread_multiplier": 1.1,  # Multiplier for the typical spread (slightly smaller than SQUID_INK)
        "min_spread": 2,           # Minimum spread to use
        "max_spread": 5,           # Maximum spread to use (slightly smaller than SQUID_INK)
        "position_scale": 0.6,     # How much to scale orders based on position
        "vol_window": 15,          # Window for calculating spread
        "order_skew_threshold": 0.25 # Threshold for skewing orders based on book imbalance
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.25, # found using grid search 0 - 1
        "clear_width": 0,
        "rolling_window": 100,  # Reduced window size for efficiency
        "spread_multiplier": 1.2,  # Multiplier for the typical spread
        "min_spread": 2,           # Minimum spread to use
        "max_spread": 6,           # Maximum spread to use
        "position_scale": 0.7,     # How much to scale orders based on position
        "vol_window": 20,          # Window for calculating spread
        "order_skew_threshold": 0.3 # Threshold for skewing orders based on book imbalance
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
        best_bid = best_ask = best_bid_amount = best_ask_amount = None
        
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
        mm_bid = mm_ask = mm_bid_amount = mm_ask_amount = None
        
        if len(order_depth.buy_orders) != 0:
            mm_bid = min(order_depth.buy_orders.keys())
            mm_bid_amount = order_depth.buy_orders[mm_bid]
            
        if len(order_depth.sell_orders) != 0:
            mm_ask = max(order_depth.sell_orders.keys())
            mm_ask_amount = -1 * order_depth.sell_orders[mm_ask]
    
        return mm_bid, mm_ask, mm_bid_amount, mm_ask_amount
    
    def rolling_mm_mid_quotes(
            self,
            product: str,
            order_depth: OrderDepth,
            traderObject: dict,
            rolling_window: int
            ):
        
        # Initialize the rolling_mid_quotes dictionary if it doesn't exist
        if "rolling_mid_quotes" not in traderObject:
            traderObject["rolling_mid_quotes"] = {}
        
        # Initialize the list for this product if it doesn't exist
        if product not in traderObject["rolling_mid_quotes"]:
            traderObject["rolling_mid_quotes"][product] = deque(maxlen=rolling_window)
        
        rolling_prices = traderObject["rolling_mid_quotes"][product]
        
        mm_bid, mm_ask, _, _ = self.mm_orders(product, order_depth)
        
        if mm_bid is not None and mm_ask is not None:
            mm_mid = (mm_bid + mm_ask) / 2
            rolling_prices.append(mm_mid)
        
        return rolling_prices
    
    
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
        if not traderObject["prev_quote"][product]:
            return 0
            
        B_n, A_n, q_B_n, q_A_n = self.best_orders(product, order_depth)
        
        if B_n is None or A_n is None:
            return 0
            
        B_n_minus_1, A_n_minus_1, q_B_n_minus_1, q_A_n_minus_1 = traderObject["prev_quote"][product]
        
        if B_n_minus_1 is None or A_n_minus_1 is None:
            return 0
        
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
            if prev_bid is not None and prev_ask is not None:
                return (prev_bid + prev_ask) / 2
            return None
        
        except (IndexError, TypeError):
            return None
    
    
    def full_book_weighted_mid_price(
            self,
            product:str,
            order_depth: OrderDepth
            ) -> float:
   
        """
        Computes the full book volume-weighted mid-price using *all* buy and sell orders.
        Optimized with numpy arrays
        """
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
            
        # Convert to numpy arrays for faster computation
        buy_prices = np.array(list(order_depth.buy_orders.keys()))
        buy_volumes = np.array(list(order_depth.buy_orders.values()))
        sell_prices = np.array(list(order_depth.sell_orders.keys()))
        sell_volumes = np.abs(np.array(list(order_depth.sell_orders.values())))
        
        weighted_sum = np.sum(buy_prices * buy_volumes) + np.sum(sell_prices * sell_volumes)
        total_volume = np.sum(buy_volumes) + np.sum(sell_volumes)
    
        if total_volume == 0:
            return None
    
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
        if mid is None:
            return None
            
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
        Optimized with numpy arrays for faster computation.
        """
        
        # Access or initialize tracking variables
        if "vol" not in traderObject:
            traderObject["vol"] = {}
        if product not in traderObject["vol"]:
            traderObject["vol"][product] = {
                "price_history": np.array([mid]),
                "ewma_var": 0.0
            }
    
        vol_data = traderObject["vol"][product]
        
        # Append current price to history
        vol_data["price_history"] = np.append(vol_data["price_history"][-100:], mid)
        
        # Calculate price differences
        if len(vol_data["price_history"]) > 1:
            # Use numpy's diff function for faster calculation
            diff = np.diff(vol_data["price_history"])
            # Calculate squared difference of most recent price change
            squared_diff = diff[-1]**2
            
            # Update EWMA variance
            vol_data["ewma_var"] = beta * vol_data["ewma_var"] + (1 - beta) * squared_diff
        
        # Return standard deviation
        return np.sqrt(vol_data["ewma_var"])

    def ink_fair_value(
            self,
            product:str,
            traderObject: dict,
            kelp_lag: int
            ) -> float:
        
        kelp_prices = traderObject.get("rolling_mid_quotes", {}).get(Product.KELP)
        
        if not kelp_prices or len(kelp_prices) <= kelp_lag:
            return None
        
        # Convert to numpy array for efficient calculation
        prices_array = np.array(kelp_prices)
        return np.mean(prices_array)
        
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
        
        if best_bid is not None and fair_value is not None and best_bid > fair_value:
            sell_volume = min(best_bid_quantity, position + position_limit)
            if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume))
                sell_quantity = sell_volume
        
        if best_ask is not None and fair_value is not None and best_ask < fair_value:
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
        if fair_value is None:
            return [], 0, 0
            
        orders: List[Order] = []
    
        # Calculate bid and ask prices
        bid_price = fair_value - default_edge
        ask_price = fair_value + default_edge
        
        buy_quantity = 0
        sell_quantity = 0
    
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity # Maximum we can buy
        max_sell_capacity = position + position_limit - take_sell_quantity # Maximum we can sell
    
        # Place bid order if does not exceed capacity
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
    
        # Place ask order if does not exceed capacity
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)
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
        
        if fair_value is None:
            return [], take_buy_quantity, take_sell_quantity
            
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
        
        if fair_value is None:
            return [], 0, 0
            
        orders: List[Order] = []
        
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask_price = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask_price = best_ask_above_fair  # join
            else:
                ask_price = best_ask_above_fair - 1  # penny

        bid_price = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid_price = best_bid_below_fair
            else:
                bid_price = best_bid_below_fair + 1
                
        buy_quantity = 0
        sell_quantity = 0
    
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity # Maximum we can buy
        max_sell_capacity = position_limit + position  - take_sell_quantity # Maximum we can sell
    
        # Place bid order if does not exceed capacity
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
    
        # Place ask order if does not exceed capacity
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    
    def calculate_optimal_spread(self, product, traderObject, current_spread):
        """
        Calculate the optimal spread based on recent price volatility
        Using numpy for faster calculations
        """
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
            
        if product not in traderObject["price_history"]:
            traderObject["price_history"][product] = deque(maxlen=100)
            return self.params[product]["min_spread"]
        
        price_history = traderObject["price_history"][product]
        
        if len(price_history) < 2:
            return self.params[product]["min_spread"]
            
        # Convert to numpy array for faster calculation
        prices = np.array(price_history)
        price_changes = np.abs(np.diff(prices))
        
        # Use only the most recent changes up to vol_window
        vol_window = min(len(price_changes), self.params[product]["vol_window"])
        recent_changes = price_changes[-vol_window:]
        
        if len(recent_changes) == 0:
            return self.params[product]["min_spread"]
            
        # Calculate volatility as the standard deviation
        volatility = np.std(recent_changes)
            
        # Scale the spread based on volatility
        optimal_spread = max(
            self.params[product]["min_spread"],
            min(
                self.params[product]["max_spread"],
                int(volatility * self.params[product]["spread_multiplier"])
            )
        )
        
        return optimal_spread
    
    
    def calculate_book_imbalance(self, order_depth):
        """
        Calculate order book imbalance to determine market direction pressure
        Using numpy for faster calculations
        """
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return 0
            
        # Convert to numpy arrays for faster computation
        buy_volumes = np.array(list(order_depth.buy_orders.values()))
        sell_volumes = np.array(list(order_depth.sell_orders.values()))
        
        buy_volume = np.sum(buy_volumes)
        sell_volume = np.abs(np.sum(sell_volumes))
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance  # Range: [-1, 1]
    
    
    def squid_ink_market_making(self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        traderObject: dict,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        """
        Advanced market making strategy for SQUID_INK based on order book distributions
        and spread-capturing opportunities
        """
        if fair_value is None:
            return [], 0, 0
            
        orders = []
        
        # Store current price in history
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
        if product not in traderObject["price_history"]:
            traderObject["price_history"][product] = deque(maxlen=100)
            
        traderObject["price_history"][product].append(fair_value)
            
        # Get best bid and ask
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        
        # Current market spread
        current_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 2
        
        # Calculate optimal spread based on volatility
        optimal_spread = self.calculate_optimal_spread(product, traderObject, current_spread)
        
        # Calculate book imbalance to detect pressure
        imbalance = self.calculate_book_imbalance(order_depth)
        
        # Adjust fair value slightly based on imbalance if significant
        adjusted_fair_value = fair_value
        if abs(imbalance) > self.params[product]["order_skew_threshold"]:
            adjusted_fair_value += imbalance * optimal_spread / 2
            
        # Calculate position scaling factor (reduce size as position grows)
        position_ratio = position / position_limit if position_limit != 0 else 0
        position_scale = 1.0 - abs(position_ratio) * self.params[product]["position_scale"]
        
        # Default order size, scaled by position
        default_order_size = 15
        adjusted_order_size = max(1, int(default_order_size * position_scale))
        
        # Skew order sizes based on current position
        buy_size = adjusted_order_size
        sell_size = adjusted_order_size
        
        if position > 0:
            # If we're long, increase sell size and decrease buy size
            sell_size = int(adjusted_order_size * (1 + position_ratio * 0.5))
            buy_size = int(adjusted_order_size * (1 - position_ratio * 0.5))
        elif position < 0:
            # If we're short, increase buy size and decrease sell size
            sell_size = int(adjusted_order_size * (1 + position_ratio * 0.5))
            buy_size = int(adjusted_order_size * (1 - position_ratio * 0.5))
            
        # Calculate bid and ask prices
        half_spread = optimal_spread / 2
        
        # If imbalance suggests strong buying pressure, adjust our prices upward
        if imbalance > self.params[product]["order_skew_threshold"]:
            bid_price = int(adjusted_fair_value - half_spread * 0.8)  # Tighter bid
            ask_price = int(adjusted_fair_value + half_spread * 1.2)  # Wider ask
        # If imbalance suggests strong selling pressure, adjust our prices downward
        elif imbalance < -self.params[product]["order_skew_threshold"]:
            bid_price = int(adjusted_fair_value - half_spread * 1.2)  # Wider bid
            ask_price = int(adjusted_fair_value + half_spread * 0.8)  # Tighter ask
        else:
            bid_price = int(adjusted_fair_value - half_spread)
            ask_price = int(adjusted_fair_value + half_spread)
            
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Adjust order sizes based on remaining capacity
        buy_size = min(buy_size, max_buy_capacity)
        sell_size = min(sell_size, max_sell_capacity)
        
        # Place orders if there's capacity
        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
            
        return orders, buy_size, sell_size
    
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

    def kelp_market_making(self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        traderObject: dict,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        """
        Advanced market making strategy for KELP based on order book distributions
        and spread-capturing opportunities
        """
        if fair_value is None:
            return [], 0, 0
            
        orders = []
        
        # Store current price in history
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
        if product not in traderObject["price_history"]:
            traderObject["price_history"][product] = deque(maxlen=100)
            
        traderObject["price_history"][product].append(fair_value)
            
        # Get best bid and ask
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        
        # Current market spread
        current_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else 2
        
        # Calculate optimal spread based on volatility
        optimal_spread = self.calculate_optimal_spread(product, traderObject, current_spread)
        
        # Calculate book imbalance to detect pressure
        imbalance = self.calculate_book_imbalance(order_depth)
        
        # Adjust fair value slightly based on imbalance if significant
        adjusted_fair_value = fair_value
        if abs(imbalance) > self.params[product]["order_skew_threshold"]:
            adjusted_fair_value += imbalance * optimal_spread / 3  # Less aggressive than SQUID_INK
            
        # Calculate position scaling factor (reduce size as position grows)
        position_ratio = position / position_limit if position_limit != 0 else 0
        position_scale = 1.0 - abs(position_ratio) * self.params[product]["position_scale"]
        
        # Default order size, scaled by position
        default_order_size = 18  # Slightly larger than SQUID_INK
        adjusted_order_size = max(1, int(default_order_size * position_scale))
        
        # Skew order sizes based on current position
        buy_size = adjusted_order_size
        sell_size = adjusted_order_size
        
        if position > 0:
            # If we're long, increase sell size and decrease buy size
            sell_size = int(adjusted_order_size * (1 + position_ratio * 0.6))
            buy_size = int(adjusted_order_size * (1 - position_ratio * 0.6))
        elif position < 0:
            # If we're short, increase buy size and decrease sell size
            sell_size = int(adjusted_order_size * (1 + position_ratio * 0.6))
            buy_size = int(adjusted_order_size * (1 - position_ratio * 0.6))
            
        # Calculate bid and ask prices
        half_spread = optimal_spread / 2
        
        # For KELP, we'll be slightly more conservative in our skewing
        if imbalance > self.params[product]["order_skew_threshold"]:
            bid_price = int(adjusted_fair_value - half_spread * 0.85)  # Tighter bid
            ask_price = int(adjusted_fair_value + half_spread * 1.15)  # Wider ask
        elif imbalance < -self.params[product]["order_skew_threshold"]:
            bid_price = int(adjusted_fair_value - half_spread * 1.15)  # Wider bid
            ask_price = int(adjusted_fair_value + half_spread * 0.85)  # Tighter ask
        else:
            bid_price = int(adjusted_fair_value - half_spread)
            ask_price = int(adjusted_fair_value + half_spread)
            
        # Check position limits
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Adjust order sizes based on remaining capacity
        buy_size = min(buy_size, max_buy_capacity)
        sell_size = min(sell_size, max_sell_capacity)
        
        # Place orders if there's capacity
        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
            
        return orders, buy_size, sell_size
    
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
                    },
                "rolling_mid_quotes": {},
                "price_history": {}
            }
        
        """ Rainforest Resin"""
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            # Save current quotes for next iteration
            resin_best = self.best_orders(Product.RAINFOREST_RESIN, resin_order_depth)
            traderObject["prev_quote"][Product.RAINFOREST_RESIN] = resin_best
            
            # taking orders
            resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.take_orders(
                Product.RAINFOREST_RESIN,
                resin_order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                self.LIMIT[Product.RAINFOREST_RESIN]
            )
            
            # making orders
            required_edge = 4
            default_order_size = 15
            
            resin_make_orders, _, _ = self.resin_make_orders(
                Product.RAINFOREST_RESIN, 
                resin_order_depth, 
                self.params[Product.RAINFOREST_RESIN]["fair_value"], 
                resin_position, 
                self.LIMIT[Product.RAINFOREST_RESIN], 
                required_edge,
                default_order_size, 
                resin_take_buy_quantity, 
                resin_take_sell_quantity
            )
            
            # clearing orders
            resin_clear_orders, _, _ = self.clear_orders(
                Product.RAINFOREST_RESIN,
                resin_order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                self.LIMIT[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_take_buy_quantity,
                resin_take_sell_quantity
            )
            
            # adding orders together
            result[Product.RAINFOREST_RESIN] = (resin_take_orders + resin_make_orders + resin_clear_orders)
        
        """ Kelp """
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_order_depth = state.order_depths[Product.KELP]
            
            # Save current quotes for next iteration
            kelp_best = self.best_orders(Product.KELP, kelp_order_depth)
            traderObject["prev_quote"][Product.KELP] = kelp_best
            
            kelp_mm_bid, kelp_mm_ask, _, _ = self.mm_orders(Product.KELP, kelp_order_depth)
            
            if kelp_mm_bid is not None and kelp_mm_ask is not None:
                kelp_mm_mid = (kelp_mm_bid + kelp_mm_ask) / 2
                
                # Keep track of market prices for volatility calculation
                self.rolling_mm_mid_quotes(Product.KELP, kelp_order_depth, traderObject, 100)
            
                kelp_fair_value = self.ewma(Product.KELP, traderObject, kelp_mm_mid, self.params[Product.KELP]["ewma_beta"])
                if kelp_fair_value:
                    self.params[Product.KELP]["fair_value"] = kelp_fair_value
                    
                # Use our advanced market making strategy for KELP
                kelp_mm_orders, kelp_buy_quantity, kelp_sell_quantity = self.kelp_market_making(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value or kelp_mm_mid,
                    kelp_position,
                    self.LIMIT[Product.KELP],
                    traderObject,
                    0,  # We're focusing on market making for KELP
                    0
                )
                
                result[Product.KELP] = kelp_mm_orders

        """ Squid Ink """
        
        if Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_order_depth = state.order_depths[Product.SQUID_INK]
            
            # Save current quotes for next iteration
            ink_best = self.best_orders(Product.SQUID_INK, ink_order_depth)
            traderObject["prev_quote"][Product.SQUID_INK] = ink_best
            
            ink_mm_bid, ink_mm_ask, _, _ = self.mm_orders(Product.SQUID_INK, ink_order_depth)
            
            if ink_mm_bid is not None and ink_mm_ask is not None:
                ink_mm_mid = (ink_mm_bid + ink_mm_ask) / 2
            
                # Keep track of market prices for volatility calculation
                self.rolling_mm_mid_quotes(Product.SQUID_INK, ink_order_depth, traderObject, 100)
                        
                ink_fair_value = self.ewma(Product.SQUID_INK, traderObject, ink_mm_mid, self.params[Product.SQUID_INK]["ewma_beta"])
                if ink_fair_value:
                    self.params[Product.SQUID_INK]["fair_value"] = ink_fair_value
                    
                # Use our advanced market making strategy for SQUID_INK
                ink_mm_orders, ink_buy_quantity, ink_sell_quantity = self.squid_ink_market_making(
                    Product.SQUID_INK,
                    ink_order_depth,
                    ink_fair_value or ink_mm_mid,
                    ink_position,
                    self.LIMIT[Product.SQUID_INK],
                    traderObject,
                    0,  # We're not using take orders for SQUID_INK, focusing on market making only
                    0
                )
                
                result[Product.SQUID_INK] = ink_mm_orders

        
        """ 
        Tidying up
        """

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData