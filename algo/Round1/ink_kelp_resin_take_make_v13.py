from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
import jsonpickle
import numpy as np
from typing import Any, List
from collections import deque

# Added necessary imports
import math

"""
GRID SEARCH param. Leave this equal to None to initialize it.

Wherever you want to perform a grid search, replace that with = grid_search

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
        # Check if state is None or if state.traderData is not initialized
        state_trader_data = state.traderData if state is not None else ""
        
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""), # Pass empty string for traderData initially
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
                    self.compress_state(state, self.truncate(state_trader_data, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Handle None state gracefully
        if state is None:
             return [0, trader_data, [], {}, {}, {}, {}, {}] # Return default empty values
        
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
        if observations is None: # Handle None observations
            return [{}, {}]

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
        # Use json.dumps with ProsperityEncoder directly
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
    
    
# Parameters - Adjusted for aggressive strategy combining v6 and v13 approaches
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "clear_width": 0
    },
    Product.KELP: {
        "fair_value": 2000,
        "ewma_beta": 0.05,  # Keep slight smoothing for stability
        "min_spread": 2,    # Tighter spreads to capture more trades
        "max_spread": 5,    # Allow wider spreads in volatile conditions
        "position_scale": 0.3, # Less aggressive scaling to maintain larger positions
        "base_order_size": 25, # Larger order size to capture more profit
        "history_maxlen": 25,  # Keep more history for better analysis
        "spread_multiplier": 1.1, # Lower multiplier for tighter spreads
        "vol_window": 10,        # Window for calculating volatility-based spread
        "order_skew_threshold": 0.8, # Lower threshold to react to smaller imbalances
        "imbalance_multiplier": 0.5, # How much to adjust prices based on imbalance
        "autocorr_weight": 0.3,   # How much to weight autocorrelation signals
        "inventory_scale_factor": 0.3 # Scale orders based on inventory
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.15,  # Slightly more adaptive
        "min_spread": 2,   # Tighter spreads to capture more trades
        "max_spread": 5,   # Allow wider spreads in volatile conditions
        "position_scale": 0.3, # Less aggressive scaling to maintain larger positions
        "base_order_size": 25, # Larger order size to capture more profit
        "history_maxlen": 50, # Keep more history for better analysis
        "spread_multiplier": 1.2, # Lower multiplier for tighter spreads
        "vol_window": 30,        # Window for calculating volatility-based spread
        "order_skew_threshold": 0.8, # Lower threshold to react to smaller imbalances  
        "imbalance_multiplier": 0.5, # How much to adjust prices based on imbalance
        "autocorr_weight": 0.15,   # How much to weight autocorrelation signals
        "inventory_scale_factor": 0.55 # Scale orders based on inventory
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
            best_ask_amount = -1 * order_depth.sell_orders[best_ask] # Volume is typically negative in sell orders

        return best_bid, best_ask, best_bid_amount, best_ask_amount
    
    
    def mm_orders(
            self,
            product: str,
            order_depth: OrderDepth
            ) -> (float, float, int, int):
        
        """
        Helper function used to find the widest bid/ask prices and volumes, the bot market maker
        """  
        mm_bid = mm_ask = mm_bid_amount = mm_ask_amount = None
        
        if len(order_depth.buy_orders) != 0:
            mm_bid = min(order_depth.buy_orders.keys())
            mm_bid_amount = order_depth.buy_orders[mm_bid]
            
        if len(order_depth.sell_orders) != 0:
            mm_ask = max(order_depth.sell_orders.keys())
            mm_ask_amount = -1 * order_depth.sell_orders[mm_ask] # Volume is typically negative in sell orders
    
        return mm_bid, mm_ask, mm_bid_amount, mm_ask_amount
    
    def rolling_mm_mid_quotes(
            self,
            product: str,
            order_depth: OrderDepth,
            traderObject: dict
            ):
        
        # Initialize the rolling_mid_quotes dictionary if it doesn't exist
        if "rolling_mid_quotes" not in traderObject:
            traderObject["rolling_mid_quotes"] = {}
        
        # Initialize the list for this product if it doesn't exist
        if product not in traderObject["rolling_mid_quotes"]:
             # Use maxlen from params, default to 100 if not found
            max_len = self.params[product].get("history_maxlen", 100)
            traderObject["rolling_mid_quotes"][product] = deque(maxlen=max_len)
        
        rolling_prices = traderObject["rolling_mid_quotes"][product]
        
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
            rolling_prices.append(mid)
        
        return rolling_prices
    
    
    def order_flow_imbalance(
        self,
        product: str,
        order_depth: OrderDepth,
        traderObject: dict
    ) -> float:
        
        """
        Helper function used to find the order flow imbalance from current data
        and previous data - useful for predicting short-term price movements
        """
        if "prev_quote" not in traderObject or product not in traderObject["prev_quote"] or not traderObject["prev_quote"][product]:
            return 0.0
            
        B_n, A_n, q_B_n, q_A_n = self.best_orders(product, order_depth)
        
        if B_n is None or A_n is None:
            return 0.0
            
        B_n_minus_1, A_n_minus_1, q_B_n_minus_1, q_A_n_minus_1 = traderObject["prev_quote"][product]
        
        if B_n_minus_1 is None or A_n_minus_1 is None or q_B_n_minus_1 is None or q_A_n_minus_1 is None:
            return 0.0
        
        # Calculate indicators for price movements
        I_B_increase = int(B_n > B_n_minus_1)  # 1 if best bid increases, else 0
        I_B_decrease = int(B_n < B_n_minus_1)  # 1 if best bid decreases, else 0
        I_A_increase = int(A_n > A_n_minus_1)  # 1 if best ask increases, else 0
        I_A_decrease = int(A_n < A_n_minus_1)  # 1 if best ask decreases, else 0

        # Calculate order flow imbalance - positive values indicate buying pressure
        e_n = (I_B_increase * q_B_n 
               - I_B_decrease * q_B_n_minus_1 
               - I_A_decrease * q_A_n 
               + I_A_increase * q_A_n_minus_1)

        return float(e_n)  # Ensure we return a float

    def previous_midprice(
            self,
            product: str,
            traderObject: dict
            ) -> float:
        """
        Helper function used to calculate the previous tick's mid price
        """  
        
        if "prev_quote" not in traderObject or product not in traderObject["prev_quote"]:
            return None
            
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
            product: str,
            order_depth: OrderDepth
            ) -> float:
   
        """
        Computes the full book volume-weighted mid-price using *all* buy and sell orders.
        More stable than simple mid price and accounts for volume imbalance.
        """
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return None
            
        # Convert to numpy arrays for faster computation
        buy_prices = np.array(list(order_depth.buy_orders.keys()))
        buy_volumes = np.array(list(order_depth.buy_orders.values()))
        sell_prices = np.array(list(order_depth.sell_orders.keys()))
        sell_volumes = np.abs(np.array(list(order_depth.sell_orders.values())))
        
        # Calculate weighted sum of prices
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
            
        # Initialize EWMA dictionary if not present
        if "ewma" not in traderObject:
            traderObject["ewma"] = {}
            
        prev_ewma = traderObject["ewma"].get(product)

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
        beta: float  # smoothing factor for EWMA variance
        ) -> float:
        
        """
        Calculates exponentially weighted volatility (standard deviation) of price *changes*.
        """
        # Access or initialize tracking variables
        if "vol" not in traderObject:
            traderObject["vol"] = {}
        if product not in traderObject["vol"]:
             # Use maxlen from params, default to 100 if not found
            max_len = self.params[product].get("history_maxlen", 100)
            traderObject["vol"][product] = {
                "price_history": deque(maxlen=max_len), # Use deque for efficiency
                "ewma_var": 0.0
            }
        
        vol_data = traderObject["vol"][product]
        
        # Append current price to history
        vol_data["price_history"].append(mid)
        
        # Calculate price differences and update variance
        if len(vol_data["price_history"]) > 1:
             # Calculate most recent price change
            diff = vol_data["price_history"][-1] - vol_data["price_history"][-2]
            squared_diff = diff**2
            
            # Update EWMA variance
            vol_data["ewma_var"] = beta * vol_data["ewma_var"] + (1 - beta) * squared_diff
        
        # Return standard deviation
        return np.sqrt(vol_data["ewma_var"])

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int
        ) -> (List[Order], int, int):
        """
        Takes orders from the order book when there's clear profit to be made.
        Returns buy/sell orders when there are acceptable prices in the book.
        """
        orders = []
        take_buy_quantity = take_sell_quantity = 0
        
        # Check for any sell orders below fair_value (we should buy these)
        if len(order_depth.sell_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < fair_value:  # Profitable to buy
                best_ask_volume = -order_depth.sell_orders[best_ask]  # Convert to positive
                # Limit by position
                max_buy_volume = min(
                    best_ask_volume,
                    position_limit - position
                )
                if max_buy_volume > 0:
                    orders.append(Order(product, best_ask, max_buy_volume))
                    take_buy_quantity = max_buy_volume
        
        # Check for any buy orders above fair_value (we should sell to these)
        if len(order_depth.buy_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > fair_value:  # Profitable to sell
                best_bid_volume = order_depth.buy_orders[best_bid]
                # Limit by position (negative position means short selling)
                max_sell_volume = min(
                    best_bid_volume,
                    position_limit + position
                )
                if max_sell_volume > 0:
                    orders.append(Order(product, best_bid, -max_sell_volume))
                    take_sell_quantity = max_sell_volume
        
        return orders, take_buy_quantity, take_sell_quantity

    def resin_make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        required_edge: float,
        default_order_size: int,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        """
        Places maker orders for Rainforest Resin with specified edge.
        This is specialized for resin with a higher edge requirement.
        """
        orders = []
        make_buy_quantity = make_sell_quantity = 0
        
        # Calculate how many more we can buy/sell
        remaining_buy = position_limit - position - take_buy_quantity
        remaining_sell = position_limit + position - take_sell_quantity
        
        # Get current best prices
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        
        # Only place orders if we can improve the spread
        if best_bid is not None and best_ask is not None:
            # Place buy order with required edge below fair_value
            bid_price = int(fair_value - required_edge)
            if bid_price > best_bid and remaining_buy > 0:  # Only if we improve best bid
                buy_quantity = min(default_order_size, remaining_buy)
                orders.append(Order(product, bid_price, buy_quantity))
                make_buy_quantity = buy_quantity
            
            # Place sell order with required edge above fair_value
            ask_price = int(fair_value + required_edge)
            if ask_price < best_ask and remaining_sell > 0:  # Only if we improve best ask
                sell_quantity = min(default_order_size, remaining_sell)
                orders.append(Order(product, ask_price, -sell_quantity))
                make_sell_quantity = sell_quantity
        
        return orders, make_buy_quantity, make_sell_quantity

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        clear_width: float,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        """
        Clears out extremely favorable orders, cleaning up anything our take_orders missed.
        clear_width specifies how far from fair_value we're willing to accept.
        """
        orders = []
        clear_buy_quantity = clear_sell_quantity = 0
        
        # If clear_width is 0, skip this function entirely
        if clear_width == 0:
            return orders, clear_buy_quantity, clear_sell_quantity
        
        # Available position for trading
        available_buy = position_limit - position - take_buy_quantity
        available_sell = position_limit + position - take_sell_quantity
        
        if available_buy > 0 and len(order_depth.sell_orders) > 0:
            sell_prices = np.array(list(order_depth.sell_orders.keys()))
            # Vectorized check for prices within clear range
            good_asks = sell_prices[sell_prices < fair_value + clear_width]
            
            # Process each good ask price
            for ask_price in sorted(good_asks):
                ask_volume = -order_depth.sell_orders[ask_price]  # Convert to positive
                buy_volume = min(ask_volume, available_buy)
                
                if buy_volume > 0:
                    orders.append(Order(product, ask_price, buy_volume))
                    clear_buy_quantity += buy_volume
                    available_buy -= buy_volume
                    
                    if available_buy <= 0:
                        break
        
        if available_sell > 0 and len(order_depth.buy_orders) > 0:
            bid_prices = np.array(list(order_depth.buy_orders.keys()))
            # Vectorized check for prices within clear range
            good_bids = bid_prices[bid_prices > fair_value - clear_width]
            
            # Process each good bid price
            for bid_price in sorted(good_bids, reverse=True):
                bid_volume = order_depth.buy_orders[bid_price]
                sell_volume = min(bid_volume, available_sell)
                
                if sell_volume > 0:
                    orders.append(Order(product, bid_price, -sell_volume))
                    clear_sell_quantity += sell_volume
                    available_sell -= sell_volume
                    
                    if available_sell <= 0:
                        break
        
        return orders, clear_buy_quantity, clear_sell_quantity

    # ... (ink_fair_value, make_orders remain the same) ...
    
    def calculate_optimal_spread(self, product, traderObject, current_spread):
        """
        Calculate the optimal spread based on recent price volatility
        """
        if "vol" not in traderObject or product not in traderObject["vol"]:
             return self.params[product]["min_spread"]
            
        vol_data = traderObject["vol"][product]
        
        # Use the calculated EWMA volatility if available
        ewma_vol = np.sqrt(vol_data.get("ewma_var", 0.0))

        # Scale the spread based on EWMA volatility
        # Add a small constant to avoid zero volatility leading to zero spread
        optimal_spread = max(
            self.params[product]["min_spread"],
            min(
                self.params[product]["max_spread"],
                int((ewma_vol + 0.1) * self.params[product]["spread_multiplier"]) # Add 0.1 to prevent spread of 0 with low vol
            )
        )
        
        # Ensure spread is at least the minimum spread
        optimal_spread = max(optimal_spread, self.params[product]["min_spread"])
        
        return optimal_spread

    def calculate_book_imbalance(self, order_depth):
        """
        Calculate order book imbalance to determine market direction pressure
        """
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return 0.0  # Return float 0.0 for consistency
            
        # Sum buy and sell volumes at the best levels
        # Consider only the best levels to gauge immediate pressure
        buy_volume = order_depth.buy_orders.get(max(order_depth.buy_orders.keys()), 0) if order_depth.buy_orders else 0
        sell_volume = abs(order_depth.sell_orders.get(min(order_depth.sell_orders.keys()), 0) if order_depth.sell_orders else 0)

        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance  # Range: [-1.0, 1.0]


    def analyze_price_autocorrelation(self, product: str, traderObject: dict, current_price: float) -> tuple[float, float, float]:
        """
        Analyze price movements to detect mean reversion patterns and determine optimal price adjustments
        Returns:
            - correlation_strength: measure of negative autocorrelation [-1 to 1]
            - bid_adjustment: suggested adjustment to bid price based on mean reversion
            - ask_adjustment: suggested adjustment to ask price based on mean reversion
        """
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
            
        if product not in traderObject["price_history"]:
            max_len = self.params[product].get("history_maxlen", 100)
            traderObject["price_history"][product] = deque(maxlen=max_len)
            return 0.0, 0.0, 0.0
            
        price_history = traderObject["price_history"][product]
        
        # Need at least 20 prices for meaningful autocorrelation analysis
        if len(price_history) < 20:
            return 0.0, 0.0, 0.0
            
        # Convert to numpy array for efficient calculations
        prices = np.array(list(price_history))
        
        # Calculate returns
        returns = np.diff(prices)
        
        # Need at least 10 returns for lag-1 autocorrelation
        if len(returns) < 10:
            return 0.0, 0.0, 0.0
            
        # Calculate lag-1 autocorrelation using numpy
        n = len(returns)
        mean_return = np.mean(returns)
        
        # Calculate autocorrelation
        try:
            numerator = np.sum((returns[1:] - mean_return) * (returns[:-1] - mean_return))
            denominator = np.sum((returns - mean_return) ** 2)
            
            if denominator == 0:
                autocorr = 0.0
            else:
                autocorr = numerator / denominator
        except:
            return 0.0, 0.0, 0.0
        
        # Recent price movement direction
        recent_movement = returns[-1] if returns.size > 0 else 0.0
        
        # Calculate appropriate adjustments based on autocorrelation
        # For mean reversion strategies, we focus on negative autocorrelation
        bid_adjustment = 0.0
        ask_adjustment = 0.0
        
        # If prices just went up and we have negative autocorrelation,
        # they'll likely go down next
        if recent_movement > 0 and autocorr < 0:
            bid_adjustment = -recent_movement * abs(autocorr)  # Lower bid
            ask_adjustment = -recent_movement * abs(autocorr)  # Lower ask
        # If prices just went down and we have negative autocorrelation,
        # they'll likely go up next  
        elif recent_movement < 0 and autocorr < 0:
            bid_adjustment = -recent_movement * abs(autocorr)  # Higher bid
            ask_adjustment = -recent_movement * abs(autocorr)  # Higher ask
        
        # Focus only on negative autocorrelation (mean reversion)
        correlation_strength = min(autocorr, 0.0)
            
        return correlation_strength, bid_adjustment, ask_adjustment

    def calculate_tick_size(self, price_history):
        """
        Estimate the minimum price increment (tick size) from price history
        """
        if not price_history or len(price_history) < 10:
            return 1  # Default to 1 if not enough data
        
        # Get unique prices
        prices = np.array(list(price_history))
        unique_prices = np.unique(prices)
        
        if len(unique_prices) <= 1:
            return 1  # Default if all prices are the same
        
        # Calculate differences between consecutive prices
        sorted_prices = np.sort(unique_prices)
        diffs = np.diff(sorted_prices)
        
        # Find the minimum non-zero difference
        non_zero_diffs = diffs[diffs > 0]
        if len(non_zero_diffs) == 0:
            return 1
        
        min_tick = np.min(non_zero_diffs)
        
        # Round to the nearest common tick size (0.01, 0.1, 1, 2, 5, etc.)
        if min_tick < 0.01:
            return 0.01
        elif min_tick < 0.1:
            return 0.1
        elif min_tick < 1:
            return 1
        else:
            # For larger ticks, try to find clean values (1, 2, 5, 10, etc.)
            for standard_tick in [1, 2, 5, 10, 20, 50, 100]:
                if abs(min_tick - standard_tick) / standard_tick < 0.1:  # Within 10%
                    return standard_tick
            
            # Otherwise return the actual minimum tick size
            return min_tick

    def get_optimal_mid_price(self, order_depth: OrderDepth, current_mid: float, position: int, position_limit: int, product: str, traderObject: dict) -> float:
        """
        Calculates an optimal fair value influenced by book imbalance and autocorrelation.
        """
        
        imbalance = self.calculate_book_imbalance(order_depth)
        # logger.print(f"{product} imbalance: {imbalance:.4f}")

        # Analyze price autocorrelation
        # Ensure traderObject["price_history"] is initialized
        if "price_history" not in traderObject or product not in traderObject["price_history"]:
            traderObject["price_history"][product] = deque(maxlen=self.params[product].get("history_maxlen", 100))
        
        # Get price history for autocorrelation analysis
        price_history = traderObject["price_history"][product]

        # Add current mid price to history (important!)
        price_history.append(current_mid)

        corr_strength, bid_adj, ask_adj = self.analyze_price_autocorrelation(
            product, traderObject, current_mid
        )
        
        # logger.print(f"{product} Corr: {corr_strength:.4f}, Bid Adj: {bid_adj:.4f}, Ask Adj: {ask_adj:.4f}")

        # Combine imbalance and autocorrelation adjustments
        # Use a weighting factor for each influence
        imbalance_weight = 0.5 # How much imbalance influences the fair value
        autocorr_weight = 0.5 # How much autocorrelation influences the fair value
        
        # Determine net adjustment based on signal
        # Positive imbalance suggests upward pressure, negative suggests downward
        imbalance_adj = imbalance * imbalance_weight * (max(order_depth.sell_orders.keys()) - min(order_depth.buy_orders.keys())) / 2 # Scale by half spread
        
        # Autocorrelation adjustments work in the opposite direction of recent movement
        # Negative correlation: If price went up, expect it to go down. Adjust fair value down.
        # If price went down, expect it to go up. Adjust fair value up.
        
        # Get recent movement from history
        recent_movement = 0.0
        if len(price_history) >= 2:
            recent_movement = price_history[-1] - price_history[-2]

        # Adjust fair value based on expected reversion (opposite of recent movement)
        autocorr_fair_adj = 0.0
        if corr_strength < -0.2: # Only apply if significant negative autocorrelation
           autocorr_fair_adj = -recent_movement * abs(corr_strength) * autocorr_weight * 0.5 # Apply with a small factor


        # Combine current mid price with adjustments
        optimal_mid = current_mid + imbalance_adj + autocorr_fair_adj
        
        # Incorporate inventory pressure (move fair value towards the limit)
        # If positive position, fair value should be higher (encourage selling)
        # If negative position, fair value should be lower (encourage buying)
        inventory_pressure_factor = 0.1 # How much inventory affects fair value
        inventory_adj = position / position_limit * inventory_pressure_factor * (max(order_depth.sell_orders.keys()) - min(order_depth.buy_orders.keys())) / 2
        optimal_mid += inventory_adj

        return optimal_mid


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
        Aggressive market making strategy for SQUID_INK optimized for maximum returns
        """
        orders = []
        
        # Get current mid price from best orders
        best_bid, best_ask, best_bid_amt, best_ask_amt = self.best_orders(product, order_depth)
        if best_bid is None or best_ask is None:
             return [], 0, 0
        
        current_mid = (best_bid + best_ask) / 2
        
        # Store price in history for volatility and autocorrelation calculations
        if product not in traderObject.get("price_history", {}):
            if "price_history" not in traderObject:
                traderObject["price_history"] = {}
            traderObject["price_history"][product] = deque(maxlen=self.params[product]["history_maxlen"])
        
        traderObject["price_history"][product].append(current_mid)
        
        # Update volatility estimate for dynamic spread calculation
        volatility = self.ewma_volatility(product, traderObject, current_mid, 0.94)
        
        # Calculate book imbalance (demand vs supply)
        imbalance = self.calculate_book_imbalance(order_depth)
        
        # Look for price autocorrelation (mean reversion)
        corr_strength, bid_adj, ask_adj = self.analyze_price_autocorrelation(
            product, traderObject, current_mid
        )
        
        # Calculate optimal fair value based on all signals
        # Start with EWMA price as baseline
        optimal_fair_value = self.ewma(product, traderObject, current_mid, self.params[product]["ewma_beta"])
        if optimal_fair_value is None:
            optimal_fair_value = current_mid
        
        # Adjust fair value based on order book imbalance - more aggressively than Kelp
        imbalance_adjustment = imbalance * (best_ask - best_bid) * self.params[product]["imbalance_multiplier"] * 1.2
        optimal_fair_value += imbalance_adjustment
        
        # Get estimated tick size for this product
        tick_size = self.calculate_tick_size(traderObject["price_history"][product])
        
        # Calculate optimal spread based on volatility - even more aggressive with tighter spreads
        # than Kelp for higher turnover
        optimal_spread = max(
            self.params[product]["min_spread"] * tick_size,
            min(
                self.params[product]["max_spread"] * tick_size,
                volatility * self.params[product]["spread_multiplier"] * 0.8  # 20% tighter than formula for Kelp
            )
        )
        
        # Calculate base prices
        bid_price = int(optimal_fair_value - optimal_spread/2)
        ask_price = int(optimal_fair_value + optimal_spread/2)
        
        # Apply autocorrelation-based adjustments for aggressive mean reversion strategies
        # More responsive to autocorrelation than Kelp
        if corr_strength < -0.15:  # Lower threshold than Kelp (-0.2)
            bid_price += int(bid_adj * self.params[product]["autocorr_weight"] * 1.2)
            ask_price += int(ask_adj * self.params[product]["autocorr_weight"] * 1.2)
        
        # Apply aggressive position-based adjustments - smaller adjustment when near zero
        # larger adjustment when nearing limits to ensure mean reversion to zero
        position_ratio = position / position_limit
        inventory_skew = position_ratio * optimal_spread * self.params[product]["inventory_scale_factor"]
        
        # Shift both bid and ask in the direction that reduces inventory
        bid_price -= int(inventory_skew)
        ask_price -= int(inventory_skew)
        
        # Dynamic order sizing - aggressively scale up size when expecting favorable price moves
        # Base size is larger than conservative approach
        base_size = self.params[product]["base_order_size"]  # Already set to 25 in parameters
        
        # Scale order sizes based on signals:
        # 1. Larger orders when we're confident about directional movement
        # 2. Smaller orders when near position limits
        
        # Position scaling factor - reduce size as position grows in either direction
        # Less reduction than Kelp to maintain larger positions
        position_scale = 1.0 - abs(position_ratio) * self.params[product]["position_scale"] * 0.8
        
        # Signal-based sizing: larger sizes when signals strongly favor a direction
        # More aggressive than Kelp
        signal_adjustment = 1.0
        
        # If imbalance and autocorrelation agree on direction, increase size more than Kelp
        if (imbalance > 0.25 and bid_adj > 0) or (imbalance < -0.25 and ask_adj < 0):
            signal_adjustment = 1.5  # 50% size increase when signals align (vs 30% for Kelp)
        
        # Calculate final sizes with aggressive scaling
        buy_size = int(base_size * position_scale * signal_adjustment)
        sell_size = int(base_size * position_scale * signal_adjustment)
        
        # Further skew sizes based on position - aggressively reduce oversized positions
        # but less aggressively than Kelp to maintain larger positions when profitable
        if position > 0:
            buy_size = int(buy_size * (1 - position_ratio * 0.7))  # Less reduction than Kelp (0.8)
            sell_size = int(sell_size * (1 + position_ratio * 0.6))  # More increase than Kelp (0.5)
        elif position < 0:
            buy_size = int(buy_size * (1 + abs(position_ratio) * 0.6))  # More increase than Kelp (0.5)
            sell_size = int(sell_size * (1 - abs(position_ratio) * 0.7))  # Less reduction than Kelp (0.8)
            
        # Calculate available capacity
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Ensure minimum sizes and respect capacity
        buy_size = max(1, min(buy_size, max_buy_capacity))
        sell_size = max(1, min(sell_size, max_sell_capacity))
        
        # Always improve the best bid/ask - aggressive penny jumping by 2 ticks when possible
        if best_bid is not None and bid_price <= best_bid:
            bid_price = best_bid + min(2, int(tick_size))  # Jump by 2 ticks when possible
            
        if best_ask is not None and ask_price >= best_ask:
            ask_price = best_ask - min(2, int(tick_size))  # Jump by 2 ticks when possible
        
        # Place orders if we have capacity and they're sensible
        if buy_size > 0 and bid_price < ask_price:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0 and ask_price > bid_price:
            orders.append(Order(product, ask_price, -sell_size))
        
        return orders, buy_size, sell_size

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
        Aggressive market making strategy for KELP optimized for maximum returns
        """
        orders = []
        
        # Get current mid price from best orders
        best_bid, best_ask, best_bid_amt, best_ask_amt = self.best_orders(product, order_depth)
        if best_bid is None or best_ask is None:
             return [], 0, 0
        
        current_mid = (best_bid + best_ask) / 2
        
        # Store price in history for volatility and autocorrelation calculations
        if product not in traderObject.get("price_history", {}):
            if "price_history" not in traderObject:
                traderObject["price_history"] = {}
            traderObject["price_history"][product] = deque(maxlen=self.params[product]["history_maxlen"])
        
        traderObject["price_history"][product].append(current_mid)
        
        # Update volatility estimate for dynamic spread calculation
        volatility = self.ewma_volatility(product, traderObject, current_mid, 0.94)
        
        # Calculate book imbalance (demand vs supply)
        imbalance = self.calculate_book_imbalance(order_depth)
        
        # Look for price autocorrelation (mean reversion)
        corr_strength, bid_adj, ask_adj = self.analyze_price_autocorrelation(
            product, traderObject, current_mid
        )
        
        # Calculate optimal fair value based on all signals
        # Start with EWMA price as baseline
        optimal_fair_value = self.ewma(product, traderObject, current_mid, self.params[product]["ewma_beta"])
        if optimal_fair_value is None:
            optimal_fair_value = current_mid
        
        # Adjust fair value based on order book imbalance
        imbalance_adjustment = imbalance * (best_ask - best_bid) * self.params[product]["imbalance_multiplier"]
        optimal_fair_value += imbalance_adjustment
        
        # Get estimated tick size for this product
        tick_size = self.calculate_tick_size(traderObject["price_history"][product])
        
        # Calculate optimal spread based on volatility - more aggressive with tighter spreads
        # but allowing wider spreads when volatility increases
        optimal_spread = max(
            self.params[product]["min_spread"] * tick_size,
            min(
                self.params[product]["max_spread"] * tick_size,
                volatility * self.params[product]["spread_multiplier"]
            )
        )
        
        # Calculate base prices
        bid_price = int(optimal_fair_value - optimal_spread/2)
        ask_price = int(optimal_fair_value + optimal_spread/2)
        
        # Apply autocorrelation-based adjustments for aggressive mean reversion strategies
        if corr_strength < -0.2:  # Only if significant negative autocorrelation
            bid_price += int(bid_adj * self.params[product]["autocorr_weight"])
            ask_price += int(ask_adj * self.params[product]["autocorr_weight"])
        
        # Apply aggressive position-based adjustments - smaller adjustment when near zero
        # larger adjustment when nearing limits to ensure mean reversion to zero
        position_ratio = position / position_limit
        inventory_skew = position_ratio * optimal_spread * self.params[product]["inventory_scale_factor"]
        
        # Shift both bid and ask in the direction that reduces inventory
        bid_price -= int(inventory_skew)
        ask_price -= int(inventory_skew)
        
        # Dynamic order sizing - aggressively scale up size when expecting favorable price moves
        # Base size is larger than conservative approach
        base_size = self.params[product]["base_order_size"]  # Already set to 25 in parameters
        
        # Scale order sizes based on signals:
        # 1. Larger orders when we're confident about directional movement
        # 2. Smaller orders when near position limits
        
        # Position scaling factor - reduce size as position grows in either direction
        position_scale = 1.0 - abs(position_ratio) * self.params[product]["position_scale"]
        
        # Signal-based sizing: larger sizes when signals strongly favor a direction
        signal_adjustment = 1.0
        
        # If imbalance and autocorrelation agree on direction, increase size
        if (imbalance > 0.3 and bid_adj > 0) or (imbalance < -0.3 and ask_adj < 0):
            signal_adjustment = 1.3  # 30% size increase when signals align
        
        # Calculate final sizes with aggressive scaling
        buy_size = int(base_size * position_scale * signal_adjustment)
        sell_size = int(base_size * position_scale * signal_adjustment)
        
        # Further skew sizes based on position - aggressively reduce oversized positions
        if position > 0:
            buy_size = int(buy_size * (1 - position_ratio * 0.8))
            sell_size = int(sell_size * (1 + position_ratio * 0.5)) # Slightly increase sell size
        elif position < 0:
            buy_size = int(buy_size * (1 + abs(position_ratio) * 0.5))
            sell_size = int(sell_size * (1 - abs(position_ratio) * 0.8))
            
        # Calculate available capacity
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Ensure minimum sizes and respect capacity
        buy_size = max(1, min(buy_size, max_buy_capacity))
        sell_size = max(1, min(sell_size, max_sell_capacity))
        
        # Always improve the best bid/ask - aggressive penny jumping
        if best_bid is not None and bid_price <= best_bid:
            bid_price = best_bid + 1
            
        if best_ask is not None and ask_price >= best_ask:
            ask_price = best_ask - 1
        
        # Place orders if we have capacity and they're sensible
        if buy_size > 0 and bid_price < ask_price:
            orders.append(Order(product, bid_price, buy_size))
            
        if sell_size > 0 and ask_price > bid_price:
            orders.append(Order(product, ask_price, -sell_size))
        
        return orders, buy_size, sell_size

    def ink_fair_value(
        self,
        product: str,
        traderObject: dict,
        kelp_lag: int = 5  # Default lag for observing correlation between products
        ) -> float:
        """
        Compute potential fair value for Squid Ink based on Kelp price movement
        This leverages potential cross-product correlations
        """
        kelp_prices = traderObject.get("rolling_mid_quotes", {}).get(Product.KELP)
        ink_prices = traderObject.get("rolling_mid_quotes", {}).get(Product.SQUID_INK)
        
        if not kelp_prices or not ink_prices or len(kelp_prices) <= kelp_lag or len(ink_prices) < 2:
            return None
        
        # Convert to numpy arrays for efficient calculation
        kelp_array = np.array(list(kelp_prices))
        ink_array = np.array(list(ink_prices))
        
        # Calculate if there's strong positive correlation between lagged kelp and current ink
        if len(kelp_array) > kelp_lag + 10 and len(ink_array) > 10:
            # Use lagged kelp prices
            lagged_kelp = kelp_array[:-kelp_lag]
            # Use matching ink prices
            matching_ink = ink_array[kelp_lag:]
            
            # Only proceed if we have enough data points
            if len(lagged_kelp) > 10:
                # Calculate correlation coefficient
                try:
                    correlation = np.corrcoef(lagged_kelp, matching_ink)[0, 1]
                    
                    # If strong positive correlation, use lagged kelp to predict ink
                    if correlation > 0.7:  # Threshold for strong correlation
                        # Predict ink based on recent kelp movement
                        recent_kelp_change = (kelp_array[-1] - kelp_array[-kelp_lag-1]) / kelp_array[-kelp_lag-1]
                        predicted_ink = ink_array[-1] * (1 + recent_kelp_change * 0.8)  # Dampen effect
                        return predicted_ink
                except:
                    pass  # If correlation calculation fails, fall back to default
        
        # Default: Use recent ink prices directly
        recent_ink = ink_array[-min(10, len(ink_array)):]
        return np.mean(recent_ink)  # Simple average of recent prices

    def analyze_trend_strength(self, product: str, traderObject: dict) -> float:
        """
        Analyze price trends to determine strength and direction
        Returns a value between -1 (strong downtrend) and 1 (strong uptrend)
        """
        if "price_history" not in traderObject or product not in traderObject["price_history"]:
            return 0.0
            
        price_history = traderObject["price_history"][product]
        
        if len(price_history) < 10:  # Need at least 10 data points
            return 0.0
            
        # Use numpy for efficient calculations
        prices = np.array(list(price_history))
        
        # Simple trend calculation: compare recent average to overall average
        short_term_avg = np.mean(prices[-5:])  # Last 5 prices
        medium_term_avg = np.mean(prices[-20:] if len(prices) >= 20 else prices)  # Last 20 or all
        
        # Calculate trend direction and normalize to [-1, 1]
        max_diff = np.std(prices) * 2  # Use 2 standard deviations as max diff
        if max_diff == 0:  # Avoid division by zero
            return 0.0
            
        trend = (short_term_avg - medium_term_avg) / max_diff
        return max(min(trend, 1.0), -1.0)  # Clamp between -1 and 1
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        # Load or initialize trader data
        if state.traderData:
             traderObject = jsonpickle.decode(state.traderData)
        else:
            traderObject = {
                "prev_quote": {p: [] for p in [Product.KELP, Product.RAINFOREST_RESIN, Product.SQUID_INK]},
                "ewma": {p: None for p in [Product.KELP, Product.RAINFOREST_RESIN, Product.SQUID_INK]},
                "rolling_mid_quotes": {},
                "price_history": {}
            }

        # Initialize price history deque for all relevant products if not already done
        for product in [Product.KELP, Product.SQUID_INK]:
             if product not in traderObject.get("price_history", {}):
                  if "price_history" not in traderObject: traderObject["price_history"] = {}
                  max_len = self.params[product].get("history_maxlen", 100)
                  traderObject["price_history"][product] = deque(maxlen=max_len)

             # Also ensure volatility tracking structure exists
             if "vol" not in traderObject: traderObject["vol"] = {}
             if product not in traderObject["vol"]:
                  max_len = self.params[product].get("history_maxlen", 100)
                  traderObject["vol"][product] = {
                       "price_history": deque(maxlen=max_len),
                       "ewma_var": 0.0
                  }


        """ Rainforest Resin Strategy (Unchanged) """
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            resin_best = self.best_orders(Product.RAINFOREST_RESIN, resin_order_depth)
            traderObject["prev_quote"][Product.RAINFOREST_RESIN] = resin_best
            
            resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.take_orders(
                Product.RAINFOREST_RESIN,
                resin_order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                self.LIMIT[Product.RAINFOREST_RESIN]
            )
            
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
            
            result[Product.RAINFOREST_RESIN] = (resin_take_orders + resin_make_orders + resin_clear_orders)
        
        """ Kelp Strategy """
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_order_depth = state.order_depths[Product.KELP]
            
            kelp_best = self.best_orders(Product.KELP, kelp_order_depth)
            traderObject["prev_quote"][Product.KELP] = kelp_best
            
            # Use current best bid/ask to get mid-price for EWMA and history
            best_bid_kelp, best_ask_kelp, _, _ = kelp_best
            if best_bid_kelp is not None and best_ask_kelp is not None:
                kelp_mid = (best_bid_kelp + best_ask_kelp) / 2
                
                # Update volatility estimate
                self.ewma_volatility(Product.KELP, traderObject, kelp_mid, self.params[Product.KELP]["ewma_beta"])
                
                # Update EWMA fair value (optional, could rely purely on dynamic mid)
                # kelp_fair_value = self.ewma(Product.KELP, traderObject, kelp_mid, self.params[Product.KELP]["ewma_beta"])
                # Use current mid as base for dynamic calculation
                kelp_fair_value_base = kelp_mid

                # Run the enhanced market making strategy
                kelp_mm_orders, _, _ = self.kelp_market_making(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value_base,
                    kelp_position,
                    self.LIMIT[Product.KELP],
                    traderObject,
                    0, # No separate take orders
                    0
                )
                
                result[Product.KELP] = kelp_mm_orders
            else:
                 # If no best bid/ask, simply place no orders for this product
                 result[Product.KELP] = []

        """ Squid Ink Strategy """
        
        if Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_order_depth = state.order_depths[Product.SQUID_INK]
            
            ink_best = self.best_orders(Product.SQUID_INK, ink_order_depth)
            traderObject["prev_quote"][Product.SQUID_INK] = ink_best
            
            # Use current best bid/ask to get mid-price for EWMA and history
            best_bid_ink, best_ask_ink, _, _ = ink_best
            if best_bid_ink is not None and best_ask_ink is not None:
                ink_mid = (best_bid_ink + best_ask_ink) / 2

                # Update volatility estimate
                self.ewma_volatility(Product.SQUID_INK, traderObject, ink_mid, self.params[Product.SQUID_INK]["ewma_beta"])

                # Update EWMA fair value (optional)
                # ink_fair_value = self.ewma(Product.SQUID_INK, traderObject, ink_mid, self.params[Product.SQUID_INK]["ewma_beta"])
                # Use current mid as base for dynamic calculation
                ink_fair_value_base = ink_mid
                
                # Run the enhanced market making strategy
                ink_mm_orders, _, _ = self.squid_ink_market_making(
                    Product.SQUID_INK,
                    ink_order_depth,
                    ink_fair_value_base,
                    ink_position,
                    self.LIMIT[Product.SQUID_INK],
                    traderObject,
                    0,  # No separate take orders
                    0
                )
                
                result[Product.SQUID_INK] = ink_mm_orders
            else:
                 # If no best bid/ask, simply place no orders for this product
                 result[Product.SQUID_INK] = []

        
        """ 
        Tidying up
        """

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData