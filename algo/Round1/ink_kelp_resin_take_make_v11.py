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
    
    
# Parameters - Adjusted based on analysis (e.g., tighter spreads, more stable EWMA, smaller orders)
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "clear_width": 0
    },
    Product.KELP: {
        "fair_value": 2000,
        "ewma_beta": 0.10, # Adjusted based on analysis for stability
        "clear_width": 0,
        "spread_multiplier": 1.5,  # Increased slightly to account for volatility spikes
        "min_spread": 1, # Minimum spread of 1 tick
        "max_spread": 3, # Max spread of 3 ticks
        "position_scale": 0.5, # How aggressively we reduce size based on position
        "vol_window": 15, # Window for volatility estimate
        "order_skew_threshold": 0.2, # Sensitivity to order book imbalance
        "base_order_size": 20,
        "history_maxlen": 100 # Increased history length for better autocorrelation analysis
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.10, # Adjusted based on analysis for stability
        "clear_width": 0,
        "spread_multiplier": 1.5, # Increased slightly
        "min_spread": 1, # Minimum spread of 1 tick
        "max_spread": 3, # Max spread of 3 ticks
        "position_scale": 0.5, # How aggressively we reduce size based on position
        "vol_window": 15, # Window for volatility estimate
        "order_skew_threshold": 0.2, # Sensitivity to order book imbalance
        "base_order_size": 20,
        "history_maxlen": 100 # Increased history length
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
    
    
    # ... (order_flow_imbalance, previous_midprice, full_book_weighted_mid_price functions remain the same) ...
    
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
        Analyze price movements to detect negative autocorrelation and adjust market making strategy
        Uses deque for price history.
        Returns:
            - correlation_strength: measure of negative autocorrelation [-1 to 0]
            - bid_adjustment: suggested adjustment to bid price based on recent movements
            - ask_adjustment: suggested adjustment to ask price based on recent movements
        """
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
            
        if product not in traderObject["price_history"]:
            max_len = self.params[product].get("history_maxlen", 100)
            traderObject["price_history"][product] = deque(maxlen=max_len)
            return 0.0, 0.0, 0.0
            
        price_history = traderObject["price_history"][product]
        
        # Append current price
        price_history.append(current_price)
        
        # Need at least 2 prices (1 return) for meaningful autocorrelation analysis
        if len(price_history) < 2:
            return 0.0, 0.0, 0.0
            
        # Convert to numpy array for efficient calculations
        prices = np.array(list(price_history))
        
        # Calculate returns
        returns = np.diff(prices)
        
        # Need at least 2 returns for lag-1 autocorrelation
        if len(returns) < 2:
            return 0.0, 0.0, 0.0
            
        # Calculate lag-1 autocorrelation using numpy
        n = len(returns)
        mean_return = np.mean(returns)
        
        # Handle case of zero variance to avoid division by zero
        if np.var(returns) == 0:
             autocorr = 0.0
        else:
             # Calculate autocorrelation
             numerator = np.sum((returns[1:] - mean_return) * (returns[:-1] - mean_return))
             denominator = np.sum((returns - mean_return) ** 2)
             
             if denominator == 0:
                 autocorr = 0.0
             else:
                 autocorr = numerator / denominator
        
        # Limit to negative correlations only (for mean reversion strategy)
        correlation_strength = min(autocorr, 0.0)
        
        # Recent price movement direction
        recent_movement = returns[-1] if returns.size > 0 else 0.0
        
        # Calculate appropriate adjustments based on negative autocorrelation
        # Apply a smaller adjustment factor based on the strength of negative correlation
        adjustment_factor = abs(correlation_strength) * 0.5  # Adjust this factor
        
        bid_adjustment = 0.0
        ask_adjustment = 0.0
        
        # If prices just went up (positive recent movement), negative autocorrelation suggests they'll likely go down next.
        # We should adjust our ask price slightly lower to capture potential sellers before the drop.
        if recent_movement > 0:
            ask_adjustment = -recent_movement * adjustment_factor
        # If prices just went down (negative recent movement), negative autocorrelation suggests they'll likely go up next.
        # We should adjust our bid price slightly higher to capture potential buyers before the rise.
        elif recent_movement < 0:
            bid_adjustment = -recent_movement * adjustment_factor  # Negate recent_movement as it's negative
            
        return correlation_strength, bid_adjustment, ask_adjustment

    def calculate_tick_size(self, product_history):
        """
        Estimate the effective tick size from recent price changes.
        """
        if not product_history or len(product_history) < 2:
            return 1.0  # Default to 1 if not enough data

        prices = np.array(list(product_history))
        diffs = np.diff(prices)

        # Consider non-zero price changes
        non_zero_diffs = np.abs(diffs[diffs != 0])
        
        if len(non_zero_diffs) == 0:
            return 1.0 # Default if all changes are zero

        # Find the GCD of the absolute non-zero differences
        # This is a heuristic for tick size
        # Use np.gcd.reduce for multiple numbers
        try:
            # Multiply by a factor to handle potential floating point inaccuracies
            # And find GCD of integers.
            factor = 100 # Assume prices are multiples of 0.01
            scaled_diffs = np.round(non_zero_diffs * factor).astype(int)
            
            if scaled_diffs.size == 0:
                 return 1.0

            # Remove zeros and handle potential negative values if any
            unique_scaled_diffs = np.unique(scaled_diffs[scaled_diffs != 0])

            if unique_scaled_diffs.size == 0:
                return 1.0

            # Compute GCD of the absolute values
            tick_gcd_scaled = np.gcd.reduce(np.abs(unique_scaled_diffs))

            # Convert back to original scale
            if tick_gcd_scaled > 0:
                 return tick_gcd_scaled / factor
            else:
                 return 1.0 / factor # Handle case where GCD is 0 (e.g., all diffs were 0)
                 
        except Exception as e:
             logger.print(f"Error calculating tick size: {e}")
             return 1.0


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
        fair_value: float, # Use the EWMA fair value as base
        position: int,
        position_limit: int,
        traderObject: dict,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):
        """
        Advanced market making strategy for SQUID_INK incorporating autocorrelation
        and refined volatility/spread calculations.
        """
        
        orders = []
        
        # Calculate current mid price from best orders
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        if best_bid is None or best_ask is None:
             return [], 0, 0
        current_mid = (best_bid + best_ask) / 2

        # Use a more dynamic fair value based on book imbalance and autocorrelation
        optimal_mid = self.get_optimal_mid_price(order_depth, current_mid, position, position_limit, product, traderObject)
        
        # Calculate optimal spread based on volatility
        optimal_spread = self.calculate_optimal_spread(product, traderObject, best_ask - best_bid)
        
        # Calculate skewed bid/ask prices based on optimal_mid and optimal_spread
        bid_price = math.floor(optimal_mid - optimal_spread / 2)
        ask_price = math.ceil(optimal_mid + optimal_spread / 2)
        
        # Ensure integer prices for orders
        bid_price = int(bid_price)
        ask_price = int(ask_price)

        # Ensure minimum spread
        if ask_price - bid_price < self.params[product]["min_spread"]:
             mid = (bid_price + ask_price) // 2
             bid_price = mid - self.params[product]["min_spread"] // 2
             ask_price = mid + self.params[product]["min_spread"] // 2
        
        # Calculate order sizes based on position and position limit
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Base order size
        base_size = self.params[product]["base_order_size"]
        
        # Adjust order sizes based on inventory pressure
        # Scale order size down as position approaches limit
        buy_size = int(base_size * (1 - max(0, position / position_limit) * self.params[product]["position_scale"]))
        sell_size = int(base_size * (1 - max(0, -position / position_limit) * self.params[product]["position_scale"]))
        
        # Ensure minimum sizes
        buy_size = max(1, buy_size)
        sell_size = max(1, sell_size)

        # Adjust based on remaining capacity
        buy_size = min(buy_size, max_buy_capacity)
        sell_size = min(sell_size, max_sell_capacity)
        
        # Place orders
        if buy_size > 0 and bid_price < best_bid: # Only place if we improve the bid
             orders.append(Order(product, bid_price, buy_size))
             
        if sell_size > 0 and ask_price > best_ask: # Only place if we improve the ask
             orders.append(Order(product, ask_price, -sell_size))
             
        # Alternative: Place orders at the calculated prices, even if not improving the book
        # This depends on market microstructure and strategy goals
        # if buy_size > 0: orders.append(Order(product, bid_price, buy_size))
        # if sell_size > 0: orders.append(Order(product, ask_price, -sell_size))

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
        Advanced market making strategy for KELP incorporating autocorrelation
        and refined volatility/spread calculations.
        """
        
        orders = []
        
        # Calculate current mid price from best orders
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        if best_bid is None or best_ask is None:
             return [], 0, 0
        current_mid = (best_bid + best_ask) / 2

        # Use a more dynamic fair value based on book imbalance and autocorrelation
        optimal_mid = self.get_optimal_mid_price(order_depth, current_mid, position, position_limit, product, traderObject)
        
        # Calculate optimal spread based on volatility
        optimal_spread = self.calculate_optimal_spread(product, traderObject, best_ask - best_bid)
        
        # Calculate skewed bid/ask prices based on optimal_mid and optimal_spread
        bid_price = math.floor(optimal_mid - optimal_spread / 2)
        ask_price = math.ceil(optimal_mid + optimal_spread / 2)
        
        # Ensure integer prices for orders
        bid_price = int(bid_price)
        ask_price = int(ask_price)

        # Ensure minimum spread
        if ask_price - bid_price < self.params[product]["min_spread"]:
             mid = (bid_price + ask_price) // 2
             bid_price = mid - self.params[product]["min_spread"] // 2
             ask_price = mid + self.params[product]["min_spread"] // 2
        
        # Calculate order sizes based on position and position limit
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
        # Base order size
        base_size = self.params[product]["base_order_size"]
        
        # Adjust order sizes based on inventory pressure
        # Scale order size down as position approaches limit
        buy_size = int(base_size * (1 - max(0, position / position_limit) * self.params[product]["position_scale"]))
        sell_size = int(base_size * (1 - max(0, -position / position_limit) * self.params[product]["position_scale"]))
        
        # Ensure minimum sizes
        buy_size = max(1, buy_size)
        sell_size = max(1, sell_size)

        # Adjust based on remaining capacity
        buy_size = min(buy_size, max_buy_capacity)
        sell_size = min(sell_size, max_sell_capacity)
        
        # Place orders
        if buy_size > 0 and bid_price < best_bid: # Only place if we improve the bid
             orders.append(Order(product, bid_price, buy_size))
             
        if sell_size > 0 and ask_price > best_ask: # Only place if we improve the ask
             orders.append(Order(product, ask_price, -sell_size))
             
        return orders, buy_size, sell_size
    
    
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