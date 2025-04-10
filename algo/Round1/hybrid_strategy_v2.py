from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import json
import jsonpickle
import numpy as np
from typing import Any, List
from collections import deque  # For efficient sliding window operations

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
        "clear_width": 0,
        "signal_threshold": 0  # Lower threshold for RESIN since strategy differences are smaller
    },
    Product.KELP: {
        "fair_value": 2000,
        "ewma_beta": 0.05, # found using grid search 0 - 1
        "clear_width": 0.1,
        "rolling_window": 25,
        "spread_multiplier": 1.1,
        "min_spread": 2,
        "max_spread": 5,
        "position_scale": 0.6,
        "vol_window": 15,
        "order_skew_threshold": 0.25,
        "signal_threshold": 0  # Medium threshold for KELP
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.15, # found using grid search 0 - 1
        "clear_width": 0.1,
        "rolling_window": 50,
        "spread_multiplier": 1.2,
        "min_spread": 2,
        "max_spread": 6,
        "position_scale": 0.85,
        "vol_window": 30,
        "order_skew_threshold": 0.4,
        "signal_threshold": 0  # Higher threshold for SQUID_INK
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
        Helper function used to find the order flow imbalance
        """
        if not traderObject.get("prev_quote", {}).get(product):
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
    
    def calculate_book_imbalance(self, order_depth):
        """
        Calculate order book imbalance to determine market direction pressure
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
        beta: float  # smoothing factor
        ) -> float:
        
        """
        Calculates exponentially weighted volatility of price levels.
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
        
    def evaluate_signal_strength(
            self,
            product: str,
            order_depth: OrderDepth,
            traderObject: dict
            ) -> float:
        """
        Evaluates the strength of market signals to determine which strategy to use.
        Returns a normalized value between 0 and 1 where higher values indicate stronger signals.
        """
        # Calculate book imbalance as one signal component
        imbalance = self.calculate_book_imbalance(order_depth)
        imbalance_strength = abs(imbalance)
        
        # Calculate order flow imbalance as another signal component
        flow_imbalance = self.order_flow_imbalance(product, order_depth, traderObject)
        
        # Normalize flow_imbalance (typically in range of volume)
        max_flow = 50  # typical max position
        flow_strength = min(1.0, abs(flow_imbalance) / max_flow)
        
        # Calculate price volatility as another signal component
        volatility = 0
        if "vol" in traderObject and product in traderObject["vol"]:
            vol_data = traderObject["vol"][product]
            if "ewma_var" in vol_data:
                volatility = vol_data["ewma_var"] ** 0.5  # Get standard deviation
                
        # Normalize volatility (typically in range of 1-10 price units)
        vol_strength = min(1.0, volatility / 10.0)
        
        # Combine signals with weights
        # Imbalance gets highest weight as it's most reliable for short-term direction
        # Volatility gets medium weight as it indicates potential trading opportunities
        # Flow gets lowest weight as it can be noisy
        signal_strength = (0.45 * imbalance_strength) + (0.25 * vol_strength) + (0.3 * flow_strength)
        
        return signal_strength

    #########################################
    ### V5 Strategy Methods (Simple) #######
    #########################################
        
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
                    
        return orders, buy_quantity, sell_quantity
    
    def resin_make_orders_v5(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        position_limit: int,
        default_edge: float,
        default_order_size: int,
        take_buy_quantity: int,
        take_sell_quantity: int
        ) -> (List[Order], int, int):

        """
        Market making for RAINFOREST RESIN from v5
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
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position + position_limit - take_sell_quantity
    
        # Place bid order if does not exceed capacity
        if max_buy_capacity > 0:
            buy_quantity = min(default_order_size, max_buy_capacity)
            orders.append(Order(product, bid_price, buy_quantity))
    
        # Place ask order if does not exceed capacity
        if max_sell_capacity > 0:
            sell_quantity = min(default_order_size, max_sell_capacity)
            orders.append(Order(product, ask_price, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    def clear_orders_v5(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            clear_width: int,
            take_buy_quantity: int,
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
    
    def kelp_make_orders_v5(
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
        max_sell_capacity = position_limit + position - take_sell_quantity
    
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
    
    def ink_make_orders_v5(self,
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
        
        if mm_bid is None or mm_ask is None:
            return orders, buy_quantity, sell_quantity
            
        bid_price = mm_bid + penny_amount
        ask_price = mm_ask - penny_amount
        
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity
        
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

    #########################################
    ### V15 Strategy Methods (Advanced) ####
    #########################################
    
    def calculate_optimal_spread(self, product, traderObject, current_spread):
        """
        Calculate the optimal spread based on recent price volatility
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
    
    def kelp_market_making_v15(self,
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
            adjusted_fair_value += imbalance * optimal_spread / 3
            
        # Calculate position scaling factor (reduce size as position grows)
        position_ratio = position / position_limit if position_limit != 0 else 0
        position_scale = 1.0 - abs(position_ratio) * self.params[product]["position_scale"]
        
        # Default order size, scaled by position
        default_order_size = 18
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
    
    def squid_ink_market_making_v15(self,
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
        
        """ Rainforest Resin - Strategy selection based on signal strength"""
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            # Save current quotes for next iteration
            resin_best = self.best_orders(Product.RAINFOREST_RESIN, resin_order_depth)
            traderObject["prev_quote"][Product.RAINFOREST_RESIN] = resin_best
            
            # Calculate volatility for signal strength evaluation
            resin_mm_bid, resin_mm_ask, _, _ = self.mm_orders(Product.RAINFOREST_RESIN, resin_order_depth)
            if resin_mm_bid is not None and resin_mm_ask is not None:
                resin_mm_mid = (resin_mm_bid + resin_mm_ask) / 2
                self.ewma_volatility(Product.RAINFOREST_RESIN, traderObject, resin_mm_mid, 0.94)
            
            # Calculate signal strength to determine which strategy to use
            signal_strength = self.evaluate_signal_strength(
                Product.RAINFOREST_RESIN, 
                resin_order_depth, 
                traderObject
            )
            
            # Both v5 and v15 use the same strategy for Resin, so we'll use v5 for simplicity
            resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.take_orders(
                Product.RAINFOREST_RESIN,
                resin_order_depth,
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                self.LIMIT[Product.RAINFOREST_RESIN]
            )
            
            # Market making orders
            required_edge = 4
            default_order_size = 15
            
            resin_make_orders, _, _ = self.resin_make_orders_v5(
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
            
            # Clearing orders
            resin_clear_orders, _, _ = self.clear_orders_v5(
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
        
        """ Kelp - Strategy selection based on signal strength """
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
                
                # Calculate volatility for signal strength evaluation
                self.ewma_volatility(Product.KELP, traderObject, kelp_mm_mid, 0.94)
            
                # Calculate EWMA for fair value
                kelp_fair_value = self.ewma(Product.KELP, traderObject, kelp_mm_mid, self.params[Product.KELP]["ewma_beta"])
                if kelp_fair_value:
                    self.params[Product.KELP]["fair_value"] = kelp_fair_value
                
                # Calculate signal strength to determine which strategy to use
                signal_strength = self.evaluate_signal_strength(
                    Product.KELP, 
                    kelp_order_depth, 
                    traderObject
                )
                
                # Log signal strength for monitoring
                logger.print(f"KELP signal strength: {signal_strength:.4f}, threshold: {self.params[Product.KELP]['signal_threshold']:.4f}")
                
                # Take orders (common to both strategies)
                kelp_take_orders, kelp_take_buy_quantity, kelp_take_sell_quantity = self.take_orders(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value,
                    kelp_position,
                    self.LIMIT[Product.KELP]
                )
                
                # Choose strategy based on signal strength
                if signal_strength >= self.params[Product.KELP]["signal_threshold"]:
                    # Use advanced strategy (v15) for strong signals
                    kelp_make_orders, kelp_buy, kelp_sell = self.kelp_market_making_v15(
                        Product.KELP,
                        kelp_order_depth,
                        kelp_fair_value,
                        kelp_position,
                        self.LIMIT[Product.KELP],
                        traderObject,
                        kelp_take_buy_quantity,
                        kelp_take_sell_quantity
                    )
                    logger.print(f"Used KELP v15 strategy (signal: {signal_strength:.4f})")
                else:
                    # Use simpler strategy (v5) for weak signals
                    kelp_make_orders, kelp_buy, kelp_sell = self.kelp_make_orders_v5(
                        Product.KELP,
                        kelp_order_depth,
                        kelp_fair_value,
                        kelp_position,
                        self.LIMIT[Product.KELP],
                        15,  # default order size
                        kelp_take_buy_quantity,
                        kelp_take_sell_quantity,
                        1,  # bias
                        state
                    )
                    logger.print(f"Used KELP v5 strategy (signal: {signal_strength:.4f})")
                
                # Clear orders (common to both strategies)
                kelp_clear_orders, _, _ = self.clear_orders_v5(
                    Product.KELP,
                    kelp_order_depth,
                    kelp_fair_value,
                    kelp_position,
                    self.LIMIT[Product.KELP],
                    self.params[Product.KELP]["clear_width"],
                    kelp_take_buy_quantity,
                    kelp_take_sell_quantity
                )
                
                result[Product.KELP] = (kelp_take_orders + kelp_make_orders + kelp_clear_orders)

        """ Squid Ink - Strategy selection based on signal strength """
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
                
                # Calculate volatility for signal strength evaluation
                self.ewma_volatility(Product.SQUID_INK, traderObject, ink_mm_mid, 0.94)
                        
                # Calculate EWMA for fair value
                ink_fair_value = self.ewma(Product.SQUID_INK, traderObject, ink_mm_mid, self.params[Product.SQUID_INK]["ewma_beta"])
                if ink_fair_value:
                    self.params[Product.SQUID_INK]["fair_value"] = ink_fair_value
                
                # Calculate signal strength to determine which strategy to use
                signal_strength = self.evaluate_signal_strength(
                    Product.SQUID_INK, 
                    ink_order_depth, 
                    traderObject
                )
                
                # Log signal strength for monitoring
                logger.print(f"SQUID_INK signal strength: {signal_strength:.4f}, threshold: {self.params[Product.SQUID_INK]['signal_threshold']:.4f}")
                
                # Take orders (common to both strategies)
                ink_take_orders, ink_take_buy_quantity, ink_take_sell_quantity = self.take_orders(
                    Product.SQUID_INK,
                    ink_order_depth,
                    ink_fair_value,
                    ink_position,
                    self.LIMIT[Product.SQUID_INK]
                )
                
                # Choose strategy based on signal strength
                if signal_strength >= self.params[Product.SQUID_INK]["signal_threshold"]:
                    # Use advanced strategy (v15) for strong signals
                    ink_make_orders, ink_buy, ink_sell = self.squid_ink_market_making_v15(
                        Product.SQUID_INK,
                        ink_order_depth,
                        ink_fair_value,
                        ink_position,
                        self.LIMIT[Product.SQUID_INK],
                        traderObject,
                        ink_take_buy_quantity,
                        ink_take_sell_quantity
                    )
                    logger.print(f"Used SQUID_INK v15 strategy (signal: {signal_strength:.4f})")
                else:
                    # Use simpler strategy (v5) for weak signals
                    ink_make_orders, ink_buy, ink_sell = self.ink_make_orders_v5(
                        Product.SQUID_INK,
                        ink_order_depth,
                        ink_fair_value,
                        ink_position,
                        self.LIMIT[Product.SQUID_INK],
                        15,  # default order size
                        ink_take_buy_quantity,
                        ink_take_sell_quantity,
                        1   # penny amount
                    )
                    logger.print(f"Used SQUID_INK v5 strategy (signal: {signal_strength:.4f})")
                
                # Clear orders (common to both strategies)
                ink_clear_orders, _, _ = self.clear_orders_v5(
                    Product.SQUID_INK,
                    ink_order_depth,
                    ink_fair_value,
                    ink_position,
                    self.LIMIT[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["clear_width"],
                    ink_take_buy_quantity,
                    ink_take_sell_quantity
                )
                
                result[Product.SQUID_INK] = (ink_take_orders + ink_make_orders + ink_clear_orders)
        
        """ 
        Tidying up
        """
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData