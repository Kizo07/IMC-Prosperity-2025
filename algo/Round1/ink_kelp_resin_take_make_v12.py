#from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState # Keep this line if running outside a simulated environment
import json
import jsonpickle
import numpy as np
from typing import Any, List, Dict # Corrected import for Dict
from collections import deque

"""
GRID SEARCH param. Leave this equal to None to initialize it.

Wherever you want to perform a grid search, replace that with = grid search

ex.

parameter_beta = grid_search

Run the gridsearch.py with this file name and follow guide there
"""
grid_search = None


"""
Boilerplate for backtester (Assume Logger class and ProsperityEncoder are defined as before)
"""
# ... (Logger class code remains unchanged) ...
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Ensure ProsperityEncoder is available or replace with standard json if needed
        try:
            encoder = ProsperityEncoder
        except NameError:
            import json # Fallback if ProsperityEncoder is not defined
            encoder = json.JSONEncoder

        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ], encoder
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3 if self.max_log_length > base_length else 100 # Avoid division by zero or negative

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ], encoder
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Assuming compress_listings etc. are defined correctly
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
            compressed[symbol] = [list(order_depth.buy_orders.items()), list(order_depth.sell_orders.items())] # Store as list of tuples
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
        if hasattr(observations, 'conversionObservations'): # Check if attribute exists
             for product, observation in observations.conversionObservations.items():
                conversion_observations[product] = [
                    # Ensure all keys exist or handle missing keys
                    getattr(observation, 'bidPrice', None),
                    getattr(observation, 'askPrice', None),
                    getattr(observation, 'transportFees', None),
                    getattr(observation, 'exportTariff', None),
                    getattr(observation, 'importTariff', None),
                    getattr(observation, 'sunlight', None), # Assuming 'sunlight' is the correct name
                    getattr(observation, 'humidity', None), # Assuming 'humidity' is the correct name
                ]
        plain_observations = getattr(observations, 'plainValueObservations', {})
        return [plain_observations, conversion_observations]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any, encoder) -> str:
         # Use the passed encoder
        return json.dumps(value, cls=encoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if not isinstance(value, str): # Ensure value is a string
            value = str(value)
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()


""" Strategy Code starts here """

# Use Symbol type if available, otherwise fallback to string
try:
    from datamodel import Symbol
except ImportError:
    Symbol = str

class Product:
    RAINFOREST_RESIN = "ORCHIDS" # Example change if needed
    KELP = "KELP"
    SQUID_INK = "SEASHELLS" # Example change if needed, use actual product names from the simulation


# <<< CHANGE: Added parameters for autocorrelation strategy >>>
PARAMS = {
     Product.RAINFOREST_RESIN: { # Assuming RAINFOREST_RESIN maps to ORCHIDS
        "fair_value": 10000, # Example fair value for ORCHIDS
        "clear_width": 1 # Example clear width
        # Add other necessary params for ORCHIDS if different logic is used
    },
    Product.KELP: {
        "fair_value": 2000, # Placeholder, will be updated by EWMA
        "ewma_beta": 0.15,
        "clear_width": 0, # Set to 0 if clearing logic not used for KELP MM
        "rolling_window": 100,
        "spread_multiplier": 0.9,
        "min_spread": 2,
        "max_spread": 4,
        "position_scale": 0.4,
        "vol_window": 20,
        "order_skew_threshold": 0.3,
        # Autocorrelation params
        "autocorr_window": 15,        # How many ticks to look back for autocorrelation
        "autocorr_threshold": -0.2,   # Min negative correlation to trigger adjustments
        "autocorr_adj_factor": 0.6,   # How much to adjust bid/ask based on last move & corr
        "autocorr_size_scale": 1.25,  # Factor to increase size when autocorr is strong
        "price_history_len": 100      # Max length for price history deque
    },
    Product.SQUID_INK: { # Assuming SQUID_INK maps to SEASHELLS
        "fair_value": 2000, # Placeholder, will be updated by EWMA
        "ewma_beta": 0.3,
        "clear_width": 0, # Set to 0 if clearing logic not used for SEASHELLS MM
        "rolling_window": 100,
        "spread_multiplier": 0.9,
        "min_spread": 2,
        "max_spread": 5,
        "position_scale": 0.5,
        "vol_window": 25,
        "order_skew_threshold": 0.35,
        "base_order_size": 12,
        # Autocorrelation params
        "autocorr_window": 20,        # How many ticks to look back for autocorrelation
        "autocorr_threshold": -0.25,  # Min negative correlation to trigger adjustments (adjust per product)
        "autocorr_adj_factor": 0.5,   # How much to adjust bid/ask based on last move & corr
        "autocorr_size_scale": 1.20,  # Factor to increase size when autocorr is strong
        "price_history_len": 100      # Max length for price history deque
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # <<< CHANGE: Use actual product names for limits >>>
        self.LIMIT = {Product.RAINFOREST_RESIN: 20, Product.KELP: 250, Product.SQUID_INK: 500} # Limits for ORCHIDS, KELP, SEASHELLS

    # ... (best_orders, mm_orders remain largely unchanged) ...
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
            best_ask_amount = -1 * order_depth.sell_orders[best_ask] # Keep negative convention

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
            mm_ask_amount = -1 * order_depth.sell_orders[mm_ask] # Keep negative convention

        return mm_bid, mm_ask, mm_bid_amount, mm_ask_amount


    # <<< CHANGE: Standardize price history tracking >>>
    def update_price_history(self, product: str, price: float, traderObject: dict):
        """Stores the latest price in a deque for the product."""
        if price is None:
            return

        history_len = self.params[product].get("price_history_len", 100) # Get from params

        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
        if product not in traderObject["price_history"]:
            traderObject["price_history"][product] = deque(maxlen=history_len)

        traderObject["price_history"][product].append(price)

    # ... (rolling_mm_mid_quotes can be removed if ewma uses price_history directly) ...
    # ... (order_flow_imbalance, previous_midprice, full_book_weighted_mid_price remain unchanged) ...
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
        if "prev_quote" not in traderObject or product not in traderObject["prev_quote"] or not traderObject["prev_quote"][product]:
             # Handle missing key or empty list gracefully
            return 0

        B_n, A_n, q_B_n, q_A_n = self.best_orders(product, order_depth)

        # Ensure current best bid/ask are available
        if B_n is None or A_n is None or q_B_n is None or q_A_n is None:
             return 0

        # Safely access previous quote elements
        prev_quote = traderObject["prev_quote"][product]
        if len(prev_quote) < 4: # Ensure all elements are present
             return 0
        B_n_minus_1, A_n_minus_1, q_B_n_minus_1, q_A_n_minus_1 = prev_quote

        # Ensure previous best bid/ask were recorded
        if B_n_minus_1 is None or A_n_minus_1 is None or q_B_n_minus_1 is None or q_A_n_minus_1 is None:
            return 0

        # Indicator logic needs adjustment based on definition (e.g., exact equality vs. change)
        # Assuming strict inequality for change detection:
        I_B_increase = int(B_n > B_n_minus_1)
        I_B_decrease = int(B_n < B_n_minus_1)
        I_A_increase = int(A_n > A_n_minus_1)
        I_A_decrease = int(A_n < A_n_minus_1)

        # Ensure q values are numbers (handle potential None or non-numeric types)
        q_B_n = q_B_n if isinstance(q_B_n, (int, float)) else 0
        q_B_n_minus_1 = q_B_n_minus_1 if isinstance(q_B_n_minus_1, (int, float)) else 0
        q_A_n = abs(q_A_n) if isinstance(q_A_n, (int, float)) else 0 # Use absolute volume
        q_A_n_minus_1 = abs(q_A_n_minus_1) if isinstance(q_A_n_minus_1, (int, float)) else 0 # Use absolute volume

        # Calculate order flow imbalance using the formula
        e_n = (I_B_increase * q_B_n
               - I_B_decrease * q_B_n_minus_1
               - I_A_decrease * q_A_n # Note: Original had subtraction, check formula source
               + I_A_increase * q_A_n_minus_1) # Note: Original had addition, check formula source

        # Example common OFI: (delta_bid_vol if price_up) - (delta_ask_vol if price_down)
        # This requires tracking volume changes at best levels, not just the levels themselves.
        # The provided formula seems different, double-check its definition and intent.
        # Let's return 0 for now until the formula is clarified or a standard OFI is implemented.
        # return e_n
        return 0 # Placeholder - requires clarification of the formula's intent

    def previous_midprice(
            self,
            product:str,
            traderObject: dict
            ) -> float:
        """
        Helper function used to calculate the previous tick's mid price
        """

        # Check if 'prev_quote' and the specific product key exist
        if "prev_quote" not in traderObject or product not in traderObject["prev_quote"]:
            return None

        prev_data = traderObject["prev_quote"][product]

        # Check if the list has at least 2 elements (bid and ask)
        if not prev_data or len(prev_data) < 2:
             return None

        # Safely access bid and ask
        prev_bid = prev_data[0]
        prev_ask = prev_data[1]

        # Check if both bid and ask are valid numbers
        if isinstance(prev_bid, (int, float)) and isinstance(prev_ask, (int, float)):
            return (prev_bid + prev_ask) / 2
        else:
            return None # Return None if either bid or ask is invalid


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

        # Handle potential empty dictionaries before creating arrays
        buy_prices_list = list(order_depth.buy_orders.keys())
        buy_volumes_list = list(order_depth.buy_orders.values())
        sell_prices_list = list(order_depth.sell_orders.keys())
        sell_volumes_list = list(order_depth.sell_orders.values())

        # Proceed only if there are orders on either side
        if not buy_prices_list and not sell_prices_list:
             return None

        # Convert to numpy arrays for faster computation
        buy_prices = np.array(buy_prices_list) if buy_prices_list else np.array([])
        buy_volumes = np.array(buy_volumes_list) if buy_volumes_list else np.array([])
        sell_prices = np.array(sell_prices_list) if sell_prices_list else np.array([])
        # Use absolute volume for sells
        sell_volumes = np.abs(np.array(sell_volumes_list)) if sell_volumes_list else np.array([])

        # Ensure arrays are not empty before calculation
        weighted_sum = 0
        total_volume = 0
        if buy_prices.size > 0:
             weighted_sum += np.sum(buy_prices * buy_volumes)
             total_volume += np.sum(buy_volumes)
        if sell_prices.size > 0:
             weighted_sum += np.sum(sell_prices * sell_volumes)
             total_volume += np.sum(sell_volumes)


        if total_volume == 0:
            return None # Avoid division by zero

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
             # Try to return the last known EWMA if current mid is None
            return traderObject.get("ewma", {}).get(product)


        prev_ewma = traderObject.get("ewma", {}).get(product)

        if prev_ewma is None:
            result = mid # Initialize with the first valid mid-price
        else:
            result = (1 - beta) * mid + beta * prev_ewma

        # Ensure 'ewma' dictionary exists before assignment
        if "ewma" not in traderObject:
            traderObject["ewma"] = {}
        traderObject["ewma"][product] = result

        return result


    # ... (ewma_volatility can be removed if calculate_optimal_spread handles it) ...
    # ... (ink_fair_value seems unused, can remove) ...
    # ... (take_orders, resin_make_orders, clear_orders, make_orders remain unchanged) ...
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

        # Ensure fair_value is a number before comparison
        if fair_value is None:
             return orders, buy_quantity, sell_quantity

        best_bid, best_ask, best_bid_quantity, best_ask_quantity = self.best_orders(product, order_depth)

        # Check if best_bid exists and is higher than fair_value
        if best_bid is not None and best_bid_quantity is not None and best_bid > fair_value:
            # Calculate volume to sell: minimum of available volume at best bid and our selling capacity
            # Selling capacity = current position + position limit (max we can be short)
            sell_volume = min(abs(best_bid_quantity), position + position_limit) # best_bid_quantity is positive
            if sell_volume > 0:
                orders.append(Order(product, best_bid, -sell_volume)) # Sell order has negative quantity
                sell_quantity = sell_volume

        # Check if best_ask exists and is lower than fair_value
        if best_ask is not None and best_ask_quantity is not None and best_ask < fair_value:
             # Calculate volume to buy: minimum of available volume at best ask and our buying capacity
             # Buying capacity = position limit - current position (max we can be long)
            buy_volume = min(abs(best_ask_quantity), position_limit - position) # best_ask_quantity is negative
            if buy_volume > 0:
                orders.append(Order(product, best_ask, buy_volume)) # Buy order has positive quantity
                buy_quantity = buy_volume

        return orders, buy_quantity, sell_quantity # Return actual quantities traded


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
        Primary function used to market make for RAINFOREST RESIN (ORCHIDS)
        """
        if fair_value is None:
            return [], 0, 0

        orders: List[Order] = []

        # Calculate bid and ask prices (ensure they are integers for orders)
        bid_price = int(round(fair_value - default_edge))
        ask_price = int(round(fair_value + default_edge))

        make_buy_quantity = 0 # Quantity we intend to place as a buy limit order
        make_sell_quantity = 0 # Quantity we intend to place as a sell limit order

        # Check position limits, considering orders already placed by 'take_orders'
        # Max we can buy = limit - current position - pending buys from take_orders
        max_buy_capacity = position_limit - position - take_buy_quantity
        # Max we can sell = limit + current position - pending sells from take_orders
        max_sell_capacity = position_limit + position - take_sell_quantity # position is signed

        # Place bid order if we have capacity
        if max_buy_capacity > 0:
            make_buy_quantity = min(default_order_size, max_buy_capacity)
            if make_buy_quantity > 0: # Ensure we place a positive quantity
                 orders.append(Order(product, bid_price, make_buy_quantity))

        # Place ask order if we have capacity
        if max_sell_capacity > 0:
            make_sell_quantity = min(default_order_size, max_sell_capacity)
            if make_sell_quantity > 0: # Ensure we place a positive quantity
                 orders.append(Order(product, ask_price, -make_sell_quantity)) # Sell order quantity is negative

        # Return the orders placed and the quantities intended for these making orders
        return orders, make_buy_quantity, make_sell_quantity


    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: int,
            position: int,
            position_limit: int,
            clear_width: int,
            take_buy_quantity: int, # existing buy orders from market taking
            take_sell_quantity: int # existing sell orders from market taking
            ) -> (List[Order], int, int):

        if fair_value is None:
            return [], take_buy_quantity, take_sell_quantity

        orders: List[Order] = []

        # Effective position including pending take orders
        effective_position = position + take_buy_quantity - take_sell_quantity

        # Thresholds for clearing trades (relative to fair value)
        ask_clear_threshold = fair_value + clear_width # Price >= this to clear long position (sell high)
        bid_clear_threshold = fair_value - clear_width # Price <= this to clear short position (buy low)

        # Remaining capacity considering pending take orders
        # How much more we *can* buy: limit - (current pos + pending buys)
        remaining_buy_capacity = position_limit - (position + take_buy_quantity)
        # How much more we *can* sell: limit + (current pos - pending sells)
        remaining_sell_capacity = position_limit + (position - take_sell_quantity) # position is signed


        # --- Try to clear LONG position (effective_position > 0) ---
        if effective_position > 0 and remaining_sell_capacity > 0:
            # Look for buyers willing to pay >= ask_clear_threshold
            # Iterate through buy orders (highest price first)
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for price, volume in sorted_bids:
                if price >= ask_clear_threshold:
                    # Quantity to sell: min(available volume at this price, our remaining long position, our remaining sell capacity)
                    qty_to_sell = min(abs(volume), effective_position, remaining_sell_capacity)
                    if qty_to_sell > 0:
                        orders.append(Order(product, price, -qty_to_sell)) # Sell order
                        effective_position -= qty_to_sell # Reduce effective long position
                        take_sell_quantity += qty_to_sell # Account for this clearing sell order
                        remaining_sell_capacity -= qty_to_sell # Reduce remaining sell capacity
                        if effective_position <= 0 or remaining_sell_capacity <= 0:
                            break # Stop if position cleared or capacity reached
                else:
                     # Bids are sorted high to low, if this one is too low, the rest will be too
                     break

        # --- Try to clear SHORT position (effective_position < 0) ---
        if effective_position < 0 and remaining_buy_capacity > 0:
            # Look for sellers willing to sell <= bid_clear_threshold
            # Iterate through sell orders (lowest price first)
            sorted_asks = sorted(order_depth.sell_orders.items())
            for price, volume in sorted_asks:
                if price <= bid_clear_threshold:
                    # Quantity to buy: min(available volume at this price, abs(our remaining short position), our remaining buy capacity)
                    qty_to_buy = min(abs(volume), abs(effective_position), remaining_buy_capacity)
                    if qty_to_buy > 0:
                        orders.append(Order(product, price, qty_to_buy)) # Buy order
                        effective_position += qty_to_buy # Reduce effective short position (moves towards 0)
                        take_buy_quantity += qty_to_buy # Account for this clearing buy order
                        remaining_buy_capacity -= qty_to_buy # Reduce remaining buy capacity
                        if effective_position >= 0 or remaining_buy_capacity <= 0:
                            break # Stop if position cleared or capacity reached
                else:
                    # Asks are sorted low to high, if this one is too high, the rest will be too
                    break

        # Return clearing orders and updated pending quantities
        return orders, take_buy_quantity, take_sell_quantity


    def make_orders(self,
        product: str,
        order_depth: OrderDepth,  # Product-specific OrderDepth
        fair_value: int,
        position: int,
        position_limit: int,
        default_edge: float,  # The distance from fair value for bids/asks
        default_order_size: float, # Can be float, will be rounded later
        disregard_edge: float, # Threshold to ignore far prices
        join_edge: float,      # Threshold to join existing orders
        take_buy_quantity: int, # existing buy orders from market taking
        take_sell_quantity: int # existing sell orders from market taking
        ) -> (List[Order], int, int):

        if fair_value is None:
            return [], 0, 0

        orders: List[Order] = []
        make_buy_quantity = 0
        make_sell_quantity = 0

        # --- Determine Target Ask Price ---
        # Find asks significantly above fair value
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge # Exclude prices too close or below fair
        ]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None

        # Default ask: fair + edge
        target_ask_price = round(fair_value + default_edge)

        # Pennying / Joining Logic for Ask
        if best_ask_above_fair is not None:
            # Calculate distance from fair value
            distance_from_fair = best_ask_above_fair - fair_value
            if distance_from_fair <= join_edge:
                # Join: Match the best existing ask if it's close enough
                target_ask_price = best_ask_above_fair
            else:
                # Penny: Place order 1 tick below the best existing ask
                target_ask_price = best_ask_above_fair - 1

        # --- Determine Target Bid Price ---
        # Find bids significantly below fair value
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge # Exclude prices too close or above fair
        ]
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        # Default bid: fair - edge
        target_bid_price = round(fair_value - default_edge)

        # Pennying / Joining Logic for Bid
        if best_bid_below_fair is not None:
             # Calculate distance from fair value
            distance_from_fair = fair_value - best_bid_below_fair
            if distance_from_fair <= join_edge:
                # Join: Match the best existing bid if it's close enough
                target_bid_price = best_bid_below_fair
            else:
                # Penny: Place order 1 tick above the best existing bid
                target_bid_price = best_bid_below_fair + 1


        # --- Calculate Order Sizes based on Capacity ---
        # Ensure default order size is an integer for min/max operations
        int_default_order_size = int(round(default_order_size))

        # Max we can buy = limit - current position - pending buys
        max_buy_capacity = position_limit - position - take_buy_quantity
        # Max we can sell = limit + current position - pending sells
        max_sell_capacity = position_limit + position - take_sell_quantity # position is signed

        # --- Place Orders ---
        # Place bid order if capacity allows
        if max_buy_capacity > 0:
            make_buy_quantity = min(int_default_order_size, max_buy_capacity)
            if make_buy_quantity > 0:
                 orders.append(Order(product, target_bid_price, make_buy_quantity))

        # Place ask order if capacity allows
        if max_sell_capacity > 0:
            make_sell_quantity = min(int_default_order_size, max_sell_capacity)
            if make_sell_quantity > 0:
                 orders.append(Order(product, target_ask_price, -make_sell_quantity)) # Negative quantity

        return orders, make_buy_quantity, make_sell_quantity


    def calculate_optimal_spread(self, product, traderObject, current_spread):
        """
        Calculate the optimal spread based on recent price volatility
        Using numpy for faster calculations
        Relies on 'price_history' being updated.
        """
        # <<< CHANGE: Use standardized price_history >>>
        price_history_deque = traderObject.get("price_history", {}).get(product)

        # Need at least vol_window + 1 prices to calculate volatility over the window
        min_hist_len = self.params[product].get("vol_window", 20) + 1

        if price_history_deque is None or len(price_history_deque) < min_hist_len:
            # Fallback to min spread if not enough history
            return self.params[product]["min_spread"]

        # Convert the relevant part of history to numpy array
        prices = np.array(list(price_history_deque)[-min_hist_len:])
        price_changes = np.diff(prices) # Calculate differences between consecutive prices

        # Calculate volatility as the standard deviation of price *changes* over the window
        vol_window = self.params[product]["vol_window"]
        # Use the last 'vol_window' changes
        recent_changes = price_changes[-vol_window:]

        if len(recent_changes) == 0:
             return self.params[product]["min_spread"] # Should not happen if min_hist_len check passed

        volatility = np.std(recent_changes)

        # Scale the spread based on volatility
        # Ensure integer spread, usually market makers quote half-spread each side
        optimal_spread = int(round(volatility * self.params[product]["spread_multiplier"]))

        # Clamp the spread within min/max bounds
        optimal_spread = max(
            self.params[product]["min_spread"],
            min(self.params[product]["max_spread"], optimal_spread)
        )

        # Ensure spread is at least 1 tick (if min_spread allows 0 or negative)
        optimal_spread = max(1, optimal_spread)

        return optimal_spread

    def calculate_book_imbalance(self, order_depth):
        """
        Calculate order book imbalance (volume-based)
        Using numpy for faster calculations
        """
        # Check if buy or sell orders exist
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return 0

        # Extract volumes using numpy for efficiency
        buy_volumes = np.array(list(order_depth.buy_orders.values())) if order_depth.buy_orders else np.array([0])
        # Sell orders have negative quantities in OrderDepth, take absolute value
        sell_volumes = np.abs(np.array(list(order_depth.sell_orders.values()))) if order_depth.sell_orders else np.array([0])

        # Sum the volumes
        total_buy_volume = np.sum(buy_volumes)
        total_sell_volume = np.sum(sell_volumes)

        # Calculate total volume across both sides
        total_volume = total_buy_volume + total_sell_volume

        # Avoid division by zero if book is empty (though checked earlier) or has zero volume
        if total_volume == 0:
            return 0

        # Calculate imbalance: (Buy Volume - Sell Volume) / Total Volume
        imbalance = (total_buy_volume - total_sell_volume) / total_volume
        return imbalance  # Range: [-1 (all sell), +1 (all buy)]


    # <<< CHANGE: New function for autocorrelation analysis >>>
    def analyze_autocorrelation(self, product: str, traderObject: dict) -> tuple[float, float]:
        """
        Calculates lag-1 autocorrelation of price changes (returns).
        Returns:
            - autocorr: The calculated lag-1 autocorrelation coefficient.
            - last_return: The most recent price change.
        """
        price_history_deque = traderObject.get("price_history", {}).get(product)
        autocorr_window = self.params[product].get("autocorr_window", 15)

        # Need at least autocorr_window + 1 prices for autocorr_window returns
        if price_history_deque is None or len(price_history_deque) < autocorr_window + 1:
            return 0.0, 0.0 # Not enough data

        prices = np.array(list(price_history_deque)[-(autocorr_window + 1):])
        returns = np.diff(prices)

        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0, returns[-1] if len(returns) > 0 else 0.0 # Cannot calculate correlation or no std dev

        # Calculate lag-1 autocorrelation manually or using numpy
        mean_return = np.mean(returns)
        numerator = np.sum((returns[1:] - mean_return) * (returns[:-1] - mean_return))
        denominator = np.sum((returns - mean_return) ** 2)

        if denominator == 0:
             autocorr = 0.0
        else:
             autocorr = numerator / denominator

        last_return = returns[-1]

        return autocorr, last_return

    # <<< CHANGE: Heavily modified market making for KELP and SQUID >>>
    def advanced_market_making(self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        traderObject: dict,
        take_buy_quantity: int, # Should typically be 0 for pure MM
        take_sell_quantity: int # Should typically be 0 for pure MM
        ) -> (List[Order], int, int):
        """
        Advanced market making strategy incorporating volatility, imbalance, and autocorrelation.
        Used for KELP and SQUID_INK.
        """
        if fair_value is None:
            return [], 0, 0

        orders = []
        params = self.params[product] # Get product-specific parameters

        # 1. Update Price History (done in main loop now)
        # self.update_price_history(product, fair_value, traderObject) # Assuming fair_value is a good proxy for price

        # 2. Calculate Optimal Spread based on Volatility
        best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
        current_spread = best_ask - best_bid if best_bid is not None and best_ask is not None else params["min_spread"] * 2 # Estimate if no quotes
        optimal_spread = self.calculate_optimal_spread(product, traderObject, current_spread)
        half_spread = max(1, optimal_spread // 2) # Ensure at least 1 tick half-spread


        # 3. Calculate Base Bid/Ask around Fair Value
        base_bid = fair_value - half_spread
        base_ask = fair_value + half_spread

        # 4. Analyze Autocorrelation and Adjust Prices
        autocorr, last_return = self.analyze_autocorrelation(product, traderObject)
        bid_adjustment = 0.0
        ask_adjustment = 0.0
        autocorr_signal_active = False

        if autocorr < params["autocorr_threshold"]: # Check if negative autocorrelation is strong enough
            autocorr_signal_active = True
            adj_magnitude = abs(last_return * params["autocorr_adj_factor"] * autocorr) # Scale adjustment by return size and corr strength
            adj_magnitude = min(adj_magnitude, half_spread * 0.5) # Cap adjustment to avoid crossing mid heavily

            if last_return > 0: # Price went UP, expect reversion DOWN
                ask_adjustment = -adj_magnitude # Lower ask aggressively
                # bid_adjustment = -adj_magnitude * 0.5 # Optionally lower bid slightly too
            elif last_return < 0: # Price went DOWN, expect reversion UP
                bid_adjustment = abs(adj_magnitude) # Raise bid aggressively
                # ask_adjustment = abs(adj_magnitude) * 0.5 # Optionally raise ask slightly too

        adjusted_bid = base_bid + bid_adjustment
        adjusted_ask = base_ask + ask_adjustment

        # 5. Adjust Prices based on Book Imbalance (applied after autocorrelation)
        imbalance = self.calculate_book_imbalance(order_depth)
        if abs(imbalance) > params["order_skew_threshold"]:
            # Skew both bid and ask slightly in direction of imbalance
            # Positive imbalance (more buys) -> push prices up
            # Negative imbalance (more sells) -> push prices down
            imbalance_adjustment = imbalance * half_spread * 0.3 # Smaller impact for imbalance skew
            adjusted_bid += imbalance_adjustment
            adjusted_ask += imbalance_adjustment

        # 6. Final Price Calculation (Rounding)
        final_bid_price = int(round(adjusted_bid))
        final_ask_price = int(round(adjusted_ask))

        # Ensure minimum spread after all adjustments
        if final_ask_price - final_bid_price < params["min_spread"]:
             mid = (final_bid_price + final_ask_price) / 2.0
             final_bid_price = int(mid - params["min_spread"] / 2.0)
             final_ask_price = int(mid + params["min_spread"] / 2.0)
             # Recalculate to ensure integer math works correctly for odd spreads
             if final_ask_price - final_bid_price < params["min_spread"]:
                 final_ask_price = final_bid_price + params["min_spread"]


        # 7. Calculate Order Sizes
        # Base size (can be product specific if needed, using a default here)
        base_order_size = params.get("base_order_size", 15) # Get from params or use default

        # Scale size based on position
        position_ratio = position / position_limit if position_limit != 0 else 0
        # Reduce size more aggressively as position limit is approached
        position_scale = max(0, 1.0 - (abs(position_ratio) ** 1.5) * params["position_scale"])


        # Scale size based on autocorrelation signal
        size_multiplier = 1.0
        if autocorr_signal_active:
            size_multiplier = params["autocorr_size_scale"]

        # Calculate adjusted size
        adjusted_order_size = max(1, int(round(base_order_size * position_scale * size_multiplier)))

        # Skew size based on position (place larger orders against current position)
        buy_order_size = adjusted_order_size
        sell_order_size = adjusted_order_size

        if position > 0: # Long position: want to sell more, buy less
            sell_order_size = int(round(adjusted_order_size * (1 + abs(position_ratio) * 0.5))) # Increase sell size
            buy_order_size = int(round(adjusted_order_size * (1 - abs(position_ratio) * 0.7))) # Decrease buy size
        elif position < 0: # Short position: want to buy more, sell less
            buy_order_size = int(round(adjusted_order_size * (1 + abs(position_ratio) * 0.5))) # Increase buy size
            sell_order_size = int(round(adjusted_order_size * (1 - abs(position_ratio) * 0.7))) # Decrease sell size

        # Ensure sizes are at least 1
        buy_order_size = max(1, buy_order_size)
        sell_order_size = max(1, sell_order_size)


        # 8. Check Capacity and Place Orders
        # Capacity calculation assumes take_buy/sell are 0 for pure MM
        max_buy_capacity = position_limit - position - take_buy_quantity
        max_sell_capacity = position_limit + position - take_sell_quantity # position is signed

        final_buy_size = min(buy_order_size, max_buy_capacity)
        final_sell_size = min(sell_order_size, max_sell_capacity)

        if final_buy_size > 0:
            orders.append(Order(product, final_bid_price, final_buy_size))

        if final_sell_size > 0:
            orders.append(Order(product, final_ask_price, -final_sell_size)) # Sell quantity is negative

        # Return orders and the sizes WE PLACED (not capacity)
        placed_buy_size = final_buy_size if final_buy_size > 0 else 0
        placed_sell_size = final_sell_size if final_sell_size > 0 else 0

        return orders, placed_buy_size, placed_sell_size


    # ... (mean_reversion_signal, analyze_price_autocorrelation removed as logic integrated) ...
    # ... (squid_ink_market_making, kelp_make_orders, ink_make_orders, kelp_market_making removed, replaced by advanced_market_making) ...

    """
    Main run function
    """
    def run(self, state: TradingState) -> tuple[Dict[Symbol, list[Order]], int, str]: # Use Dict
        result: Dict[Symbol, list[Order]] = {} # Use Dict
        conversions = 0
        traderObject = {} # Initialize empty traderObject

        # load any prev data
        if state.traderData:
            try:
                # Use jsonpickle for complex objects like deque
                traderObject = jsonpickle.decode(state.traderData)
                # Ensure nested dictionaries exist after decoding
                if "prev_quote" not in traderObject: traderObject["prev_quote"] = {}
                if "ewma" not in traderObject: traderObject["ewma"] = {}
                if "price_history" not in traderObject: traderObject["price_history"] = {}
            except Exception as e:
                print(f"Error decoding traderData: {e}")
                # Fallback to initializing a new object if decoding fails
                traderObject = {
                     "prev_quote": {}, "ewma": {}, "price_history": {}
                }
        else:
             # Initialize fresh if no traderData
            traderObject = {
                "prev_quote": {}, "ewma": {}, "price_history": {}
            }

        # Ensure keys exist for all products in nested dicts
        for product in self.params.keys():
            if product not in traderObject["prev_quote"]: traderObject["prev_quote"][product] = []
            if product not in traderObject["ewma"]: traderObject["ewma"][product] = None
            if product not in traderObject["price_history"]:
                history_len = self.params[product].get("price_history_len", 100)
                traderObject["price_history"][product] = deque(maxlen=history_len)


        # --- Process Each Product ---
        for product, params in self.params.items():
            if product in state.order_depths:
                position = state.position.get(product, 0)
                order_depth = state.order_depths[product]
                limit = self.LIMIT.get(product, 0) # Get limit for the product

                # Store current best bid/ask for potential use next tick (e.g., OFI)
                current_best = self.best_orders(product, order_depth)
                traderObject["prev_quote"][product] = current_best

                # Calculate mid-price (e.g., using mm orders or best orders)
                mm_bid, mm_ask, _, _ = self.mm_orders(product, order_depth)
                best_bid, best_ask, _, _ = self.best_orders(product, order_depth)

                # Choose a representative mid-price for EWMA and history
                mid_price = None
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2.0
                elif mm_bid is not None and mm_ask is not None:
                    mid_price = (mm_bid + mm_ask) / 2.0

                # Update EWMA fair value
                fair_value = self.ewma(product, traderObject, mid_price, params["ewma_beta"])

                # Update price history (use EWMA or mid_price)
                price_to_store = fair_value if fair_value is not None else mid_price
                self.update_price_history(product, price_to_store, traderObject)

                # Use the most recent fair_value, fall back to mid_price if EWMA is None
                current_fair_value = fair_value if fair_value is not None else mid_price

                # --- Apply Strategy based on Product ---
                product_orders = []
                if product == Product.RAINFOREST_RESIN:
                    # Use original take/make/clear logic for Resin (ORCHIDS)
                     if current_fair_value is not None: # Need a fair value for these strategies
                        # Taking orders
                        resin_take_orders, resin_take_buy, resin_take_sell = self.take_orders(
                            product, order_depth, int(round(current_fair_value)), position, limit)

                        # Clearing orders (use updated take quantities)
                        resin_clear_orders, resin_updated_take_buy, resin_updated_take_sell = self.clear_orders(
                             product, order_depth, int(round(current_fair_value)), position, limit,
                             params["clear_width"], resin_take_buy, resin_take_sell)

                        # Making orders (use updated take quantities from clearing)
                        resin_make_orders, _, _ = self.resin_make_orders(
                            product, order_depth, int(round(current_fair_value)), position, limit,
                            default_edge=4, default_order_size=15, # Example params for Resin
                            take_buy_quantity=resin_updated_take_buy,
                            take_sell_quantity=resin_updated_take_sell)

                        product_orders.extend(resin_take_orders)
                        product_orders.extend(resin_clear_orders)
                        product_orders.extend(resin_make_orders)

                elif product == Product.KELP or product == Product.SQUID_INK:
                    # Use the new advanced market making for Kelp and Squid Ink (SEASHELLS)
                     if current_fair_value is not None: # Need a fair value
                        mm_orders, _, _ = self.advanced_market_making(
                            product,
                            order_depth,
                            current_fair_value,
                            position,
                            limit,
                            traderObject,
                            0, # take_buy_quantity = 0 for pure MM
                            0  # take_sell_quantity = 0 for pure MM
                        )
                        product_orders.extend(mm_orders)

                # Add product orders to the result dictionary
                if product_orders:
                    result[product] = product_orders


        """
        Tidying up
        """
        # Use jsonpickle to encode traderObject with deque
        traderData = jsonpickle.encode(traderObject, unpicklable=False) # Use unpicklable=False for smaller string if re-loading isn't needed in the same exact state

        # Assuming ProsperityEncoder is defined globally or passed correctly
        # logger.flush(state, result, conversions, traderData) # Uncomment when Logger and Encoder are fully defined

        return result, conversions, traderData

# Example Usage (within simulation environment):
# trader = Trader()
# result, conversions, traderData = trader.run(state)