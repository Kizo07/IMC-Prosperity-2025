from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import json
import jsonpickle
from typing import Any, List
import math
import numpy as np



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
    PB1 = "PICNIC_BASKET1"
    PB2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    SPREAD3 = "SPREAD3"
    SPREAD4 = "SPREAD4"
    
    
PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "clear_width": 0
    },
    Product.KELP: {
        "fair_value": 2000,
        "ewma_beta": 0, # found using grid search 0 - 1
        "clear_width": 0
    },
    Product.SQUID_INK: {
        "fair_value": 2000,
        "ewma_beta": 0.95, # found using grid search 0 - 1
        "clear_width": 0
    },
    Product.CROISSANTS: {
        "fair_value": None
    },
    Product.JAMS: {
        "fair_value": None
    },
    Product.DJEMBES: {
        "fair_value": None
    },
    Product.PB1: {
        "fair_value": None
    },
    Product.PB2: {
        "fair_value": None
    },
    Product.SPREAD1: { 
        "default_spread_mean": 50, # from data exploration
        "default_spread_std": 9.8034198531218,
        "spread_std_window": 50,
        "zscore_threshold": 13,
        "base":  Product.PB1,
        "components" : {
            Product.CROISSANTS: 6,
            Product.JAMS: 3,
            Product.DJEMBES: 1,
        }
    },
    Product.SPREAD2: {
        "default_spread_mean": 20,
        "default_spread_std": 9.059942109103167,
        "spread_std_window": 50,
        "zscore_threshold": 13,
        "base": Product.PB1,
        "components" : {
            Product.PB2: 1,
            Product.CROISSANTS: 2,
            Product.JAMS: 1,
            Product.DJEMBES: 1
        }
    },
    Product.SPREAD3: {
        "default_spread_mean": 0,
        "default_spread_std": 12.073812106201045,
        "spread_std_window": 50,
        "zscore_threshold": 13,
        "base":  Product.PB1,
        "components" : {
            Product.PB2: 1.5,
            Product.DJEMBES: 1
        }
    },
    Product.SPREAD4: {
        "default_spread_mean": 30,
        "default_spread_std": 9.709557672931483,
        "spread_std_window": 50,
        "zscore_threshold": 13,
        "base": Product.PB2,
        "components" : {
            Product.CROISSANTS: 4,
            Product.JAMS: 2
        }
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
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50,
                      Product.PB1: 60, Product.PB2: 100, Product.CROISSANTS : 250, 
                      Product.JAMS : 350, Product.DJEMBES : 60}
        
    
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
    
    
    def ewma(
        self,
        product: str,
        traderObject: dict,
        value: float,
        beta: float # exp weight parameter
        ) -> float:
        
        """
        Helper function used to calculate an exponentially weighted moving average
        """  
        
        prev_ewma = traderObject.get("ewma", {}).get(product)

        if prev_ewma is None:
            result = value
        else:
            result = (1 - beta) * value + beta * prev_ewma
            
        traderObject["ewma"][product] = result
    
        return result
        
    
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
    
    
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        position_limit: int,
        clear_width: int,
        take_buy_quantity: int,  # existing buy and sell orders we have placed due to market taking to prevent cancellation
        take_sell_quantity: int,
        desired_position: int = 0   # NEW: target position to move toward
        ) -> (List[Order], int, int):
    
        orders: List[Order] = []
        
        # Position after accounting for already-placed take orders
        effective_position = position + take_buy_quantity - take_sell_quantity
        
        # Fair value thresholds
        fair_for_bid = fair_value - clear_width
        fair_for_ask = fair_value + clear_width
    
        # Remaining quantities we are allowed to trade
        remaining_buy_qty = position_limit - (position + take_buy_quantity)
        remaining_sell_qty = position_limit + (position - take_sell_quantity)
    
        # Delta between current effective position and desired
        position_diff = effective_position - desired_position
    
        # If we are long and need to reduce to desired_position
        if position_diff > 0 and remaining_sell_qty > 0:
            # Look for aggressive buyers (bid >= fair + width)
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if price >= fair_for_ask:
                    qty = min(abs(volume), position_diff, remaining_sell_qty)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        effective_position -= qty
                        take_sell_quantity += qty
                        remaining_sell_qty -= qty
                        position_diff -= qty
                        if position_diff <= 0:
                            break
    
        # If we are short and need to increase to desired_position
        if position_diff < 0 and remaining_buy_qty > 0:
            for price, volume in sorted(order_depth.sell_orders.items()):
                if price <= fair_for_bid:
                    qty = min(abs(volume), abs(position_diff), remaining_buy_qty)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        effective_position += qty
                        take_buy_quantity += qty
                        remaining_buy_qty -= qty
                        position_diff += qty
                        if position_diff >= 0:
                            break
    
        return orders, take_buy_quantity, take_sell_quantity
    
    
    def make_orders(self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        desired_edge: float,
        default_order_size: int,
        disregard_edge: float,
        direction_sensitive: bool,
        max_deviation: int,
        take_buy_quantity: int,
        take_sell_quantity: int,
        desired_position: int = 0,
        ) -> (List[Order], int, int):
    
        orders: List[Order] = []
    
        # Step 1: Define edge-based range
        your_bid = fair_value - desired_edge
        your_ask = fair_value + desired_edge
    
        # Step 2: Filter out quotes too close to fair
        competing_bids = [
            price for price in order_depth.buy_orders
            if fair_value - price > disregard_edge
        ]
        competing_asks = [
            price for price in order_depth.sell_orders
            if price - fair_value > disregard_edge
        ]
    
        # Step 3: Set prices (penny competing quotes if any), but not at fair value if diregard_edge = 0
        bid_price = min(max(competing_bids)+ 1, fair_value - 1)  if competing_bids else math.floor(your_bid)
        ask_price = max(min(competing_asks) - 1, fair_value + 1) if competing_asks else math.ceil(your_ask)
    
        # Step 4: Compute remaining capacity
        max_buy_capacity = position_limit - (position + take_buy_quantity)
        max_sell_capacity = position + position_limit - take_sell_quantity
    
        # Step 5: Direction-sensitive sizing (optional)
        inventory_delta = position - desired_position
    
        if direction_sensitive:
            # Linear scaling: 1 at center, 0 at ±max_deviation
            if inventory_delta >= 0:
                # Long → reduce bid size
                scale_bid = max(0.0, 1 - inventory_delta / max_deviation)
                scale_ask = 1.0  # still okay to sell
            else:
                # Short → reduce ask size
                scale_ask = max(0.0, 1 + inventory_delta / max_deviation)
                scale_bid = 1.0  # still okay to buy
        else:
            scale_bid = scale_ask = 1.0
    
        # Step 6: Final quote sizes with clipping
        buy_quantity = min(default_order_size * scale_bid, max_buy_capacity)
        sell_quantity = min(default_order_size * scale_ask, max_sell_capacity)
    
        # Step 7: Round and place orders
        if buy_quantity > 0:
            orders.append(Order(product, int(bid_price), int(buy_quantity)))
        if sell_quantity > 0:
            orders.append(Order(product, int(ask_price), -int(sell_quantity)))
    
        return orders, int(buy_quantity), int(sell_quantity)

    def sweep_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        position_limit: int,
        take_buy_quantity: int,
        take_sell_quantity: int,
        desired_position: int = 0
        ) -> (List[Order], int, int):
    
        orders: List[Order] = []
    
        # Account for already-placed sweep orders
        effective_position = position + take_buy_quantity - take_sell_quantity
        position_diff = desired_position - effective_position
    
        # How much we can still buy/sell
        remaining_buy_qty = position_limit - effective_position
        remaining_sell_qty = effective_position + position_limit
    
        # Sweep to buy
        if position_diff > 0 and remaining_buy_qty > 0:
            for price, volume in sorted(order_depth.sell_orders.items()):
                qty = min(abs(volume), position_diff, remaining_buy_qty)
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    effective_position += qty
                    take_buy_quantity += qty
                    remaining_buy_qty -= qty
                    position_diff -= qty
                    if position_diff <= 0:
                        break
    
        # Sweep to sell
        elif position_diff < 0 and remaining_sell_qty > 0:
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                qty = min(abs(volume), abs(position_diff), remaining_sell_qty)
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    effective_position -= qty
                    take_sell_quantity += qty
                    remaining_sell_qty -= qty
                    position_diff += qty
                    if position_diff >= 0:
                        break
    
        return orders, take_buy_quantity, take_sell_quantity
    
    def target_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        position_limit: int,
        desired_position: int = 0
    ) -> List[Order]:
        """
        Attempts to move toward the desired_position using ONLY the best bid/ask level.
    
        Args:
            product (str): Product name
            order_depth (OrderDepth): Current market depth
            fair_value (float): Unused here but kept for signature consistency
            position (int): Current position
            position_limit (int): Max allowed absolute position
            desired_position (int): Target position
    
        Returns:
            List[Order]: List of marketable orders at best price only
        """
        orders: List[Order] = []
        position_diff = desired_position - position
    
        buy_quantity = 0
        sell_quantity = 0
        
        if position_diff > 0:
            # Need to buy
            best_ask = min(order_depth.sell_orders, default=None)
            if best_ask is not None:
                volume = order_depth.sell_orders[best_ask]
                buy_quantity = min(position_diff, position_limit - position, abs(volume))
                if buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
    
        elif position_diff < 0:
            # Need to sell
            best_bid = max(order_depth.buy_orders, default=None)
            if best_bid is not None:
                volume = order_depth.buy_orders[best_bid]
                sell_quantity = min(-position_diff, position + position_limit, abs(volume))
                if sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
    
        return orders, buy_quantity, sell_quantity
    
    
    
    def calculate_spread(self, spread_product: str, order_depths: dict) -> float | None:
        """
        Calculates the spread value for a given spread product based on current market mid prices.
    
        Args:
            spread_product (str): Product.SPREAD1...SPREAD4
            order_depths (dict): {product: OrderDepth}
    
        Returns:
            float or None if a product is missing bid/ask
        """
        spread_info = self.params[spread_product]
        base_product = spread_info["base"]
        components = spread_info["components"]
    
        try:

            # Base mid price
            base_depth = order_depths[base_product]
            base_bid, base_ask, *_ = self.best_orders(base_product, base_depth)
            base_mid = (base_bid + base_ask) / 2
        except:
            return None
    
        component_total = 0
        for comp_product, qty in components.items():
            try:
                comp_depth = order_depths[comp_product]
                comp_bid, comp_ask, *_ = self.best_orders(comp_product, comp_depth)
                comp_mid = (comp_bid + comp_ask) / 2
                component_total += qty * comp_mid
            except:
                return None
    
        spread = base_mid - component_total
        return spread
        
    
    def update_spread_history(self, 
                              trader_object: dict, 
                              spread_product: str, 
                              new_value: float):
        """
        Appends a new spread value to the history and ensures it remains within window size.
    
        Args:
            trader_object (dict): Trader state object passed across time
            spread_product (str): e.g., Product.SPREAD1
            new_value (float): New spread value to store
        """
        history = trader_object["spread_history"][spread_product]
        history.append(new_value)
    
        max_len = self.params[spread_product]["spread_std_window"]
        if len(history) > max_len:
            history.pop(0)
    
        trader_object["spread_history"][spread_product] = history
    
    
    def get_rolling_std(self, 
                        trader_object: dict, 
                        spread_product: str) -> float | None:
        """
        Returns rolling std deviation for a spread if enough data is available.
    
        Returns:
            float or None
        """
        print(f"update_spread_history is a {type(trader_object)}")
        history = trader_object["spread_history"][spread_product]
        window = self.params[spread_product]["spread_std_window"]
    
        if len(history) < window:
            return None  # Not enough data yet
        
        return float(np.std(history))

    

    def calculate_spread_direction(self, 
                               trader_object: dict, 
                               spread_product: str,        
                               current_spread: float) -> int:
        """
        Determines persistent trading signal (long/short/neutral) based on z-score.
        Signal stays until spread crosses back through mean.
    
        Args:
            trader_object (dict): Trader state holding spread history + previous signals
            spread_product (str): e.g., Product.SPREAD1
            current_spread (float): Current calculated spread value
    
        Returns:
            int: +1 (go long), -1 (go short), 0 (neutral/exit)
        """
        # Get current signal
        current_signal = trader_object["spread_signal"][spread_product]
        
        rolling_std = self.get_rolling_std(trader_object, spread_product)
        if rolling_std is None or rolling_std == 0:
            return current_signal
    
        mean = self.params[spread_product]["default_spread_mean"]
        threshold = self.params[spread_product]["zscore_threshold"]
    
        z_score = (current_spread - mean) / rolling_std
    
        
    
        # Only assign new position if we don't already have one
        if z_score > threshold:
            return -1   # enter long (inverted logic now fixed)
        elif z_score < -threshold:
            return 1  # enter short
    
        # Hold position if already in one
        return current_signal
        
        
    def calculate_net_positions(self, trader_object: dict, order_depths: dict) -> tuple[int, int, int, int, int]:
        """
        Aggregates net positions across all spreads and scales each spread based on the most limiting product.
        Returns positions in the fixed order: (PB1, PB2, CROISSANTS, JAMS, DJEMBES)
        """
        net_pos = {
            Product.PB1: 0,
            Product.PB2: 0,
            Product.CROISSANTS: 0,
            Product.JAMS: 0,
            Product.DJEMBES: 0
        }
    
        for spread_product in [Product.SPREAD1, Product.SPREAD2, Product.SPREAD3, Product.SPREAD4]:
            spread_value = self.calculate_spread(spread_product, order_depths)
    
            if spread_value is None:
                continue
    
            self.update_spread_history(trader_object, spread_product, spread_value)
            direction = self.calculate_spread_direction(trader_object, spread_product, spread_value)
            trader_object["spread_signal"][spread_product] = direction
            
            if direction == 0:
                
                continue
    
            spread_params = self.params[spread_product]
            base = spread_params["base"]
            components = spread_params["components"]
    
            # Get absolute quantity requirements per 1 spread unit
            spread_requirements = {base: 1}
            for comp_product, qty in components.items():
                spread_requirements[comp_product] = abs(qty)
    
            # Find max number of spread units we can trade without violating any limit
            max_units = float('inf')
            for product, qty in spread_requirements.items():
                product_limit = self.LIMIT[product]
                max_spread_units = product_limit // qty
                max_units = min(max_units, max_spread_units)
    
            # Apply scaled positions (using signed direction)
            scale = direction * max_units
    
            net_pos[base] += scale
            for comp_product, qty in components.items():
                net_pos[comp_product] += -scale * qty  # inverse direction for components
    
        return (
            int(round(net_pos[Product.PB1])),
            int(round(net_pos[Product.PB2])),
            int(round(net_pos[Product.CROISSANTS])),
            int(round(net_pos[Product.JAMS])),
            int(round(net_pos[Product.DJEMBES]))
        )
        
    """
    Main run function
    """
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        print('hiiiiiiiiiiiii')
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
                "signal": {
                    Product.SQUID_INK: 0},
                "spread_signal": {
                    Product.SPREAD1: 0,
                    Product.SPREAD2: 0,
                    Product.SPREAD3: 0,
                    Product.SPREAD4: 0
                    },
                "spread_history": {  # NEW: empty rolling history buffers
                    Product.SPREAD1: [],
                    Product.SPREAD2: [],
                    Product.SPREAD3: [],
                    Product.SPREAD4: []
                }
            }
            
        print("DECODE THIIIIS IS A f{type(traderObject)")
        
        """ SPREAD TRADES """
        
        core_products = [
            Product.PB1,
            Product.PB2,
            Product.CROISSANTS,
            Product.JAMS,
            Product.DJEMBES
        ]
        
        # Get target net positions from spreads
        pb1_pos, pb2_pos, croissant_pos, jam_pos, djembe_pos = self.calculate_net_positions(traderObject, state.order_depths)
        desired_positions = {
            Product.PB1: pb1_pos,
            Product.PB2: pb2_pos,
            Product.CROISSANTS: croissant_pos,
            Product.JAMS: jam_pos,
            Product.DJEMBES: djembe_pos
        }
        
        
        for product in core_products:
            if product not in state.order_depths:
                continue  # skip if no data
        
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            desired_position = desired_positions[product]
        
            # Get current market mid price
            best_bid, best_ask, _, _ = self.best_orders(product, order_depth)
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
                self.params[product]["fair_value"] = mid_price  # update fair value
        
            print(f"{product} desired posn {desired_position}")
        
            # Generate target orders
            orders, target_buy_quantity, target_sell_quantity = self.target_orders(
                                                        product=product,
                                                        order_depth=order_depth,
                                                        fair_value=self.params[product]["fair_value"],
                                                        position=position,
                                                        position_limit=self.LIMIT[product],
                                                        desired_position=desired_position
                                                    )
            
            
            make_orders, make_buy_quantity, make_sell_quantity = self.make_orders(
                                                        product,
                                                        order_depth,  # Product-specific OrderDepth
                                                        mid_price,
                                                        position,
                                                        self.LIMIT[product],
                                                        desired_edge = 5,
                                                        default_order_size = 15,
                                                        disregard_edge = 1,
                                                        direction_sensitive = True,
                                                        desired_position = 0,
                                                        max_deviation = 5,
                                                        take_buy_quantity = target_buy_quantity,
                                                        take_sell_quantity = target_sell_quantity
                                                        )
            
            if orders:
                result[product] = (orders + make_orders)
                    
        """ 
        Tidying up
        """    
        
        """ SQUID INK """
        
        if Product.SQUID_INK in state.order_depths:
            ink_position = state.position.get(Product.SQUID_INK, 0)
            ink_order_depth = state.order_depths[Product.SQUID_INK]
            
            ink_mm_bid, ink_mm_ask, _, _ = self.mm_orders(Product.SQUID_INK, ink_order_depth)
            ink_mm_mid = (ink_mm_bid + ink_mm_ask) / 2
        
            ink_bid, ink_ask, _, _ = self.best_orders(Product.SQUID_INK, ink_order_depth)    
                        
            ink_fair_value = self.ewma(Product.SQUID_INK, traderObject, ink_mm_mid, self.params[Product.SQUID_INK]["ewma_beta"])
            
            deviation = ink_mm_mid - ink_fair_value
            max_bias = 50
            
            ink_desired_position = int(-max_bias * (deviation / 50)) # slightly optimized via grid search
            
            if ink_fair_value:
                self.params[Product.SQUID_INK]["fair_value"] = ink_fair_value
            
            ink_buy_quantity = 0
            ink_sell_quantity = 0
            
            # taking orders
            ink_take_orders, ink_take_buy_quantity, ink_take_sell_quantity = self.take_orders(Product.SQUID_INK,
                                                                                                 ink_order_depth,
                                                                                                 ink_mm_mid,
                                                                                                 ink_position,
                                                                                                 self.LIMIT[Product.SQUID_INK])
            
            ink_buy_quantity += ink_take_buy_quantity
            ink_sell_quantity += ink_take_sell_quantity
            
            # if we have a signal to empty position
            if traderObject["signal"][Product.SQUID_INK] == 1:
                ink_desired_position = 0
                
                ink_sweep_orders, _, _ = self.sweep_orders(Product.SQUID_INK,
                                                            ink_order_depth,
                                                            (ink_bid + ink_ask)/2,
                                                            ink_position,
                                                            self.LIMIT[Product.SQUID_INK],
                                                            ink_buy_quantity,
                                                            ink_sell_quantity,
                                                            desired_position = 0)
                
                # turn off signal
                traderObject["signal"][Product.SQUID_INK] = 0
                ink_make_orders = []
                ink_clear_orders = []

            
            # if no existing signal, check if we should initiate a signal
            elif abs(ink_desired_position) >= self.LIMIT[Product.SQUID_INK] and traderObject["signal"][Product.SQUID_INK] == 0:
                ink_sweep_orders, _, _ = self.sweep_orders(Product.SQUID_INK,
                                                            ink_order_depth,
                                                            (ink_bid + ink_ask)/2,
                                                            ink_position,
                                                            self.LIMIT[Product.SQUID_INK],
                                                            ink_buy_quantity,
                                                            ink_sell_quantity,
                                                            desired_position = ink_desired_position)
                
                # signal to empty position in this amount of rounds
                traderObject["signal"][Product.SQUID_INK] = 5
                ink_make_orders = []
                ink_clear_orders = []
                
        
            # otherwise, market making and clearing, get to 0 if possible but don't force it
            else:
                ink_make_orders, ink_make_buy_quantity, ink_make_sell_quantity = self.make_orders(Product.SQUID_INK,
                                                            ink_order_depth,  # Product-specific OrderDepth
                                                            ink_mm_mid,
                                                            ink_position,
                                                            self.LIMIT[Product.SQUID_INK],
                                                            desired_edge = 2,
                                                            default_order_size = 15,
                                                            disregard_edge = 0.5,
                                                            direction_sensitive = True,
                                                            desired_position = 0,
                                                            max_deviation = 5,
                                                            take_buy_quantity = ink_buy_quantity,
                                                            take_sell_quantity = ink_sell_quantity
                                                            )
            
            
                
                ink_buy_quantity += ink_make_buy_quantity
                ink_sell_quantity += ink_make_sell_quantity
                
                # clear orders
                ink_clear_orders, _, _ = self.clear_orders(Product.SQUID_INK,
                                                            ink_order_depth,
                                                            ink_mm_mid,
                                                            ink_position,
                                                            self.LIMIT[Product.SQUID_INK],
                                                            self.params[Product.SQUID_INK]["clear_width"],
                                                            ink_buy_quantity,
                                                            ink_sell_quantity,
                                                            desired_position = 0)
                
                ink_sweep_orders = []
        
        
                # remove one from the holding 
                if traderObject["signal"][Product.SQUID_INK] > 1:
                    traderObject["signal"][Product.SQUID_INK] -= 1
                
        """ RAINFOREST RESIN """
        
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            
            # taking orders
            resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.take_orders(Product.RAINFOREST_RESIN,
                                                                      resin_order_depth,
                                                                      self.params[Product.RAINFOREST_RESIN]["fair_value"],
                                                                      resin_position,
                                                                      self.LIMIT[Product.RAINFOREST_RESIN])
        
            
            resin_make_orders, _, _ = self.make_orders(Product.RAINFOREST_RESIN,
                                                        resin_order_depth,  # Product-specific OrderDepth
                                                        10000,
                                                        resin_position,
                                                        self.LIMIT[Product.RAINFOREST_RESIN],
                                                        desired_edge = 7,
                                                        default_order_size = 15 ,
                                                        disregard_edge = 1,
                                                        direction_sensitive = True,
                                                        desired_position = 0,
                                                        max_deviation = 40,
                                                        take_buy_quantity = resin_take_buy_quantity,
                                                        take_sell_quantity = resin_take_sell_quantity
                                                        )
            
            # clearing orders
            resin_clear_orders, _, _ = self.clear_orders(Product.RAINFOREST_RESIN,
                                                        resin_order_depth,
                                                        self.params[Product.RAINFOREST_RESIN]["fair_value"],
                                                        resin_position,
                                                        self.LIMIT[Product.RAINFOREST_RESIN],
                                                        self.params[Product.RAINFOREST_RESIN]["clear_width"],
                                                        resin_take_buy_quantity,
                                                        resin_take_sell_quantity)

        
        """ KELP """
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_order_depth = state.order_depths[Product.KELP]
            
            kelp_mm_bid, kelp_mm_ask, _, _ = self.mm_orders(Product.KELP, kelp_order_depth)
            kelp_mm_mid = (kelp_mm_bid + kelp_mm_ask) / 2
        
            kelp_bid, kelp_ask, _, _ = self.best_orders(Product.KELP, kelp_order_depth)    
                        
            kelp_fair_value = self.ewma(Product.KELP, traderObject, kelp_mm_mid, self.params[Product.KELP]["ewma_beta"])
            if kelp_fair_value:
                self.params[Product.KELP]["fair_value"] = kelp_fair_value
            
            kelp_buy_quantity = 0
            kelp_sell_quantity = 0
            
            # taking orders
            kelp_take_orders, kelp_take_buy_quantity, kelp_take_sell_quantity = self.take_orders(Product.KELP,
                                                                                                 kelp_order_depth,
                                                                                                 self.params[Product.KELP]["fair_value"],
                                                                                                 kelp_position,
                                                                                                 self.LIMIT[Product.KELP])
            
            kelp_buy_quantity += kelp_take_buy_quantity
            kelp_sell_quantity += kelp_take_sell_quantity
            
            # making orders
            kelp_make_orders, kelp_make_buy_quantity, kelp_make_sell_quantity = self.make_orders(Product.KELP,
                                                        kelp_order_depth,  # Product-specific OrderDepth
                                                        kelp_fair_value,
                                                        kelp_position,
                                                        self.LIMIT[Product.KELP],
                                                        desired_edge = 1.5,
                                                        default_order_size = 15,
                                                        disregard_edge = 1,
                                                        direction_sensitive = True,
                                                        max_deviation = 50,
                                                        take_buy_quantity = kelp_buy_quantity,
                                                        take_sell_quantity = kelp_sell_quantity,
                                                        desired_position = 0)
            
            kelp_buy_quantity += kelp_make_buy_quantity
            kelp_sell_quantity += kelp_make_sell_quantity
            
            kelp_clear_orders, _, _ = self.clear_orders(Product.KELP,
                                                        kelp_order_depth,
                                                        kelp_mm_mid,
                                                        kelp_position,
                                                        self.LIMIT[Product.KELP],
                                                        self.params[Product.KELP]["clear_width"],
                                                        kelp_buy_quantity,
                                                        kelp_sell_quantity)
        
        result[Product.KELP] = (kelp_take_orders + kelp_make_orders + kelp_clear_orders)
        result[Product.RAINFOREST_RESIN] = (resin_take_orders + resin_make_orders + resin_clear_orders)            
        result[Product.SQUID_INK] = (ink_take_orders +  ink_clear_orders + ink_make_orders + ink_sweep_orders )
        
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData