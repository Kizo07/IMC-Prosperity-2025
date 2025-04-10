from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import json
import jsonpickle
import numpy as np
from typing import Any, List
from collections import deque  # For efficient sliding window operations

# Import both strategy classes
from ink_kelp_resin_take_make_v14 import Trader as TraderV14
from ink_kelp_resin_take_make_v5 import Trader as TraderV5

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


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

"""
Strategy switcher class
"""
class HybridTrader:
    def __init__(self):
        # Initialize both strategy instances
        self.trader_v14 = TraderV14()
        self.trader_v5 = TraderV5()
        
        # Thresholds for signal strength
        self.signal_threshold = {
            Product.SQUID_INK: 0.3,  # Threshold for ink imbalance
            Product.KELP: 0.25,      # Threshold for kelp imbalance
            Product.RAINFOREST_RESIN: 0.0  # Always use v14 for resin
        }
        
        # History of which strategy was used for each product
        self.strategy_history = {
            Product.SQUID_INK: [],
            Product.KELP: [],
            Product.RAINFOREST_RESIN: []
        }
        
        # Limit positions same as individual strategies
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}

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

    def detect_price_momentum(self, product, traderObject):
        """
        Detect price momentum using recent price history
        """
        if "price_history" not in traderObject:
            return 0
            
        if product not in traderObject["price_history"] or len(traderObject["price_history"][product]) < 5:
            return 0
            
        # Get last 5 prices
        recent_prices = list(traderObject["price_history"][product])[-5:]
        
        # Calculate momentum (simple slope)
        if len(recent_prices) >= 3:
            return (recent_prices[-1] - recent_prices[0]) / (len(recent_prices) - 1)
        return 0

    def should_use_v14_strategy(self, product, order_depth, traderObject):
        """
        Determine if we should use v14 strategy based on market signals
        """
        # Calculate book imbalance
        imbalance = abs(self.calculate_book_imbalance(order_depth))
        
        # Calculate price momentum
        momentum = abs(self.detect_price_momentum(product, traderObject))
        
        # Combine signals (we could weight these differently if needed)
        signal_strength = max(imbalance, momentum/5)  # Normalize momentum
        
        # Log the signal strength for debugging
        logger.print(f"Product: {product}, Signal strength: {signal_strength}, " 
                     f"Imbalance: {imbalance}, Momentum: {momentum}")
        
        # Use v14 if signal strength exceeds threshold
        return signal_strength > self.signal_threshold[product]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Initialize results
        result = {}
        conversions = 0
        
        # Load traderData or create new if first run
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        else:
            traderObject = {
                "prev_quote": {
                    Product.KELP: [],
                    Product.RAINFOREST_RESIN: [],
                    Product.SQUID_INK: []
                },
                "ewma": {
                    Product.KELP: None,
                    Product.RAINFOREST_RESIN: None,
                    Product.SQUID_INK: None
                },
                "rolling_mid_quotes": {},
                "price_history": {},
                "strategy_used": {}
            }
        
        # Process each product
        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                
                # Decide which strategy to use
                use_v14 = self.should_use_v14_strategy(product, order_depth, traderObject)
                
                # Record which strategy was used
                if "strategy_used" not in traderObject:
                    traderObject["strategy_used"] = {}
                traderObject["strategy_used"][product] = "v14" if use_v14 else "v5"
                
                # Log the strategy choice
                logger.print(f"Using {'v14' if use_v14 else 'v5'} strategy for {product}")
                
                # Execute the chosen strategy
                if use_v14:
                    # Get MM quotes for price tracking regardless of strategy
                    if hasattr(self.trader_v14, "mm_orders"):
                        mm_bid, mm_ask, _, _ = self.trader_v14.mm_orders(product, order_depth)
                        if mm_bid is not None and mm_ask is not None:
                            mm_mid = (mm_bid + mm_ask) / 2
                            
                            # Store prices in history for both traders
                            if "price_history" not in traderObject:
                                traderObject["price_history"] = {}
                            if product not in traderObject["price_history"]:
                                traderObject["price_history"][product] = deque(maxlen=100)
                            traderObject["price_history"][product].append(mm_mid)
                            
                            # Keep track of rolling mid quotes for v14
                            if hasattr(self.trader_v14, "rolling_mm_mid_quotes"):
                                self.trader_v14.rolling_mm_mid_quotes(product, order_depth, traderObject, 
                                                                     self.trader_v14.params[product].get("rolling_window", 50))
                    
                    # For SQUID_INK or KELP
                    if product == Product.SQUID_INK:
                        # Use v14's advanced market making
                        position = state.position.get(product, 0)
                        ink_mm_mid = mm_mid if 'mm_mid' in locals() else None
                        
                        if ink_mm_mid:
                            ink_fair_value = self.trader_v14.ewma(
                                product, 
                                traderObject, 
                                ink_mm_mid, 
                                self.trader_v14.params[product]["ewma_beta"]
                            )
                            
                            if ink_fair_value:
                                # Use v14's squid_ink_market_making
                                product_orders, _, _ = self.trader_v14.squid_ink_market_making(
                                    product,
                                    order_depth,
                                    ink_fair_value,
                                    position,
                                    self.LIMIT[product],
                                    traderObject,
                                    0, 0
                                )
                                result[product] = product_orders
                    
                    elif product == Product.KELP:
                        # Use v14's advanced market making
                        position = state.position.get(product, 0)
                        kelp_mm_mid = mm_mid if 'mm_mid' in locals() else None
                        
                        if kelp_mm_mid:
                            kelp_fair_value = self.trader_v14.ewma(
                                product, 
                                traderObject, 
                                kelp_mm_mid, 
                                self.trader_v14.params[product]["ewma_beta"]
                            )
                            
                            if kelp_fair_value:
                                # Use v14's kelp_market_making
                                product_orders, _, _ = self.trader_v14.kelp_market_making(
                                    product,
                                    order_depth,
                                    kelp_fair_value,
                                    position,
                                    self.LIMIT[product],
                                    traderObject,
                                    0, 0
                                )
                                result[product] = product_orders
                    
                    elif product == Product.RAINFOREST_RESIN:
                        # Use v14 strategy for RAINFOREST_RESIN
                        position = state.position.get(product, 0)
                        
                        # Taking orders
                        resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.trader_v14.take_orders(
                            product,
                            order_depth,
                            self.trader_v14.params[product]["fair_value"],
                            position,
                            self.LIMIT[product]
                        )
                        
                        # Making orders
                        required_edge = 4
                        default_order_size = 15
                        
                        resin_make_orders, _, _ = self.trader_v14.resin_make_orders(
                            product, 
                            order_depth, 
                            self.trader_v14.params[product]["fair_value"], 
                            position, 
                            self.LIMIT[product], 
                            required_edge,
                            default_order_size, 
                            resin_take_buy_quantity, 
                            resin_take_sell_quantity
                        )
                        
                        # Clearing orders
                        resin_clear_orders, _, _ = self.trader_v14.clear_orders(
                            product,
                            order_depth,
                            self.trader_v14.params[product]["fair_value"],
                            position,
                            self.LIMIT[product],
                            self.trader_v14.params[product]["clear_width"],
                            resin_take_buy_quantity,
                            resin_take_sell_quantity
                        )
                        
                        result[product] = resin_take_orders + resin_make_orders + resin_clear_orders
                        
                else:
                    # Use v5 strategy
                    position = state.position.get(product, 0)
                    
                    # Get MM quotes for price tracking regardless of strategy
                    mm_bid, mm_ask, _, _ = self.trader_v5.mm_orders(product, order_depth)
                    best_bid, best_ask, _, _ = self.trader_v5.best_orders(product, order_depth)
                    
                    if mm_bid is not None and mm_ask is not None:
                        mm_mid = (mm_bid + mm_ask) / 2
                        
                        # Store prices in history
                        if "price_history" not in traderObject:
                            traderObject["price_history"] = {}
                        if product not in traderObject["price_history"]:
                            traderObject["price_history"][product] = deque(maxlen=100)
                        traderObject["price_history"][product].append(mm_mid)
                    
                    if product == Product.SQUID_INK:
                        ink_fair_value = self.trader_v5.ewma(product, traderObject, mm_mid, self.trader_v5.params[product]["ewma_beta"])
                        if ink_fair_value:
                            self.trader_v5.params[product]["fair_value"] = ink_fair_value
                            
                        # Taking orders
                        ink_take_orders, ink_take_buy_quantity, ink_take_sell_quantity = self.trader_v5.take_orders(
                            product,
                            order_depth,
                            mm_mid,
                            position,
                            self.LIMIT[product]
                        )
                        
                        # One way market making
                        ink_make_orders, _, _ = self.trader_v5.ink_make_orders(
                            product,
                            order_depth,
                            mm_mid,
                            position,
                            self.LIMIT[product],
                            default_order_size = 15,
                            take_buy_quantity= ink_take_buy_quantity,
                            take_sell_quantity= ink_take_sell_quantity
                        )
                        
                        # Clear orders
                        ink_clear_orders, _, _ = self.trader_v5.clear_orders(
                            product,
                            order_depth,
                            mm_mid,
                            position,
                            self.LIMIT[product],
                            self.trader_v5.params[product]["clear_width"],
                            ink_take_buy_quantity,
                            ink_take_sell_quantity
                        )
                        
                        result[product] = ink_take_orders + ink_clear_orders + ink_make_orders
                        
                    elif product == Product.KELP:
                        kelp_fair_value = self.trader_v5.ewma(product, traderObject, mm_mid, self.trader_v5.params[product]["ewma_beta"])
                        if kelp_fair_value:
                            self.trader_v5.params[product]["fair_value"] = kelp_fair_value
                            
                        # Taking orders
                        kelp_take_orders, kelp_take_buy_quantity, kelp_take_sell_quantity = self.trader_v5.take_orders(
                            product,
                            order_depth,
                            self.trader_v5.params[product]["fair_value"],
                            position,
                            self.LIMIT[product]
                        )
                        
                        # Making orders with v5 method
                        kelp_make_orders, _, _ = self.trader_v5.kelp_make_orders(
                            product,
                            order_depth,
                            mm_mid,
                            position=position,
                            position_limit=50,
                            default_order_size=15,
                            take_buy_quantity=kelp_take_buy_quantity,
                            take_sell_quantity=kelp_take_sell_quantity,
                            bias=1,
                            state=state
                        )
                        
                        kelp_clear_orders, _, _ = self.trader_v5.clear_orders(
                            product,
                            order_depth,
                            mm_mid,
                            position,
                            self.LIMIT[product],
                            self.trader_v5.params[product]["clear_width"],
                            kelp_take_buy_quantity,
                            kelp_take_sell_quantity
                        )
                        
                        result[product] = kelp_take_orders + kelp_make_orders + kelp_clear_orders
                        
                    elif product == Product.RAINFOREST_RESIN:
                        # Fallback to v5 strategy for RAINFOREST_RESIN if ever needed
                        # Taking orders
                        resin_take_orders, resin_take_buy_quantity, resin_take_sell_quantity = self.trader_v5.take_orders(
                            product,
                            order_depth,
                            self.trader_v5.params[product]["fair_value"],
                            position,
                            self.LIMIT[product]
                        )
                        
                        # Making orders
                        required_edge = 4
                        default_order_size = 15
                        
                        resin_make_orders, _, _ = self.trader_v5.resin_make_orders(
                            product, 
                            order_depth, 
                            self.trader_v5.params[product]["fair_value"], 
                            position, 
                            self.LIMIT[product], 
                            required_edge,
                            default_order_size, 
                            resin_take_buy_quantity, 
                            resin_take_sell_quantity
                        )
                        
                        # Clearing orders
                        resin_clear_orders, _, _ = self.trader_v5.clear_orders(
                            product,
                            order_depth,
                            self.trader_v5.params[product]["fair_value"],
                            position,
                            self.LIMIT[product],
                            self.trader_v5.params[product]["clear_width"],
                            resin_take_buy_quantity,
                            resin_take_sell_quantity
                        )
                        
                        result[product] = resin_take_orders + resin_make_orders + resin_clear_orders
        
        # Encode trader data and flush logs
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        
        return result, conversions, traderData

if __name__ == "__main__":
    # For local testing
    trader = HybridTrader()