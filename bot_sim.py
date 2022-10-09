import pandas as pd
import datetime
import os
import time
from collections import deque
from talib import EMA
from package.utils import *
import warnings
from package.dataset import *

warnings.filterwarnings("ignore")


class TradingBot_v3_sim:
    def __init__(
        self,
        client,
        crypto,
        strategy,
        leverage: int = 1,
        balance: int = 1000000,
        name=None,
        invest_ratio=0.1,
        constant_invest_amount=None,
        TRX_FEE_ratio=0.0004,
        stop_loss_ratio=0.015,
        unlimit: bool = False,
        sim_stop_profit_loss=False,
        sim_stop_interval=None,
        stop_profit_ratio=None,
        simulate_from_csv=False,
    ):
        self.crypto = crypto
        self.client = client
        self.balance = balance
        self.name = name
        self.strategy = strategy  # trading strategy
        self.leverage = leverage
        self.order_history = pd.DataFrame(
            columns=[
                "pair",
                "order time",
                "end time",
                "true amount",
                "contract amount",
                "order price",
                "cross price",
                "profit rate",
                "profit",
                "leverage",
                "TRX fee",
                "position",
                "balance",
            ]
        )
        self.orders = deque()
        self.invest_ratio = invest_ratio  # Investment amount = balance * invest_ratio
        self.TRX_FEE_ratio = TRX_FEE_ratio  # Simulate Transaction fees
        self.stop_loss_ratio = stop_loss_ratio
        self.stop_profit_ratio = stop_profit_ratio
        self.unlimit = unlimit  # unlimit balance
        self.constant_invest_amount = (
            constant_invest_amount  # invest with constant amount
        )
        self.sim_stop_profit_loss = sim_stop_profit_loss
        self.sim_stop_interval = sim_stop_interval
        self.simulate_from_csv = simulate_from_csv  # use pre-download data
        if simulate_from_csv:
            self.sim_data_df = fetch_sim_data_from_csv()

        if self.constant_invest_amount and not self.unlimit:
            raise Exception("Set unlimit to True when using constant_invest_amount")
        if self.unlimit and not self.constant_invest_amount:
            raise Exception("Set constant_invest_amount when unlimit is True")
        if self.unlimit:
            self.balance = float("inf")

        if self.sim_stop_profit_loss and not self.sim_stop_interval:
            raise Exception(
                "Set sim_stop_profit_loss to Ture but not setting sim_stop_interval"
            )
        if self.sim_stop_profit_loss and (
            not self.stop_loss_ratio and not stop_profit_ratio
        ):
            raise Exception(
                "Set sim_stop_profit_loss to Ture but not provide stop_loss_ratio or stop_profit_ratio"
            )

    def _buy(self, data, crypto_amount=None, spend_amount=0):  # 做多
        if crypto_amount and spend_amount:
            raise Exception("crypto_amount and spend_amount provide simultaneously")
        fee = spend_amount * self.leverage * self.TRX_FEE_ratio
        if self.balance < spend_amount + fee: #check balance
            print("Not Enough Balance!")
            return False
        self.balance -= spend_amount + fee
        #place the order
        info = {
            "order_time": data[6] + 1,
            "price": float(data[4]),
            "amount": spend_amount,
            "side": Status.BUY,
            "hold_time": 0,
            "leverage": self.leverage,
            "fee": fee,
        }
        self.orders.append(info)
        print(
            "-------Place BUY Order-------\n",
            f"Order time: {datetime.datetime.fromtimestamp((data[6]+1)/1000)}\n",
            f"Order Price: {data[4]}\n",
            f"True Amount: {spend_amount}USDT\n",
            f"Contract Amount: {spend_amount*self.leverage}USDT\n",
            f"TRX FEE: {spend_amount * self.leverage * self.TRX_FEE_ratio}USDT\n",
            f"Leverage: {self.leverage}\n",
        )
        return True

    def _sell(self, data, crypto_amount=None, spend_amount=0):  # 做空
        if crypto_amount and spend_amount:
            raise Exception("crypto_amount and spend_amount provide simultaneously")
        fee = spend_amount * self.leverage * self.TRX_FEE_ratio
        #check blance
        if self.balance < spend_amount + fee:
            print("Not Enough Balance!")
            return False
        self.balance -= spend_amount + fee
        # place order
        info = {
            "order_time": data[6] + 1,
            "price": float(data[4]),
            "amount": spend_amount,
            "side": Status.SELL,
            "hold_time": 0,
            "leverage": self.leverage,
            "fee": fee,
        }
        self.orders.append(info)
        print(
            "-------Place SELL Order-------\n",
            f"Order time: {datetime.datetime.fromtimestamp((data[6]+1)/1000)}\n",
            f"Order Price: {data[4]}\n",
            f"True Amount: {spend_amount}USDT\n",
            f"Contract Amount: {spend_amount*self.leverage}USDT\n",
            f"TRX FEE: {spend_amount * self.leverage * self.TRX_FEE_ratio}USDT\n",
            f"Leverage: {self.leverage}\n",
        )
        return True

    def _liquidate(self, order, data, cross_ratio=1, simulate=False):  # 平倉
        # if simluate == true the bot will simluate TP/SL
        if simulate:
            # Rough check if stop in this interval
            if self.stop_loss_ratio and self.stop_profit_ratio:
                if order["side"] == Status.BUY and (float(data[2])-float(order["price"]))/float(order["price"])<self.stop_profit_ratio and (float(data[3])-float(order["price"]))/float(order["price"])>self.stop_loss_ratio*-1:
                    return False
                elif order["side"] == Status.SELL and (float(order["price"])-float(data[3]))/float(order["price"])<self.stop_profit_ratio and (float(order["price"])-float(data[2]))/float(order["price"])>self.stop_loss_ratio*-1:
                    return False
            
            elif self.stop_loss_ratio:
                if order["side"] == Status.BUY and (float(data[3])-float(order["price"]))/float(order["price"])>self.stop_loss_ratio*-1:
                    return False
                elif order["side"] == Status.SELL and (float(order["price"])-float(data[2]))/float(order["price"])>self.stop_loss_ratio*-1:
                    return False
            elif self.stop_profit_ratio:
                if order["side"] == Status.BUY and (float(data[2])-float(order["price"]))/float(order["price"])<self.stop_profit_ratio:
                    return False
                elif order["side"] == Status.SELL and (float(order["price"])-float(data[3]))/float(order["price"])<self.stop_profit_ratio:
                    return False
            # did stop in this interval so find the exact time
            stop_status, cross_price, cross_time = self._simulate_stop_profit_loss(
                order["price"],
                data[0],
                data[6] - 1,
                order["side"],
                self.sim_stop_interval,
            )
            if not stop_status:
                return False
            
            #order is cross by TP/SL
            else:
                print(f"\n{stop_status}\n")
                #calculate profit
                if stop_status == Stop_status.LOSS:
                    profit_ratio = -1 * self.stop_loss_ratio * order["leverage"]
                    profit = profit_ratio * order["amount"] * cross_ratio
                    total = order["amount"] * cross_ratio + profit
                    cur_price = cross_price
                elif stop_status == Stop_status.PROFIT:
                    profit_ratio = self.stop_profit_ratio * order["leverage"]
                    profit = profit_ratio * order["amount"] * cross_ratio
                    total = order["amount"] * cross_ratio + profit
                    cur_price = cross_price
                #Change the Status
                self.strategy.change_status(Status.OBSERVING)
        else:
            # Actually liquidate the order
            cur_price = float(data[4])  # open price of next market time section (approximate)
            #calculate profit
            if order["side"] == Status.BUY:
                profit_ratio = ((cur_price - order["price"]) / order["price"]) * order[
                    "leverage"
                ]
            elif order["side"] == Status.SELL:
                profit_ratio = ((order["price"] - cur_price) / order["price"]) * order[
                    "leverage"
                ]
            profit = order["amount"] * cross_ratio * profit_ratio
            total = order["amount"] * cross_ratio + profit
            cross_time = data[6]
        #TRX fee
        fee = total * self.TRX_FEE_ratio
        self.balance += total - fee
        order_p = order["price"]
        if order["side"] == Status.BUY:
            print(
                f"-------Cross Order  -------\n",
                f"Order type: BUY\n",
                f"Order Price: {order_p}\n",
                f"Cross Price: {cur_price}\n",
                f"Profit Ratio: {profit_ratio}\n",
                f"Profit: {profit}\n",
                f"Leverage: {self.leverage}\n",
                f"TRX FEE: {fee}\n",
                f"Balance: {self.balance}\n",
            )
        elif order["side"] == Status.SELL:
            print(
                f"-------Cross Order  -------\n",
                f"Order type: SELL\n",
                f"Order Price: {order_p}\n",
                f"Cross Price: {cur_price}\n",
                f"Profit Ratio: {profit_ratio}\n",
                f"Profit: {profit}\n",
                f"Leverage: {self.leverage}\n",
                f"TRX FEE: {fee}\n",
                f"Balance: {self.balance}\n",
            )
        #Write df
        self.order_history.loc[len(self.order_history)] = [
            self.crypto,
            datetime.datetime.fromtimestamp(order["order_time"] / 1000),
            datetime.datetime.fromtimestamp(cross_time / 1000),
            str(order["amount"]) + "USDT",
            str(order["amount"] * order["leverage"]) + "USDT",
            order["price"],
            cur_price,
            profit_ratio,
            profit,
            order["leverage"],
            order["fee"]+fee,
            "BUY" if order["side"] == Status.BUY else "SELL",
            self.balance,
        ]

        return True

    def _check_cross(self, data):
        num_orders = len(self.orders)
        examined = 0
        # iter all orders
        while self.orders and examined < num_orders:
            order = self.orders.popleft()
            # CHeck stop loss & profit
            if self.sim_stop_profit_loss and self._liquidate(
                order, data, simulate=True
            ):
                examined += 1
                continue
            # Cross
            if self.strategy.eval_cross(order):
                self._liquidate(
                    order, data, simulate=False
                )  # consider cross ratio when pop
                examined += 1
                continue
            # Keep the uncrossed order
            self.orders.append(order)
            examined += 1
        return

    def _increase_time(
        self,
    ):
        # increase the time the order being hold
        for order in self.orders:
            order["hold_time"] += 1
        return

    def _simulate_stop_profit_loss(self, order_price, start_time: int, end_time: int, position, interval="1m"):
        # Pull from Binance API (might triger error, adjust sleep time in dataset.py)
        if not self.simulate_from_csv:
            data_list = fetch_data(
                self.client,
                self.crypto,
                interval,
                start_time_int=start_time / 1000,
                end_time_int=end_time / 1000,
            )
        #Get 1 min klines from csv
        else:
            print(
                "---------DATA INFO---------\n",
                f"Fetch Data: {self.crypto}\n",
                f"Data Start from: {datetime.datetime.fromtimestamp(start_time/1000)}\n",
                f"Data End at: {datetime.datetime.fromtimestamp(end_time/1000)}\n",
                f"Sep. Interval: 1min",
            )
            start_index = self.sim_data_df.loc[
                self.sim_data_df["0"] == start_time
            ].index[0]
            data_list = [
                self.sim_data_df.iloc[start_index : start_index + 60].values.tolist()
            ]
        # CHECK TP/SL
        for sim_data in data_list:
            for data in sim_data:
                if position == Status.BUY:
                    bottom = float(data[3])
                    top = float(data[2])
                    if (
                        self.stop_loss_ratio
                        and (bottom - order_price) / order_price
                        <= -1 * self.stop_loss_ratio
                    ):
                        return Stop_status.LOSS, bottom, data[0]
                    if (
                        self.stop_profit_ratio
                        and (top - order_price) / order_price >= self.stop_profit_ratio
                    ):
                        return Stop_status.PROFIT, top, data[0]

                elif position == Status.SELL:
                    bottom = float(data[3])
                    top = float(data[2])
                    if self.stop_loss_ratio and (order_price - top) / order_price <= -1 * self.stop_loss_ratio:
                        return Stop_status.LOSS, top, data[0]
                    if self.stop_profit_ratio and (order_price - bottom) / order_price >= self.stop_profit_ratio:
                        return Stop_status.PROFIT, bottom, data[0]

        return None, None, None

    def eval(self, data):
        # eval Market
        self.strategy.eval_market(data)

        # increase hold_time
        if self.orders != []:
            self._increase_time()
        # check cross simulate stop loss & profit
        if self.orders != []:
            self._check_cross(data)

        # eval hold
        action = self.strategy.eval_hold()
        if action:
            # set invest amount
            if hasattr(self.strategy, "use_invest_ratio") and self.strategy.use_invest_ratio:
                spend_amount = self.balance * self.strategy.get_invest_ratio(self)
            elif self.unlimit:
                spend_amount = self.constant_invest_amount
            else:
                spend_amount = self.balance * self.invest_ratio
            # action
            if action == Status.BUY:
                self._buy(data, spend_amount=spend_amount)
            elif action == Status.SELL:
                self._sell(data, spend_amount=spend_amount)
        return action

