# Trading Bot
## Introduction
Using Binance API to simulate transaction strategies.  
Functions include:  
* market evaluation 
* hold & buy & sell evaluations   
  
![](https://media.giphy.com/media/ioopmOHLqIDfGxLLKG/giphy.gif)
## Requirements
* Python 
* binance-futures-connector
* Pandas
* Numpy
* talib
## Reference 
[Binance API doc](https://binance-docs.github.io/apidocs/futures/en/#general-info)
## Installation 
```python
pip install binance-futures-connector
```
## Content
### **Intro**
First, go to [Binance Future testnet](https://testnet.binancefuture.com/en/futures/BTCUSDT) to register an API key to access service provide by Binance.  
Note that this key and the environment is only for testing, no need to worry about any losses or mistake you make (or money you earn) with this API key.
> NOTE: The price and trade are different from the real Binance Future.  
  
**Secretly** store your API key and Secret Key in .env file.  
> NOTE: In .env file 
> FUTURE_VIRTUAL_KEY = *******************  
> FUTURE_VIRTUAL_SECRET = **************** 
  
Now we can start!

### **Setting up**
First we import required packages  
```python
from binance.um_futures import UMFutures
import numpy as np
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
import time
from enum import Enum
```
Start a new connection  
```python
load_dotenv()
#create client connection
client = UMFutures(base_url="https://testnet.binancefuture.com",
                    key=os.getenv("FUTURE_VIRTUAL_KEY"),
                    secret=os.getenv("FUTURE_VIRTUAL_SECRET"))
```
With **client**, we can fetch data and make transactions

### **Fetch market data**
![](https://media.giphy.com/media/S0kRodfOj6kvWrtRwM/giphy.gif)   
Here we fetch k-lines  
> NOTE: Since Binance API has call-limit, we can only fetch 500 data points in one call
  Define a function to get data we needed with respect to:
- Start time ("2021-01-01")
- End time ("2021-01-01")
- Sep (Interval: "1h", "1m")
- crypto (Symbol: "BTCUSDT")

```python
def fetch_data(
    client,
    crypto: str,
    sep: str,
    start_time: str = None,
    end_time: str = None,
    start_time_int: int = None,
    end_time_int: int = None,
):
    # format:
    #
    # [  
    #   [  
    # 0    1499040000000,      // Open time   /1000 to get real time
    # 1    "0.01634790",       // Open  
    # 2    "0.80000000",       // High  
    # 3    "0.01575800",       // Low  
    # 4    "0.01577100",       // Close  
    # 5    "148976.11427815",  // Volume  
    # 6    1499644799999,      // Close time  
    # 7    "2434.19055334",    // Quote asset volume  
    # 8    308,                // Number of trades  
    # 9    "1756.87402397",    // Taker buy base asset volume  
    # 10    "28.46694368",      // Taker buy quote asset volume  
    #     "17928899.62484339" // Ignore.  
    #   ]  
    # ]
    
    data_list = []
    #fetch with string date
    if start_time and end_time:
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d")
    #fetch with timestamp
    elif start_time_int and end_time_int:
        start = datetime.datetime.fromtimestamp(start_time_int)
        end = datetime.datetime.fromtimestamp(end_time_int)
    cur_start = start
    # Data sep by hours
    if sep[-1] == "h":
        limit = 500 // (24 // int(sep[:-1]))
        while cur_start + datetime.timedelta(days=limit) <= end:
            cur_end = cur_start + datetime.timedelta(days=limit)
            cur_start_stamp = int(datetime.datetime.timestamp(cur_start) * 1000)
            cur_end_stamp = int(datetime.datetime.timestamp(cur_end) * 1000 - 1)
            data = client.klines(crypto, sep, startTime=cur_start_stamp, endTime=cur_end_stamp)
            data_list += data
            cur_start = cur_end
            time.sleep(0.1) #prevent limit error
    # Data sep by minutes
    elif sep[-1] == "m":
        limit = 500 // (60 // int(sep[:-1]))
        while cur_start + datetime.timedelta(hours=limit) <= end:
            cur_end = cur_start + datetime.timedelta(hours=limit)
            cur_start_stamp = int(datetime.datetime.timestamp(cur_start) * 1000)
            cur_end_stamp = int(datetime.datetime.timestamp(cur_end) * 1000 - 1)
            data = client.klines(crypto, sep, startTime=cur_start_stamp, endTime=cur_end_stamp)
            data_list += data
            cur_start = cur_end
            time.sleep(0.6) #prevent limit error
    # Pick up rest data
    if cur_start < end:
        cur_start_stamp = int(datetime.datetime.timestamp(cur_start) * 1000)
        cur_end_stamp = int(datetime.datetime.timestamp(end) * 1000 - 1)
        data = client.klines(crypto, sep, startTime=cur_start_stamp, endTime=cur_end_stamp)
        data_list += data
    # log
    print(
        "---------DATA INFO---------\n",
        f"Fetch Data: {crypto}\n",
        f"Data Start from: {datetime.datetime.fromtimestamp(data_list[0][0]/1000)}\n",
        f"Data End at: {datetime.datetime.fromtimestamp(data_list[-1][6]/1000)}\n",
        f"Sep. Interval: {sep}",
    )

    return data_list
```

### **Market Simulation**
![](https://media.giphy.com/media/y31rRE5h3wyPXey8vx/giphy.gif)   
Now we create a object that will loop through all the data point and store corresponding info   
The market data will be fed to bot.
Parameters:
- market_data (data fectched by function above)
- bot (we will cover later)
- info_out (path storing info)
```python
class market_sim:
    def __init__(self, market_data: list, bot, info_out, time_sep):
        self.market_data = market_data
        self.bot = bot
        self.time_sep = time_sep
        self.cur = 0
        self.market_history = pd.DataFrame(
            columns=[
                "open time",
                "close time",
                "open price",
                "close price",
                "high",
                "low",
                "color",
                "action",
            ]
        )
        self.info_out = info_out

    def _write_info(
        self,
    ):
        with open(os.path.join(self.info_out, "info.txt"), "w") as f:
            f.write(
                "----------info----------\n"
                + f"Crypto: {self.bot.crypto}\n"
                + f"Simulation time: {datetime.datetime.fromtimestamp(self.market_data[0][0]/1000)} - {datetime.datetime.fromtimestamp(self.market_data[-1][6]/1000)}\n"
                + f"Time Interval: {self.time_sep}\n"
                + f"Strategy: {self.bot.strategy.name}\n"
                + f"Start Balance: {self.bot.balance}\n"
                + f"Leverage: {self.bot.leverage}\n"
                + f"Invest Ratio: {self.bot.invest_ratio}\n"
                + f"TRX FEE Ratio: {self.bot.TRX_FEE_ratio}\n"
                + f"Stop Loss Ratio: {self.bot.stop_loss_ratio}\n"
                + f"Stop Profit Ratio: {self.bot.stop_profit_ratio}\n"
            )

    def start(self):
        # loop through data point
        self._write_info()
        for data in self.market_data:
            action = self.bot.eval(data)  # get action
            if action == Status.BUY:
                self._fill_df(data, "BUY")
            elif action == Status.SELL:
                self._fill_df(data, "SELL")
            else:
                self._fill_df(data)

    def _fill_df(self, data, action=None):
        # Fill info into dataframe
        self.market_history.loc[len(self.market_history)] = [
            datetime.datetime.fromtimestamp(data[0] / 1000),
            datetime.datetime.fromtimestamp(data[6] / 1000),
            data[1],
            data[4],
            data[2],
            data[3],
            "RED" if float(data[4]) <= float(data[1]) else "GREEN",
            action,
        ]
```
### **Enums**
Here we create some useful enum objects  
```python
class Status(Enum):
    OBSERVING = 0
    SELL = 1
    BUY = 2
    HOLDING_BUY = 3
    HOLDING_SELL = 4

class Stop_status(Enum):
    LOSS = 0
    PROFIT = 1

class market_status(Enum):
    SHORT = 0
    LONG = 1
    NONE = 2
```

### **Strategy**
![](https://media.giphy.com/media/gEvab1ilmJjA82FaSV/giphy.gif)   
Strategy is the core of a trading bot  
It will tell the bot when to place or liquidate an order  
  
There are three main function that will be called by the bot:  
- eval_market  (return None)  (this function will eval the market and update the status based on strategy)
- eval_hold    (return None or Status(Enum)) (Status.BUY to place long order, Status.SELL to place short order)
- eval_cross   (return Bool) (True to cross the order else Flase)
- change_status (return None) (Change the current status to OBSERVEING when the order is crossed by TP/SL)
  
#### Here we take an example  
Assume that we want to place an order every n klines with same color.  
That is:
- Place a short order after n Red klines
- Place a long order after n Green klines
  
Then we immediately cross the order when the n+1th kline closed.  
For example, with 1 hour data interval, at 1:59PM, 2:59PM and 3:59PM the klines were all red, thus we place a **SHORT** order at 4:00PM,  
then we cross the order at 4:59PM.

Below is the code that fulfill the requirements  
Parameters:
- decision_time (oberve n klines before action)
- limit_hold_time (hold the order for how long)
```python
class interval_than_make_one:
    def __init__(self, decision_time, limit_hold_time):
        self.status = Status.OBSERVING
        self.decision_time=decision_time
        self.observed_time = 0
        self.limit_hold_time = limit_hold_time
        self.name = f"Observe_{decision_time}_and_hold_{limit_hold_time}"
    
    def eval_market(self, data):
        open_price = float(data[1])
        closed_price = float(data[4])
        if open_price >= closed_price and self.status == Status.SELL:
            self.observed_time += 1
        elif open_price >= closed_price and self.status != Status.SELL:
            self.status = Status.SELL
            self.observed_time = 1
        elif open_price < closed_price and self.status == Status.BUY:
            self.observed_time += 1
        elif open_price < closed_price and self.status != Status.BUY:
            self.status = Status.BUY
            self.observed_time = 1
    
    def eval_hold(self,):
        if self.observed_time >= self.decision_time and self.status != Status.OBSERVING:
            self.observed_time = 0
            return self.status
        else:
            return None
    
    def eval_cross(self, order):
        if order["hold_time"] >= self.limit_hold_time:
            return True
        else:
            return False
    def change_status(self, status: Status):
        self.status = status
        self.observed_time = 0
        return
        
```

### **Bot**
![](https://media.giphy.com/media/NHIecaiSc7YjK/giphy.gif)  
Now comes the bot  
  
After defining the rule(strategy) that the bot needs to follow, lets see how bot do the transaction.  
The bot will first evaluate the new market data, then check previous orders if needed to be cross, and then see whether to place a new order.  
For this example, we set our bot to evaluate the market every one hour.  
That is:
- Using 1 hour klines as data point (for example: 2:00 - 2:59)
- Bot will do those above actions every 1 hour (start at 3:00 then sleep for 1 hour)  
Since we simulate on history data there is no need to sleep.  

The Bot work in three step:
1. Evaluates the market data (strategy.eval_hold)
2. Check the status of existing orders (if needed to be crossed or already cross by TP/SL) (strategy.eval_cros)
3. New order or not (strategy.eval_hold)
  
When the bot call strategy.eval_hold and get the signal Status.BUY or Status.SELL, the bot will open an order and store info of that order.  
When there is an existing order and strategy.eval_cross return True, the order will be crossed and the profit will be calculated.   

Take profit and stop loss is important when trading.  
This bot can also simulate the TP/SL with resolution of 1 min.  
  
There is only one public function in bot that will be called by market_sim
- eval() return Status

** Code in repo **

## Main
![](https://media.giphy.com/media/ViBN1GDg1MdgKKz9gj/giphy.gif)   
To simulate and test the strategy, we need three components.  
- market_sim
- TradingBot_v3_sim
- strategy
  
First fetch the market data, then we define the strategy and finially we set up the trading bot.
```python

def SIM(client: UMFutures, crypto, start, end, time_sep, bot_strategy, stop_loss = None, stop_profit = None, sim_profit_loss = False, sim_interval = "1m", market_from_csv=False, sim_from_csv=True):
    # Fetch market data
    if not market_from_csv:
        data_list = fetch_data(client, crypto, time_sep, start, end)
    else:
        data_list = fetch_data_from_csv("PATH/DATA/POINT", start, end)
        
    #set up bot
    strategy_name = bot_strategy.name
    bot = TradingBot_v3_sim(client, crypto, bot_strategy, leverage=1, balance=1000000,invest_ratio=0.95, stop_loss_ratio=stop_loss, sim_stop_profit_loss = sim_profit_loss, sim_stop_interval=sim_interval, stop_profit_ratio=stop_profit, simulate_from_csv=sim_from_csv)
    
    #set output pth
    OUTPUT_pth = os.path.join(f"sim output/{crypto}/", strategy_name)
    file_name_base = start + "_" + end + "_"
    OUTPUT_pth = os.path.join(OUTPUT_pth, file_name_base)
    if not os.path.exists(OUTPUT_pth):
        os.makedirs(OUTPUT_pth)

    #set market 
    market_1 = market_sim(data_list, bot, OUTPUT_pth, market_from_csv)
    market_1.start()

    # save report
    bot.order_history.to_csv(os.path.join(OUTPUT_pth,f"trade_history.csv"))
    market_1.market_history.to_csv(os.path.join(OUTPUT_pth,f"market_history.csv"))
    if hasattr(bot_strategy, "history"):
      bot_strategy.history.to_csv(os.path.join(OUTPUT_pth,f"strategy_history.csv"))
      
    return
```
```python
if __name__ == "main":
    true_client = UMFutures(key=os.getenv("TRUE_KEY"), secret=os.getenv("TRUE_SECRET"))
    bot_strategy = interval_than_make_one(3,1)
    SIM(
        true_client, 
        "BTCUSDT",
        "2020-01-01", 
        "2022-08-30", 
        "1h", 
        bot_strategy,
        sim_profit_loss=False
    )
```

## Output
The simulation will generate 4 files
- info.txt
- market_history.csv
- strategy_history.csv
- trade_history.csv
  
Info will store the information of this simulation.   
Market history contains the information of each kline.  
Strategy history will record every evalution your strategy done.  
Trade history contains all the transaction the trading bot done.  




  

















