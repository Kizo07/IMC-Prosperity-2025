import pandas as pd
import numpy as np

# 4/11 between 160 and 200, 7/11 between 250 and 320
def turtle_cdf(x):
    if x < 160:
        return 0
    elif x <= 200:
        return (x - 160) / 40 * (4/11)
    elif x < 250:
        return 4/11
    elif x <= 320:
        return 4/11 + (x - 250) / 70 * (7/11)
    else:
        return 1

# calculate total profit
def profit(bid1, bid2, avg_bid=None):
    # assign avg bid to bid2 if none
    if avg_bid is None:
        avg_bid = float(bid2)
        
    if bid1 >= bid2 or avg_bid <= bid1:
        return -np.inf
    
    # probability of capturing sea turtle with given bid
    p1 = turtle_cdf(bid1)
    p2 = turtle_cdf(bid2)
    
    # decide to apply penalty or not
    if avg_bid > bid2:
        penalty = ((320 - float(avg_bid)) / (320 - float(bid2)))**3
    else:
        penalty = 1
        
    # profit from each bid
    profit1 = (320 - bid1) * p1
    profit2 = (320 - bid2) * (p2 - p1) * penalty
    
    return profit1 + profit2

# Valid bids for first bid: 160–319 excluding 201–249
valid_bids = [x for x in range(160, 320) if x <= 200 or x >= 250]

""" Without penalty """

rows = []
for i, bid1 in enumerate(valid_bids):
    for bid2 in valid_bids[i+1:]:
        p = profit(bid1, bid2)
        rows.append((bid1, bid2, p))

df_bids = pd.DataFrame(rows, columns=["bid1", "bid2", "profit"])
df_bids = df_bids.sort_values(by="profit", ascending=False).reset_index(drop=True)


""" With average bid penalty """

# Choose avg_bid resolution
avg_step = 1

# Build rows of bid1, bid2, avg_bid, profit
rows = []
for i, bid1 in enumerate(valid_bids):
    for bid2 in valid_bids[i+1:]:
        avg_start = bid1 + avg_step
        avg_end = 320 + avg_step / 2  # inclusive of 320
        avg_bids = np.arange(avg_start, avg_end, avg_step)
        for avg_bid in avg_bids:
            p = profit(bid1, bid2, avg_bid)
            rows.append((bid1, bid2, avg_bid, p))

df_avg_sim = pd.DataFrame(rows, columns=["bid1", "bid2", "avg_bid", "profit"])
df_avg_sim = df_avg_sim.sort_values(by="profit", ascending=False).reset_index(drop=True)

""" Viewing ideal bids under an expected average scenario """

# change this number as you see fit
expected_avg = 288
df_avg_scenario = df_avg_sim[df_avg_sim["avg_bid"] == expected_avg]
df_avg_sim = df_avg_sim.sort_values(by="profit", ascending=False).reset_index(drop=True)