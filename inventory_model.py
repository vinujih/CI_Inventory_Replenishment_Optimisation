import numpy as np
import pandas as pd

# ---- Data ----
scen = np.load("scenarios.npy")            # (1000, 50, 28)
actual = np.load("actual_demand.npy")      # (50, 28)
forecast = np.load("forecast_demand.npy")  # (50, 28)
pairs = pd.read_csv("pairs.csv")

N, D, W = 50, 28, 4                        # pairs, days, weeks
train_scen = scen[:500]                    # GA optimises on these
val_scen = scen[500:]                      # GA is judged on these

# ---- Assumptions ----
PRICE, COST, HOLD = 18.0, 10.0, 0.1
BUDGET_FACTOR = 0.95
CAP_FACTOR = 1.15
PENALTY_BUDGET = 5.0
PENALTY_CAP = 50.0

weekly_fc = forecast.reshape(N, W, 7).sum(axis=2)        # (50, 4)
BUDGET = BUDGET_FACTOR * COST * weekly_fc.sum()

stores = sorted(pairs["store"].unique())
M = np.array([(pairs["store"] == s).values.astype(float) for s in stores])  # (5, 50)
CAP = CAP_FACTOR * (M @ weekly_fc).max(axis=1)           # (5,) per-store weekly limit

GENE_HIGH = (1.5 * weekly_fc).flatten()                  # upper bound for each of 200 genes


def simulate_profit(plan, demand):
    """plan: (50, 4) weekly orders. demand: (S, 50, 28) or (50, 28).
    Returns the profit for each scenario."""
    if demand.ndim == 2:
        demand = demand[None, :, :]
    S = demand.shape[0]
    stock = np.zeros((S, N))
    sold_total = np.zeros(S)
    hold_total = np.zeros(S)
    for d in range(D):
        if d % 7 == 0:
            stock = stock + plan[:, d // 7]      # new weekly delivery arrives
        sold = np.minimum(stock, demand[:, :, d])
        stock = stock - sold
        sold_total += sold.sum(axis=1)
        hold_total += stock.sum(axis=1)
    return PRICE * sold_total - COST * plan.sum() - HOLD * hold_total


def penalty(plan):
    over_budget = max(0.0, COST * plan.sum() - BUDGET)
    over_cap = np.maximum(0.0, (M @ plan) - CAP[:, None]).sum()
    return PENALTY_BUDGET * over_budget + PENALTY_CAP * over_cap


def fitness(plan, demand=train_scen):
    return simulate_profit(plan, demand).mean() - penalty(plan)