def size_position(cash, price, risk_fraction=0.01):
    amount = cash * risk_fraction
    if price <= 0:
        return 0
    return amount / price
