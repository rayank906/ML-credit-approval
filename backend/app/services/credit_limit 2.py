def recommend_credit_limit(blended_risk, income, credit_score):
    base = income * 0.3

    if blended_risk < 0.10:
        multiplier = 2.0
    elif blended_risk < 0.20:
        multiplier = 1.5
    elif blended_risk < 0.30:
        multiplier = 1.2
    elif blended_risk < 0.40:
        multiplier = 1.0
    elif blended_risk < 0.55:
        multiplier = 0.5
    elif blended_risk < 0.70:
        multiplier = 0.3
    else:
        multiplier = 0.15

    credit_bonus = (credit_score - 300) / 550
    multiplier *= (0.7 + 0.6 * credit_bonus)

    limit = base * multiplier
    limit = max(500, min(limit, 100000))
    return round(limit, -2)
