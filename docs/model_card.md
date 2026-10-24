# Model card: purchase likelihood

| | |
|---|---|
| **Name** | `fleetpilot-purchase` (MLflow registry), version shown in every score as `model_version` |
| **Task** | Probability that a fleet customer orders new trucks **from us** in the next 6 months |
| **Algorithm** | XGBoost classifier (500 trees, depth 4), Platt-calibrated on later months |
| **Explanations** | SHAP values per customer (exact TreeSHAP from XGBoost), top 3 shown in plain words |
| **Data** | Simulated: 2,000 customers x 36 monthly snapshots (see [data card](data_card.md)) |
| **Owner** | Saeed Hojjatpanah |

## Intended use
Ranking customers for sales reps to contact, with reasons. A rep decides; the model does not contact anyone and
does not set prices or credit terms.

## Not intended for
Decisions about individuals (customers are companies), credit, pricing or anything legally significant. Not validated
on real data.

## Features
Fleet size and age, share of trucks over 6 years, financing and service contracts ending soon, km per truck, service
cost and its trend, breakdowns, sales visits and quotes, time since the last order, freight demand, fleet growth,
telematics, years as a customer, industry, country.

## Evaluation (held-out later months, never used for training)
See the app's **Model report** tab or `data/metrics.json` after a build. Compared against a logistic regression
and the common rule of thumb "call the oldest fleets first", on AUC, PR-AUC, share of buyers found in the top
10% / 20%, and calibration.

## Known limits
- Simulated data: the numbers show the method, not real-world performance.
- Quotes and visits are strong signals but partly follow the rep's own judgement (a feedback loop in real use).
- Calibration drifts when market conditions move; monitor predicted vs actual monthly and recalibrate.

## Monitoring (production plan)
Monthly: predicted vs actual orders per segment and country, feature drift (fleet age, freight index), share of
customers with score above 50%. Retrain when the top-10% capture drops more than 5 points.
