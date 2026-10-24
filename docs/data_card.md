# Data card: simulated fleet sales

**Everything is simulated.** Company names are generated (e.g. "Atlas Freight 0012"); no real company or person.

## What the simulator does
- 2,000 fleet customers in 8 EU countries and 5 industries (long-haul, regional distribution, construction, urban
  delivery, waste & municipal), each with a fleet of trucks by age, financing share, service contract cycle, our share
  of the fleet and a hidden "propensity" no model can see.
- Every month: trucks age, service cost and breakdowns rise with age, financing and contracts end, freight demand
  moves, reps visit and quote. A hidden rule turns this into the chance of an order; a second rule decides whether we
  win it or a competitor does. Orders replace old trucks and add growth.
- 36 months of history after a 36-month warm-up, so fleets are in a steady state.

## Tables
| Table | Rows | Content |
|---|---|---|
| customers | 2,000 | pseudonymised ID, country, industry, dealer, customer since, segment |
| snapshot | 2,000 | latest month's fleet, service, contract and sales facts |
| scores | 2,000 | score, rank, top 3 reasons, model version |
| deals | ~3,800 | won and lost orders, trucks, value, main trigger |
| segments | 5 | segment profiles |

## Label
`y = 1` if the customer places an order with us in the snapshot month or the 5 months after. The last 5 months have
no complete window and are not used for training or testing.

## Privacy
Customer IDs are keyed pseudonyms (HMAC-SHA256). Company names are kept apart and never reach the copilot, Claude or
the logs.
