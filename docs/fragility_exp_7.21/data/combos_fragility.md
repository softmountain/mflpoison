# Fragility-driven Target Selection (v2)

M* acc=74.94855967078189, recall>=70.0, source cap=90.0

## sources (TSTR top, real recall <= cap)

| source | name | TSTR | real recall |
|---|---|---|---|
| 38 |  | 100.0 | 89.3 |
| 8 |  | 97.0 | 84.8 |
| 42 |  | 80.5 | 82.9 |
| 21 |  | 75.7 | 83.8 |
| 20 |  | 70.3 | 78.4 |

## A: fragile target × farthest source (main)

| group | target | name | fragility | recall | source | cosine |
|---|---|---|---|---|---|---|
| A1 | 49 |  | +3.81 | 80.0 | 38  | -0.026 |
| A2 | 14 |  | +2.42 | 74.4 | 42  | -0.034 |
| A3 | 35 |  | +2.35 | 85.7 | 21  | -0.021 |
| A4 | 2 |  | +2.03 | 75.6 | 20  | 0.087 |
| A5 | 23 |  | +1.20 | 88.9 | 21  | 0.158 |

## B: fragile target × nearest source (cosine control vs A)

| group | target | source | cosine |
|---|---|---|---|
| B1 | 49  | 42  | 0.180 |
| B2 | 14  | 38  | 0.452 |
| B3 | 35  | 38  | 0.431 |
| B4 | 2  | 38  | 0.398 |
| B5 | 23  | 42  | 0.516 |

## C: robust target × farthest source (fragility control vs A)

| group | target | name | fragility | recall | source | cosine |
|---|---|---|---|---|---|---|
| C1 | 38 |  | -2.39 | 89.3 | 42  | 0.068 |
| C2 | 41 |  | -2.25 | 93.5 | 42  | 0.057 |
| C3 | 8 |  | -2.13 | 84.8 | 42  | -0.081 |
| C4 | 36 |  | -2.05 | 100.0 | 42  | 0.005 |
| C5 | 37 |  | -1.99 | 97.7 | 21  | 0.003 |