# Baseline-corrected Attack Attribution

baseline_noattack (same M\*, 15 rounds, sr0.2, 0 malicious): acc 63.48 -> 68.42 (drift +4.94)

## per-class natural recall drift (baseline, biggest 15 drops)

| class | name | base | round14 | drift |
|---|---|---|---|---|
| 0 | ApplyEyeMakeup | 77.3 | 4.5 | -72.7 |
| 11 | BoxingPunchingBag | 89.8 | 38.8 | -51.0 |
| 15 | CricketBowling | 75.0 | 27.8 | -47.2 |
| 4 | BalanceBeam | 80.7 | 38.7 | -41.9 |
| 49 | WallPushups | 68.6 | 31.4 | -37.1 |
| 22 | Haircut | 39.4 | 3.0 | -36.4 |
| 2 | Archery | 65.8 | 34.1 | -31.7 |
| 47 | Typing | 95.3 | 76.7 | -18.6 |
| 40 | Shotput | 82.6 | 69.6 | -13.0 |
| 13 | BrushingTeeth | 36.1 | 25.0 | -11.1 |
| 46 | TableTennisShot | 97.4 | 87.2 | -10.3 |
| 3 | BabyCrawling | 74.3 | 65.7 | -8.6 |
| 25 | HandstandPushups | 64.3 | 57.1 | -7.2 |
| 38 | Rafting | 89.3 | 82.1 | -7.2 |
| 41 | SkyDiving | 96.8 | 90.3 | -6.5 |

## correlation: cosine vs attack effectiveness (OLD vs NET)

- OLD cosine vs target破坏: **-0.326**
- NET cosine vs target破坏 (baseline-corrected): **-0.234**
- OLD cosine vs acc下降: 0.131
- NET cosine vs acc下降: 0.131

## per-target avg Δt_recall (OLD vs NET)

| target | name | drift | old Δt | NET Δt |
|---|---|---|---|---|
| 2 | Archery | -31.7 | -21.9 | +9.8 |
| 9 | BodyWeightSquats | +26.7 | +1.1 | -25.6 |
| 13 | BrushingTeeth | -11.1 | -24.4 | -13.3 |
| 23 | HammerThrow | +4.4 | +2.2 | -2.2 |
| 26 | HandstandWalking | +5.9 | +9.8 | +3.9 |
| 28 | IceDancing | +30.4 | +30.4 | +0.0 |
| 30 | LongJump | +7.7 | +30.8 | +23.1 |
| 31 | MoppingFloor | +35.3 | -2.9 | -38.2 |
| 35 | PlayingDhol | +10.2 | +2.5 | -7.7 |
| 39 | ShavingBeard | +2.3 | +11.6 | +9.3 |
| 40 | Shotput | -13.0 | -61.6 | -48.6 |