# 55 组特征碰撞攻击实验完整数据表

每行一组实验。折线图列点开见该组的 source/target recall 与 global acc 随攻击轮次变化。
`base` = M\*（攻击前），`4/9/14` = 攻击轮次。

| method | source | target | cosine | s2t base→after | Δs2t | Δt_recall | Δs_recall | Δacc | inject | 折线图 |
|---|---|---|---|---|---|---|---|---|---|---|
| m1 | 9 BodyWeightSquats | 26 HandstandWalking | 0.869 | 0.0→0.0 | +0.0 | +11.8 | +10.0 | +3.1 | 29 | [📈](line_plots/line_s9_t26_m1.png) |
| m1 | 13 BrushingTeeth | 39 ShavingBeard | 0.798 | 22.2→44.4 | +22.2 | +11.6 | -16.7 | +6.9 | 29 | [📈](line_plots/line_s13_t39_m1.png) |
| m1 | 26 HandstandWalking | 23 HammerThrow | 0.779 | 0.0→0.0 | +0.0 | +6.7 | +8.8 | +2.9 | 35 | [📈](line_plots/line_s26_t23_m1.png) |
| m1 | 35 PlayingDhol | 31 MoppingFloor | 0.799 | 0.0→0.0 | +0.0 | -2.9 | +30.6 | +5.8 | 23 | [📈](line_plots/line_s35_t31_m1.png) |
| m1 | 36 PlayingFlute | 26 HandstandWalking | 0.762 | 0.0→0.0 | +0.0 | +0.0 | +14.6 | +3.2 | 29 | [📈](line_plots/line_s36_t26_m1.png) |
| m1 | 37 PlayingSitar | 2 Archery | 0.759 | 0.0→0.0 | +0.0 | -21.9 | +0.0 | +3.9 | 43 | [📈](line_plots/line_s37_t2_m1.png) |
| m1 | 38 Rafting | 26 HandstandWalking | 0.800 | 0.0→0.0 | +0.0 | +5.9 | -3.6 | +3.5 | 29 | [📈](line_plots/line_s38_t26_m1.png) |
| m1 | 40 Shotput | 23 HammerThrow | 0.876 | 4.3→6.5 | +2.2 | -2.2 | +4.3 | +4.0 | 35 | [📈](line_plots/line_s40_t23_m1.png) |
| m1 | 41 SkyDiving | 28 IceDancing | 0.822 | 0.0→0.0 | +0.0 | +30.4 | -3.2 | +3.1 | 51 | [📈](line_plots/line_s41_t28_m1.png) |
| m1 | 46 TableTennisShot | 30 LongJump | 0.844 | 0.0→0.0 | +0.0 | +30.8 | -2.6 | +4.9 | 38 | [📈](line_plots/line_s46_t30_m1.png) |
| m2 | 9 BodyWeightSquats | 13 BrushingTeeth | 0.540 | 0.0→0.0 | +0.0 | -25.0 | -3.3 | +2.5 | 35 | [📈](line_plots/line_s9_t13_m2.png) |
| m2 | 9 BodyWeightSquats | 26 HandstandWalking | 0.869 | 0.0→0.0 | +0.0 | +11.8 | +10.0 | +3.1 | 29 | [📈](line_plots/line_s9_t26_m2.png) |
| m2 | 9 BodyWeightSquats | 35 PlayingDhol | 0.598 | 3.3→0.0 | -3.3 | +24.5 | -3.3 | +5.2 | 24 | [📈](line_plots/line_s9_t35_m2.png) |
| m2 | 9 BodyWeightSquats | 40 Shotput | 0.733 | 13.3→0.0 | -13.3 | -80.4 | -10.0 | +1.8 | 36 | [📈](line_plots/line_s9_t40_m2.png) |
| m2 | 13 BrushingTeeth | 9 BodyWeightSquats | 0.724 | 0.0→0.0 | +0.0 | -3.3 | -30.6 | +3.5 | 13 | [📈](line_plots/line_s13_t9_m2.png) |
| m2 | 13 BrushingTeeth | 26 HandstandWalking | 0.750 | 0.0→0.0 | +0.0 | +11.8 | -25.0 | +5.9 | 29 | [📈](line_plots/line_s13_t26_m2.png) |
| m2 | 13 BrushingTeeth | 35 PlayingDhol | 0.527 | 0.0→0.0 | +0.0 | -4.1 | -36.1 | +4.5 | 24 | [📈](line_plots/line_s13_t35_m2.png) |
| m2 | 13 BrushingTeeth | 40 Shotput | 0.647 | 0.0→0.0 | +0.0 | -60.9 | -27.8 | +6.5 | 36 | [📈](line_plots/line_s13_t40_m2.png) |
| m2 | 26 HandstandWalking | 9 BodyWeightSquats | 0.667 | 8.8→5.9 | -2.9 | -6.7 | +11.8 | +6.4 | 13 | [📈](line_plots/line_s26_t9_m2.png) |
| m2 | 26 HandstandWalking | 13 BrushingTeeth | 0.489 | 0.0→0.0 | +0.0 | -16.7 | +5.9 | +5.6 | 35 | [📈](line_plots/line_s26_t13_m2.png) |
| m2 | 26 HandstandWalking | 35 PlayingDhol | 0.543 | 0.0→0.0 | +0.0 | +12.2 | +11.8 | +5.6 | 24 | [📈](line_plots/line_s26_t35_m2.png) |
| m2 | 26 HandstandWalking | 40 Shotput | 0.673 | 23.5→0.0 | -23.5 | -63.0 | +11.8 | +5.0 | 36 | [📈](line_plots/line_s26_t40_m2.png) |
| m2 | 35 PlayingDhol | 9 BodyWeightSquats | 0.669 | 0.0→0.0 | +0.0 | -3.3 | +2.0 | +3.3 | 13 | [📈](line_plots/line_s35_t9_m2.png) |
| m2 | 35 PlayingDhol | 13 BrushingTeeth | 0.483 | 0.0→0.0 | +0.0 | -19.4 | +28.6 | +4.6 | 35 | [📈](line_plots/line_s35_t13_m2.png) |
| m2 | 35 PlayingDhol | 26 HandstandWalking | 0.695 | 0.0→0.0 | +0.0 | +14.7 | +26.5 | +6.6 | 29 | [📈](line_plots/line_s35_t26_m2.png) |
| m2 | 35 PlayingDhol | 40 Shotput | 0.662 | 0.0→0.0 | +0.0 | -52.2 | +18.4 | +4.9 | 36 | [📈](line_plots/line_s35_t40_m2.png) |
| m2 | 36 PlayingFlute | 9 BodyWeightSquats | 0.684 | 0.0→0.0 | +0.0 | +0.0 | +6.2 | +6.8 | 13 | [📈](line_plots/line_s36_t9_m2.png) |
| m2 | 36 PlayingFlute | 13 BrushingTeeth | 0.686 | 2.1→0.0 | -2.1 | -16.7 | +4.2 | +4.6 | 35 | [📈](line_plots/line_s36_t13_m2.png) |
| m2 | 36 PlayingFlute | 26 HandstandWalking | 0.762 | 0.0→0.0 | +0.0 | +0.0 | +14.6 | +3.2 | 29 | [📈](line_plots/line_s36_t26_m2.png) |
| m2 | 36 PlayingFlute | 35 PlayingDhol | 0.658 | 0.0→0.0 | +0.0 | -8.2 | +12.5 | +5.0 | 24 | [📈](line_plots/line_s36_t35_m2.png) |
| m2 | 36 PlayingFlute | 40 Shotput | 0.677 | 0.0→0.0 | +0.0 | -69.6 | +16.7 | +3.0 | 36 | [📈](line_plots/line_s36_t40_m2.png) |
| m2 | 37 PlayingSitar | 9 BodyWeightSquats | 0.663 | 0.0→0.0 | +0.0 | +0.0 | +0.0 | +4.3 | 13 | [📈](line_plots/line_s37_t9_m2.png) |
| m2 | 37 PlayingSitar | 13 BrushingTeeth | 0.466 | 0.0→0.0 | +0.0 | -30.6 | +0.0 | +6.1 | 35 | [📈](line_plots/line_s37_t13_m2.png) |
| m2 | 37 PlayingSitar | 26 HandstandWalking | 0.639 | 0.0→0.0 | +0.0 | +14.7 | +0.0 | +4.7 | 29 | [📈](line_plots/line_s37_t26_m2.png) |
| m2 | 37 PlayingSitar | 35 PlayingDhol | 0.748 | 2.3→0.0 | -2.3 | +4.1 | +0.0 | +5.3 | 24 | [📈](line_plots/line_s37_t35_m2.png) |
| m2 | 37 PlayingSitar | 40 Shotput | 0.606 | 0.0→0.0 | +0.0 | -67.4 | +0.0 | +3.9 | 36 | [📈](line_plots/line_s37_t40_m2.png) |
| m2 | 38 Rafting | 9 BodyWeightSquats | 0.629 | 0.0→0.0 | +0.0 | +16.7 | -7.2 | +3.8 | 13 | [📈](line_plots/line_s38_t9_m2.png) |
| m2 | 38 Rafting | 13 BrushingTeeth | 0.511 | 0.0→0.0 | +0.0 | -22.2 | -3.6 | +4.9 | 35 | [📈](line_plots/line_s38_t13_m2.png) |
| m2 | 38 Rafting | 26 HandstandWalking | 0.800 | 0.0→0.0 | +0.0 | +5.9 | -3.6 | +3.5 | 29 | [📈](line_plots/line_s38_t26_m2.png) |
| m2 | 38 Rafting | 35 PlayingDhol | 0.684 | 0.0→0.0 | +0.0 | +0.0 | -3.6 | +3.4 | 24 | [📈](line_plots/line_s38_t35_m2.png) |
| m2 | 38 Rafting | 40 Shotput | 0.754 | 0.0→0.0 | +0.0 | -56.5 | -7.2 | +2.6 | 36 | [📈](line_plots/line_s38_t40_m2.png) |
| m2 | 40 Shotput | 9 BodyWeightSquats | 0.722 | 0.0→0.0 | +0.0 | +0.0 | +6.5 | +5.4 | 13 | [📈](line_plots/line_s40_t9_m2.png) |
| m2 | 40 Shotput | 13 BrushingTeeth | 0.483 | 0.0→0.0 | +0.0 | -27.8 | -50.0 | +4.3 | 35 | [📈](line_plots/line_s40_t13_m2.png) |
| m2 | 40 Shotput | 26 HandstandWalking | 0.847 | 0.0→0.0 | +0.0 | +11.8 | +4.3 | +7.1 | 29 | [📈](line_plots/line_s40_t26_m2.png) |
| m2 | 40 Shotput | 35 PlayingDhol | 0.588 | 0.0→0.0 | +0.0 | -14.3 | -15.2 | +3.7 | 24 | [📈](line_plots/line_s40_t35_m2.png) |
| m2 | 41 SkyDiving | 9 BodyWeightSquats | 0.633 | 0.0→0.0 | +0.0 | -10.0 | -3.2 | +4.7 | 13 | [📈](line_plots/line_s41_t9_m2.png) |
| m2 | 41 SkyDiving | 13 BrushingTeeth | 0.454 | 0.0→0.0 | +0.0 | -33.3 | -9.7 | +7.4 | 35 | [📈](line_plots/line_s41_t13_m2.png) |
| m2 | 41 SkyDiving | 26 HandstandWalking | 0.803 | 0.0→3.2 | +3.2 | +26.5 | +0.0 | +6.4 | 29 | [📈](line_plots/line_s41_t26_m2.png) |
| m2 | 41 SkyDiving | 35 PlayingDhol | 0.575 | 0.0→0.0 | +0.0 | +12.2 | -3.2 | +2.3 | 24 | [📈](line_plots/line_s41_t35_m2.png) |
| m2 | 41 SkyDiving | 40 Shotput | 0.696 | 0.0→0.0 | +0.0 | -76.1 | +3.2 | +2.8 | 36 | [📈](line_plots/line_s41_t40_m2.png) |
| m2 | 46 TableTennisShot | 9 BodyWeightSquats | 0.751 | 0.0→0.0 | +0.0 | +16.7 | -5.1 | +2.6 | 13 | [📈](line_plots/line_s46_t9_m2.png) |
| m2 | 46 TableTennisShot | 13 BrushingTeeth | 0.524 | 0.0→0.0 | +0.0 | -27.8 | +2.6 | +4.4 | 35 | [📈](line_plots/line_s46_t13_m2.png) |
| m2 | 46 TableTennisShot | 26 HandstandWalking | 0.778 | 0.0→0.0 | +0.0 | +2.9 | -7.7 | +5.5 | 29 | [📈](line_plots/line_s46_t26_m2.png) |
| m2 | 46 TableTennisShot | 35 PlayingDhol | 0.696 | 0.0→0.0 | +0.0 | -4.1 | +0.0 | +3.7 | 24 | [📈](line_plots/line_s46_t35_m2.png) |
| m2 | 46 TableTennisShot | 40 Shotput | 0.784 | 0.0→0.0 | +0.0 | -28.3 | -2.6 | +6.6 | 36 | [📈](line_plots/line_s46_t40_m2.png) |