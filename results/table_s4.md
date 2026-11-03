| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.846 | 4.95 | 0 | 3.8 | 93.9 | 415 |  |
| clean | trt | HOLD | 0.846 | 4.95 | 0 | 3.8 | 22.9 | 438 | event_recall, false_alerts_per_h |
| clean | hailo8 | HOLD | 0.885 | 4.95 | 0 | 3.33 | 20.9 | 252 | event_recall, false_alerts_per_h |
| bgr | trt | HOLD | 0.636 | 4.5 | 1 | 3.33 | 23.3 | 445 | eval_set |
| bgr | hailo8 | HOLD | 0.727 | 2.63 | 2 | 3.47 | 20.2 | 252 | eval_set |
| no_norm | trt | HOLD | 0 | 0 | 0 | - | 53.6 | 441 | eval_set |
| no_norm | hailo8 | HOLD | 0.318 | 3.38 | 0 | 2.53 | 30.6 | 255 | eval_set |
| w4 | hailo8 | HOLD | 0.273 | 0.375 | 0 | 3.4 | 20.9 | 248 | eval_set |
| calib10 | hailo8 | HOLD | 0.864 | 3.38 | 2 | 3.13 | 20.7 | 252 | eval_set |
