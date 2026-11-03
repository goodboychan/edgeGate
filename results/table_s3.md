| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.962 | 4.24 | 0 | 3.33 | 65.5 | 335 |  |
| clean | trt | HOLD | 0.962 | 4.24 | 0 | 3.33 | 19.5 | 438 | false_alerts_per_h |
| clean | hailo8 | HOLD | 0.962 | 4.59 | 1 | 3 | 15.9 | 238 | false_alerts_per_h, duplicate_alerts |
| bgr | trt | HOLD | 0.769 | 3.53 | 1 | 3.2 | 20.4 | 400 | event_recall, false_alerts_per_h, duplicate_alerts |
| bgr | hailo8 | HOLD | 0.808 | 3.89 | 1 | 3.4 | 17.1 | 230 | event_recall, false_alerts_per_h, duplicate_alerts |
| no_norm | trt | HOLD | 0 | 0 | 0 | - | 44.8 | 407 | eval_set |
| no_norm | hailo8 | HOLD | 0 | 0 | 0 | - | 39.2 | 252 | eval_set |
| w4 | hailo8 | HOLD | 0.636 | 3.75 | 4 | 6.8 | 14.4 | 230 | eval_set |
| calib10 | hailo8 | HOLD | 0.909 | 4.88 | 1 | 3.13 | 16.8 | 233 | eval_set |
