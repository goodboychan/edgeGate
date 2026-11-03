| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.731 | 3.18 | 0 | 3.2 | 38.7 | 323 |  |
| clean | trt | HOLD | 0.731 | 3.18 | 0 | 3.2 | 19.4 | 401 | event_recall, false_alerts_per_h |
| clean | hailo8 | HOLD | 0.731 | 2.12 | 0 | 3.2 | 15.4 | 233 | event_recall, false_alerts_per_h |
| bgr | trt | HOLD | 0.682 | 1.88 | 1 | 4.53 | 20.6 | 419 | eval_set, stale |
| bgr | hailo8 | HOLD | 0.636 | 1.13 | 0 | 3 | 16.8 | 229 | eval_set, stale |
| no_norm | trt | HOLD | 0 | 0 | 0 | - | 44.7 | 407 | eval_set, stale |
| no_norm | hailo8 | HOLD | 0 | 0 | 0 | - | 40.2 | 249 | eval_set, stale |
| w4 | hailo8 | HOLD | 0.409 | 0 | 0 | 3.07 | 15 | 232 | eval_set, stale |
| calib10 | hailo8 | HOLD | 0.682 | 2.25 | 0 | 3.13 | 15.1 | 236 | eval_set, stale |
