| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.846 | 5.65 | 1 | 3.4 | 52.7 | 323 |  |
| clean | trt | HOLD | 0.885 | 5.65 | 1 | 3.4 | 19.4 | 408 | event_recall, false_alerts_per_h, duplicate_alerts |
| clean | hailo8 | HOLD | 0.846 | 7.06 | 1 | 4.27 | 15.5 | 234 | event_recall, false_alerts_per_h, duplicate_alerts |
| bgr | trt | HOLD | 0.955 | 2.63 | 0 | 5.07 | 20.3 | 409 | eval_set, stale |
| bgr | hailo8 | HOLD | 0.909 | 3.38 | 0 | 3.4 | 14.8 | 233 | eval_set, stale |
| no_norm | trt | HOLD | 0 | 1.13 | 0 | - | 44.6 | 412 | eval_set, stale |
| no_norm | hailo8 | HOLD | 0.0909 | 5.63 | 0 | 2.87 | 40.3 | 249 | eval_set, stale |
| w4 | hailo8 | HOLD | 0.5 | 1.5 | 0 | 10 | 15.1 | 228 | eval_set, stale |
| calib10 | hailo8 | HOLD | 0.864 | 4.5 | 1 | 9.73 | 16.2 | 246 | eval_set, stale |
