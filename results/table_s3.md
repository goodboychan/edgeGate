| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.962 | 4.24 | 0 | 3.33 | 65.5 | 335 |  |
| clean | trt | HOLD | 0.962 | 4.24 | 0 | 3.33 | 19.5 | 438 | false_alerts_per_h |
| clean | hailo8 | HOLD | 0.962 | 4.59 | 1 | 3 | 15.9 | 238 | false_alerts_per_h, duplicate_alerts |
| bgr | trt | HOLD | 0.769 | 3.53 | 1 | 3.2 | 20.4 | 400 | event_recall, false_alerts_per_h, duplicate_alerts |
| bgr | hailo8 | HOLD | 0.808 | 3.89 | 1 | 3.4 | 17.1 | 230 | event_recall, false_alerts_per_h, duplicate_alerts |
| no_norm | trt | HOLD | 0 | 0 | 0 | - | 44.4 | 406 | event_recall, detection_delay_s_p95 |
| no_norm | hailo8 | HOLD | 0 | 0 | 0 | - | 42 | 251 | event_recall, detection_delay_s_p95 |
| w4 | hailo8 | HOLD | 0.615 | 4.59 | 4 | 6.8 | 14.9 | 228 | event_recall, false_alerts_per_h, duplicate_alerts, detection_delay_s_p95 |
| calib10 | hailo8 | HOLD | 0.923 | 5.3 | 1 | 3.13 | 15 | 232 | false_alerts_per_h, duplicate_alerts |
