| candidate | backend | verdict | recall | false/h | dup | delay p95 s | latency p95 ms | mem MB | held for |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | onnx-gx10 | - | 0.846 | 4.95 | 0 | 3.8 | 93.9 | 415 |  |
| clean | trt | HOLD | 0.846 | 4.95 | 0 | 3.8 | 22.9 | 438 | event_recall, false_alerts_per_h |
| clean | hailo8 | HOLD | 0.885 | 4.95 | 0 | 3.33 | 20.9 | 252 | event_recall, false_alerts_per_h |
| bgr | trt | HOLD | 0.692 | 5.3 | 1 | 3.33 | 23.3 | 457 | event_recall, false_alerts_per_h, duplicate_alerts |
| bgr | hailo8 | HOLD | 0.769 | 3.18 | 2 | 3.33 | 20.4 | 251 | event_recall, false_alerts_per_h, duplicate_alerts |
| no_norm | trt | HOLD | 0 | 0 | 0 | - | 53.6 | 440 | event_recall, detection_delay_s_p95 |
| no_norm | hailo8 | HOLD | 0.269 | 3.18 | 0 | 2.53 | 30.8 | 254 | event_recall, false_alerts_per_h |
| w4 | hailo8 | HOLD | 0.308 | 0.353 | 0 | 3.4 | 21.2 | 247 | event_recall |
| calib10 | hailo8 | HOLD | 0.885 | 3.89 | 2 | 3.07 | 21.1 | 245 | event_recall, false_alerts_per_h, duplicate_alerts |
