| run | candidate | backend | recall | false/h | always deploy | false alerts only | rules | rules + Glimmer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | clean | trt | 0.885 | 5.65 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s1 | clean | hailo8 | 0.846 | 7.06 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s2 | clean | trt | 0.731 | 3.18 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s2 | clean | hailo8 | 0.731 | 2.12 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s3 | clean | trt | 0.962 | 4.24 | ships | holds | HOLD (false_alerts_per_h) | HOLD (rules), audited |
| s3 | clean | hailo8 | 0.962 | 4.59 | ships | holds | HOLD (false_alerts_per_h, duplicate_alerts) | HOLD (rules), audited |
| s3 | bgr | trt | 0.769 | 3.53 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s3 | bgr | hailo8 | 0.808 | 3.89 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s3 | no_norm | trt | 0 | 0 | ships | ships | HOLD (event_recall, detection_delay_s_p95) | HOLD (rules) |
| s3 | no_norm | hailo8 | 0 | 0 | ships | ships | HOLD (event_recall, detection_delay_s_p95) | HOLD (rules) |
| s3 | w4 | hailo8 | 0.615 | 4.59 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts, detection_delay_s_p95) | HOLD (rules), audited |
| s3 | calib10 | hailo8 | 0.923 | 5.3 | ships | holds | HOLD (false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s4 | clean | trt | 0.846 | 4.95 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s4 | clean | hailo8 | 0.885 | 4.95 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s4 | bgr | trt | 0.692 | 5.3 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s4 | bgr | hailo8 | 0.769 | 3.18 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |
| s4 | no_norm | trt | 0 | 0 | ships | ships | HOLD (event_recall, detection_delay_s_p95) | HOLD (rules) |
| s4 | no_norm | hailo8 | 0.269 | 3.18 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s4 | w4 | hailo8 | 0.308 | 0.353 | ships | ships | HOLD (event_recall) | HOLD (rules), audited |
| s4 | calib10 | hailo8 | 0.885 | 3.89 | ships | holds | HOLD (event_recall, false_alerts_per_h, duplicate_alerts) | HOLD (rules) |

always deploy: ships all 20 builds; 17 would miss more than 10% of arrivals, 16 would raise more than 1 false alert/h
false alerts only: ships 4: s3 no_norm trt (recall 0.00, 0 false/h), s3 no_norm hailo8 (recall 0.00, 0 false/h), s4 no_norm trt (recall 0.00, 0 false/h), s4 w4 hailo8 (recall 0.31, 0.35 false/h)
rules: ships 0
rules + Glimmer: ships 0; 4 audited, 0 vetoed
left out: 12 builds measured on another eval set (--all shows them)
