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
| s4 | clean | trt | 0.846 | 4.95 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |
| s4 | clean | hailo8 | 0.885 | 4.95 | ships | holds | HOLD (event_recall, false_alerts_per_h) | HOLD (rules) |

always deploy: ships all 10 builds; 8 would miss more than 10% of arrivals, 10 would raise more than 1 false alert/h
false alerts only: ships 0
rules: ships 0
rules + Glimmer: ships 0; 2 audited, 0 vetoed
left out: 22 builds measured on another eval set (--all shows them)
