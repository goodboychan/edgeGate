# EdgeGate

한 문장(음성 또는 텍스트)으로 집 안의 이기종 엣지 기기에 새 인식 능력을 가르치고, 배포 전에 칩별 위험을 감사 에이전트가 판정한다.
Meta Global AI Developer Hackathon, 인디 트랙 "Push the Frontier".

**흐름:** "현관에 택배 상자가 놓이면 알려줘" → Muse Voice Transcribe → Muse Spark 계획 → SAM 3.1 자동 라벨 → 학생 검출기 증류 → 백엔드별 컴파일 → Muse Glimmer 감사 → 통과한 백엔드에만 배포 → Home Assistant 알림, Creoqode Nova가 목표를 바라봄.

## 해커톤 기간 작업

이 저장소의 코드는 모두 해커톤 기간(2026-10-26 ~ 11-04)에 작성했다. 작업 시점은 커밋 기록으로 확인할 수 있다.
기존 코드베이스는 이 저장소에 포함하지 않았다.

- **기존에 있던 것:** 홈랩 하드웨어(GX10, 엣지 보드, 센서, Nova), 아래 오픈소스 도구, 공개 모델 가중치
- **이번 주에 만든 것:** 이 저장소의 모든 코드 (라벨링·증류·컴파일·측정 파이프라인, 감사 에이전트와 MCP 도구, 제어·집 연동)

아래 외부 도구는 수정 없이 그대로 사용한다.

| 외부 도구 | 용도 |
|---|---|
| vLLM | Muse Glimmer 서빙 |
| facebookresearch/sam3 | SAM 3.1 추론 |
| Qualcomm QNN SDK | VENTUNO Q Hexagon 컴파일 |
| HailoRT / Dataflow Compiler | Hailo-8 컴파일 |
| edgetpu_compiler | Coral Edge TPU 컴파일 |
| TensorRT | Jetson 컴파일 |
| DepthAI | OAK-D Lite (Myriad X) |

## 사용한 Meta 모델

Muse Glimmer 30B (로컬 감사 에이전트), Muse Spark / Muse Spark Open (계획·검증), Muse Voice Transcribe (음성 지시), SAM 3.1 (교사 라벨), Muse Code (개발 보조).

## GX10 환경

```bash
bash scripts/setup_gx10.sh               # 모델 다운로드 + SAM 3.1 env + 점검
bash scripts/start_glimmer.sh            # Glimmer vLLM 서버 (:8001, vLLM >= 0.32, 기본 FP8)
bash scripts/start_glimmer.sh test       # 응답 확인
bash scripts/start_glimmer.sh toolcheck  # 도구 호출 1회 왕복
```
