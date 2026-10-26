# EdgeGate

텍스트 한 줄로 엣지 기기에 새 인식 능력을 만들고, 배포 전에 백엔드별 양자화 위험을 감사 에이전트가 판정한다.
Meta Global AI Developer Hackathon (인디 트랙, "Push the Frontier").

흐름: 지시 입력 → SAM 3.1 자동 라벨 → 경량 검출기 증류 → 백엔드별 양자화 → 감사 에이전트(Muse Glimmer) 판정 → 통과한 백엔드에만 배포 → 팬/틸트 짐벌 폐루프 추종.

## GX10 환경

```bash
bash scripts/setup_gx10.sh          # 모델 다운로드 + SAM 3.1 env + 점검
bash scripts/start_glimmer.sh       # Glimmer vLLM 서버 (:8001)
bash scripts/start_glimmer.sh test
```
