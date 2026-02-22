# 대시보드 ~ ML 논의 최종 정리

요청 기준: `코스피/코스닥` 초기 대화는 제외하고, **대시보드 생성부터 머신러닝 관련 논의까지** 정리.

## 1) 대시보드 구현 시작점

- `dashboard/PLAN.md`를 기준으로 구현 진행.
- 초기 상태는 계획 문서만 있었고 실행 코드가 없었음.
- 계획서에 제시된 구조대로 `index.html`, `css/`, `js/`, `data/` 파일 세트를 신규 생성.

## 2) 실제 구현된 대시보드 기능

- 다크 테마 레이아웃, 헤더/탭/범례/상태바
- JSON 기반 노드/그룹 렌더링
- SVG Bezier 연결선 자동 생성
- 점선 흐름 애니메이션(`stroke-dasharray`, `stroke-dashoffset`)
- 화살표 마커
- 노드 드래그 + 연결선 실시간 업데이트
- 캔버스 패닝 + 휠 줌
- 탭 전환(복수 뷰)
- 노드 타입 필터(PI/Function/Agent/Tool)
- 네온 glow/호버 효과

## 3) 계획서(PLAN) 반영 상태

- `dashboard/PLAN.md`의 Phase 2~5 항목을 완료 상태로 갱신.
- 실행 명령을 명확하게 보완:
  - `cd /home/userkh/workspace/Codex_test`
  - `python3 -m http.server 8080 --bind 0.0.0.0 --directory dashboard`

## 4) 접속 이슈 점검과 정리

- 리소스 응답 검증(`index/css/js/json`)은 모두 `HTTP 200` 확인.
- 접속 문제는 코드 자체보다는 실행 컨텍스트/포트/프로세스 상태 이슈로 정리.
- `favicon.ico 404` 노이즈 제거를 위해 `index.html`에 빈 파비콘(`data:,`) 추가.
- 요청에 따라 실행 중 서버 프로세스를 종료하고, 직접 실행/종료 방법 안내.

## 5) 코드 리뷰 스킬 탐색

- 설치 전 탐색 요청에 따라 스킬 조회 수행.
- 공식 curated 목록에는 코드리뷰 전용 스킬이 없었음.
- 외부 생태계에서 후보 확인:
  - `coderabbitai/skills/code-review`
  - `jwynia/agent-skills/code-review`
  - `yeachan-heo/oh-my-claudecode/code-review`
- 비교 후 `coderabbitai`를 1순위로 제안했으나 설치는 보류.

## 6) 사용자 워크플로우(진동/FDC + Swin-MAE/PatchTST + 통합 + 이상탐지 + 리포트)

- 질문 핵심: 현재 UI 방식으로 표현 가능한지.
- 결론: **가능**.
- 제안한 시각화:
  - 단계별 노드 상태(`idle/running/done/error`)
  - 실행 중 노드 반짝임
  - 활성 연결선 강조
  - 리포팅 단계에서 리포트 노드 강조
- 연동에 필요한 백엔드 인터페이스:
  - 작업 시작
  - 작업 상태 조회
  - 실시간 이벤트(SSE/WS)
  - 리포트 결과 제공

## 7) 라이브러리 사용 여부

- 현재 대시보드는 React Flow/D3 없이 **Vanilla JS + SVG**로 작성.
- 사용 기술:
  - HTML/CSS/Vanilla JavaScript
  - SVG
  - 브라우저 기본 API(pointer/fetch)
  - Google Fonts
- 고급 편집/대규모 그래프가 필요해지면 React Flow/D3 도입 검토 가능.

## 8) 학습 중 epoch/loss 그래프 표시 가능 여부

- 결론: **가능**.
- 권장 아키텍처:
  - 백엔드: `epoch/loss/val_loss` 스트리밍
  - 프론트: 실시간 미니 차트/상세 차트 렌더링
- 정리:
  - 실시간 대시보드: 프론트 렌더링이 유리
  - 최종 리포트: 백엔드(matplotlib 등) 정적 그래프도 적합

## 9) 상업적 사용 관련 확인

- 의도 확인: “내가 자유롭게 써도 되는지, 제한 코드 포함 여부”.
- 정리:
  - 대시보드 코드 자체는 제한적 상용 라이브러리 의존 없이 구성
  - Google Fonts는 상업적 사용 가능 범주
  - 실제 라이선스 리스크는 향후 붙일 모델/가중치/데이터셋에서 더 큼

## 10) PatchTST 논문/코드베이스 확인

- 제공한 arXiv 링크(`2211.14730`)가 PatchTST 논문인지 확인 완료.
- 공식 GitHub도 확인 완료.
- 해당 출처 기반 코드 작성 가능하다고 답변 후 구현 준비 단계로 진행.

## 11) 로컬 vs Colab 실행 전략

- 사용자 환경: `torch` 미설치, `cuda` 미보유.
- 결론: 초기 실험/검증은 **Colab 권장**.
- 이유:
  - GPU 즉시 사용 가능
  - 학습 반복 속도 유리
  - 이후 GPU PC로 이식 용이
- 보완:
  - Colab은 직접 원격 조작은 불가하지만, 실행 가능한 셀/코드는 제공 가능.
  - Colab 검증 후 GPU PC 이식은 가능(버전 고정/경로 정리/체크포인트 로딩 처리).

## 12) 폴더 정리 요청 처리

- 요청에 따라 `dashboard`를 제외한 폴더 삭제 수행.
- `AGENTS.md`는 유지.
- 현재 구성:
  - `dashboard/`
  - `AGENTS.md`

