# 🖥️ Interactive Node-Graph Dashboard 구현 계획

> Lab Director Dashboard를 재현하는 인터랙티브 웹 대시보드

---

## 1. 프로젝트 개요

이미지에서 보이는 **Lab Director Dashboard**를 순수 HTML/CSS/JS로 구현합니다.
핵심은 노드 간 **동적으로 흐르는 연결선 애니메이션**입니다.

---

## 2. 동적 연결선 구현 원리 ⚡

노드 사이를 잇는 **파란색 점선이 흐르는 듯 움직이는 효과**의 핵심 원리:

### Step 1: SVG Path로 곡선 정의
```html
<svg>
  <path d="M 100,200 C 200,100 300,300 400,200" class="connection-line"/>
</svg>
```

### Step 2: CSS로 점선 패턴 생성
```css
.connection-line {
  stroke: #00bfff;
  stroke-width: 2;
  fill: none;
  stroke-dasharray: 8 4;  /* 8px 선 + 4px 간격 = 점선 */
}
```

### Step 3: 애니메이션으로 흐르는 효과
```css
@keyframes dash-flow {
  to { stroke-dashoffset: -20; }
}

.connection-line {
  animation: dash-flow 0.8s linear infinite;
}
```

> 💡 `stroke-dashoffset`을 계속 변화시키면 점선 패턴이 이동하면서 **"흐르는"** 것처럼 보입니다!

---

## 3. 프로젝트 구조

```
dashboard/
├── index.html              # 메인 HTML
├── css/
│   ├── main.css            # 레이아웃 & 다크테마
│   ├── nodes.css           # 노드 카드 스타일
│   └── animations.css      # 연결선 애니메이션
├── js/
│   ├── app.js              # 앱 초기화 & 이벤트
│   ├── nodes.js            # 노드 데이터 & 렌더링
│   ├── connections.js      # SVG 연결선 생성 & 관리
│   └── drag.js             # 드래그 & 줌 인터랙션
└── data/
    └── dashboard-data.json # 노드/연결 데이터 정의
```

---

## 4. 구현 단계

### Phase 1: 기본 레이아웃 (다크 테마)
- [x] `#0a0e1a` 배경의 다크 테마
- [x] 상단 헤더 (제목, 날짜, 탭 내비게이션)
- [x] 하단 범례/필터 바

### Phase 2: 노드 렌더링
- [x] 노드 데이터 JSON 정의
- [x] HTML div 기반 노드 카드 생성
- [x] 노드 타입별 스타일 (PI=빨강, 기능=파랑, 아이콘=분홍 등)
- [x] 그룹 영역 (Research Assistant, Lecture Assistant) 점선 테두리

### Phase 3: 동적 연결선 (⭐ 핵심 기능)
- [x] SVG 레이어 생성 (노드 HTML 뒤에 배치)
- [x] 노드 좌표 기반 Bezier 곡선 Path 자동 생성
- [x] CSS `stroke-dasharray` + `@keyframes` 애니메이션 적용
- [x] 연결 방향 표시 (화살표 마커)

### Phase 4: 인터랙션
- [x] 노드 드래그 앤 드롭
- [x] 드래그 시 연결선 실시간 업데이트
- [x] 캔버스 패닝 (배경 드래그)
- [x] 마우스 휠 줌

### Phase 5: 폴리싱
- [x] 네온 glow 효과
- [x] 호버 애니메이션
- [x] 탭 전환 기능
- [x] 연결선 위 파티클 효과 (선택)

---

## 5. 사용 기술

| 기술 | 용도 |
|------|------|
| **HTML5** | 전체 구조, 노드 요소 |
| **CSS3** | 다크 테마, 글래스모피즘, 애니메이션 |
| **JavaScript (Vanilla)** | 노드 렌더링, 드래그, 이벤트 |
| **SVG** | 연결선 (path, marker) |
| **CSS Keyframes** | 점선 흐름 애니메이션 |

> 외부 라이브러리 없이 순수 웹 기술만으로 구현합니다.

---

## 6. 예상 결과물

구현이 완료되면 다음과 같은 대시보드를 볼 수 있습니다:

- 🎨 **다크 네이비 배경**에 네온 테두리의 노드 카드들
- 🔗 노드 간 **파란 점선이 끊임없이 흘러가는** 애니메이션
- 📦 "Research Assistant", "Lecture Assistant" 그룹 영역
- 🖱️ 노드를 **드래그하면 연결선이 따라 움직이는** 인터랙션
- ✨ 호버 시 **glow 효과**가 강해지는 반응형 UI
- 🔍 마우스 휠로 **줌 인/아웃** 가능

---

## 7. 실행 방법

```bash
cd /home/userkh/workspace/Codex_test
python3 -m http.server 8080 --bind 0.0.0.0 --directory dashboard
# 로컬 실행: http://localhost:8080
# 원격/클라우드 IDE: 8080 포트 포워딩 URL 접속
```

---

## 8. 참고

- 모든 노드 위치는 JSON 데이터로 관리하여 쉽게 수정 가능
- 연결 관계도 JSON에서 정의하므로 노드 추가/삭제가 간편
- 추후 React/Vue 등 프레임워크로 마이그레이션 가능한 구조
