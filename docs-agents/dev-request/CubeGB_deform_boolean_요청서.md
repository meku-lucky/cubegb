# CubeGB `.cgb` 포맷 확장 요청: Deformation & Boolean

> **상태: 전부 구현 완료 ✅** (Priority 1–3). Partial Sweep · taper · bevel · shear ·
> boolean(difference/union/intersection)이 cgb → bake → viewer(×2) → blender_addon 전
> 구간에 들어갔고, 하위 호환·테스트 통과. 인식(recognition) 자동 활용은 의도적으로 보류.
> 규약은 [`docs/cgb-format.md`](../../docs/cgb-format.md), 동작 예시는
> `samples/{treasure_chest,keyhole_lock,cat_knight_master}.cgb`. 아래는 원본 요청서.


현재 `.cgb`는 큐브·실린더·콘·스피어 프리미티브를 `transform`(위치/회전/스케일)으로만
조합한다. 표현력을 높이기 위해 **deformation(변형)** 과 **boolean(CSG)** 연산을
추가한다. 아래 우선순위 순으로 검토·구현한다.

## 설계 원칙 (반드시 지킬 것)

- **하위 호환:** 새 필드는 전부 선택적(optional). 필드가 없으면 기존과 100% 동일하게
  동작해야 하고, 기존 샘플·테스트가 그대로 통과해야 한다.
- **단일 원본 유지:** `.cgb`는 여전히 작고(KB), 사람이 읽을 수 있고, git-diff 친화적인
  파라메트릭 JSON이어야 한다. 정점을 직접 나열하지 말 것.
- **변형 수학은 한 곳에 정의:** 베이커(`bake/`)와 뷰어(`viewer/`)가 같은 변형 함수를
  공유(동일 알고리즘으로 동일 메쉬 생성)해, "뷰어와 베이크 결과 불일치"가 생기지 않게
  할 것.
- **영향 범위:** 각 기능은 `cgb/`(스키마·검증) → `bake/`(메쉬 생성 수학) →
  `viewer/`(three.js 렌더) → `blender_addon/`(임포트) → `recognition/`(피팅) 순으로
  다운스트림부터 적용. 인식(피팅)은 맨 마지막.

## 우선순위 1 — Partial Sweep (부분 실린더/콘)

곡면 뚜껑·아치·배럴·터널 등을 표현하기 위해, 실린더와 콘을 360°가 아니라 일부 각도만
그릴 수 있게 한다.

- 파라미터: `sweep_start`, `sweep_end` (각도, 기본 0~360). 예: 반원통 뚜껑은 0~180.
- 닫힌 단면을 만들지(끝면 cap) 여부 옵션.
- 베벨/테이퍼보다 먼저 — 보물상자 곡면 뚜껑 같은 흔한 케이스를 가장 적은 비용으로 해결.

## 우선순위 2 — Deformation 필드

각 프리미티브에 선택적 `deform` 객체를 추가한다. 아래 세 가지부터 (bend/twist는 1차 제외).

- `taper`: 한 축을 따라 단면을 좁히거나 넓힘. 예: `taper: [x_ratio, y_ratio]` (끝단 배율).
  다리·칼날·화분·기둥 등.
- `bevel`: 모서리를 깎거나 둥글림. 파라미터: 깎는 폭 하나. 각진 블록 느낌을 줄여 톤↑.
- `shear`: 한 축을 기울임. 비스듬한 지붕·받침 등.

구현 주의:

- taper/bend 등 곡률이 생기는 변형은 베이크 시 프리미티브의 세그먼트(subdivision)를 충분히
  높여야 변형이 매끄럽게 보인다. 면당 사각형 하나면 직선 보간이라 변형이 안 드러난다.
- 검증(validation)에 파라미터 범위 체크 추가 (예: bevel 0~0.5, taper > 0).
- 권장: taper 하나만 먼저 `cgb→bake→viewer→blender_addon→recognition` 전 구간을
  관통시켜 본다. 그 경로가 bevel·shear 추가의 템플릿이 된다.

Blender 애드온 매핑(참고): taper/shear/bend는 Simple Deform 모디파이어, bevel은 Bevel
모디파이어에 대응 가능. 1차에는 변형된 메쉬로 임포트(간단)하고, 이후 모디파이어 매핑으로
편집성을 살리는 방향으로 업그레이드.

## 우선순위 3 — Boolean / CSG (difference 중심)

프리미티브를 빼기용으로 중첩시켜 구멍·음각·홈을 표현한다 (자물쇠 구멍, 관통 홀, 서랍 홈
등). 연산: union, difference(subtract), intersection. 우선은 difference부터.

- `.cgb`에는 boolean을 **선언적으로 저장만** 한다. 어떤 프리미티브가 "cutter(빼기용)"인지
  표시하고, 어떤 대상에서 빼는지 관계를 명시.
- 실시간 뷰어에서 매 프레임 메쉬 교차를 계산하지 말 것 (느리고 불안정). 대신:
  - **뷰어:** cutter 프리미티브를 반투명(예: 빨강)으로 "빠질 영역"만 표시하고 실제로는 안
    뺀다.
  - **베이크:** glTF/OBJ로 구울 때만 실제 mesh boolean을 1회 수행해 최종 결과 메쉬를 만든다.
  - **Blender 애드온:** cutter를 Boolean 모디파이어(difference) 대상으로 매핑.
- mesh boolean은 엣지 케이스(면이 정확히 겹치거나 거의 닿는 경우)에서 깨지기 쉬우니, 검증된
  boolean 라이브러리 사용을 검토할 것.

## 권장 작업 순서

1. **Partial Sweep** (실린더/콘 부분 스윕)
2. **Deformation** — taper 먼저 전 구간 관통 → bevel → shear
3. **Boolean(difference)** — 선언적 저장 + 베이크 시점 계산 + 뷰어는 표시만

각 단계는 독립적으로 머지 가능하게, 하위 호환을 깨지 않는 선에서 진행한다.
