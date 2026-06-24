# CubeGB (큐브공방 · CGB) 개발 요청서

> **이 문서의 용도**: Claude Code / Codex 등 코드 에이전트에게 전달하여 단계적으로 개발하기 위한 사양서입니다.
> 각 Phase는 **독립적으로 동작 확인이 가능한 단위**로 나뉘어 있습니다. **반드시 Phase 순서대로 진행하고, 각 Phase 끝의 "검증 체크포인트"를 통과한 뒤 다음 Phase로 넘어가세요.** 한 번에 전체를 구현하지 마세요.

---

## 0. 프로젝트 개요

**한 줄 정의**: 이미지 한 장을 입력받아, **기초 3D 프리미티브(큐브·실린더·콘·스피어 등)의 조합**으로 분해·재구성하는 초경량 "이미지 → 블록아웃(blockout)" 생성기.

**핵심 차별점**
- 기존 image-to-3D 모델(Hunyuan3D, TRELLIS, Tripo 등)은 편집이 어려운 고밀도 텍스처 메쉬/가우시안 스플랫을 출력함.
- CubeGB는 **편집 가능한 파라메트릭 프리미티브**를 출력함 → 파일 용량 KB 단위, Blender 등에서 즉시 재가공 가능.
- 타겟: 디자이너 / 3D 아티스트가 디테일 작업 전 **블록아웃(greybox) 단계**를 자동화하는 용도.

**타겟 사용 환경**: ComfyUI 커스텀 노드로 패키징하여 오픈소스(GitHub) 공개. 최종적으로 ComfyUI Registry 등록을 목표로 함.

**스코프 제외(비목표)** — 의도적으로 다루지 않음:
- 유기적 형태(얼굴, 동식물, 천 등)의 정밀 복원 → 본 도구는 **하드서피스(가구·건물·기계·소품 등 인공물)** 에 특화.
- 고품질 텍스처 생성.
- 단일 이미지로부터의 완벽한 메트릭(실측) 복원. 가려진 면은 합리적 추정.

---

## 1. 핵심 아키텍처 원칙 (가장 중요)

1. **원본 / 파생 분리**: `.cgb`(프리미티브 파라메트릭 표현)가 **유일한 원본(source of truth)**. glTF·OBJ·FBX 같은 메쉬는 모두 `.cgb`에서 **구워내는(bake) 파생물**.
2. **출력 3계층**:
   - 기본·원본: `.cgb` (JSON, 초경량, 무손실, git 친화적)
   - 메쉬 익스포트: glTF/GLB(1순위) · OBJ · FBX(호환용) — 저폴리 유지
   - Blender 연동: **별도 임포터 애드온**이 `.cgb`를 읽어 **편집 가능한 네이티브 프리미티브로 복원**(메쉬로 굽지 않음 → 편집성 유지)
3. **미들아웃(middle-out) 개발**: 어려운 인식(AI) 부분을 먼저 만들지 않는다. **`.cgb` 포맷 → 뷰어 → 베이커 → 임포터**(다운스트림 도구)를 먼저 완성해 출력을 눈으로 검증할 수 있게 한 뒤, 인식 파이프라인이 이 포맷을 "채우게" 한다. 인식이 불완전해도 전체 골격이 동작한다.

---

## 2. 기술 스택

| 영역 | 선택 | 비고 |
|---|---|---|
| 언어 | Python 3.10+ | 메인 |
| 딥러닝 | PyTorch | 추론 위주 |
| 세분화 | SAM (Segment Anything) | 사전학습, zero-shot |
| 깊이 추정 | Depth Anything V2 (또는 MiDaS) | 사전학습, zero-shot |
| 점군/기하 | Open3D, trimesh, numpy, scipy | 프리미티브 피팅·RANSAC |
| 메쉬 익스포트 | trimesh, pygltflib | glTF/OBJ. FBX는 후순위 |
| 뷰어 | three.js (웹, 단일 HTML) | `.cgb` 렌더 |
| Blender 연동 | Blender Python(`bpy`) 애드온 | `.cgb` 임포터 |
| 패키징 | ComfyUI custom node | 최종 배포 |

> **라이선스 주의**: 본체 코드는 MIT 권장. 단, 사용하는 사전학습 모델의 라이선스를 **반드시 확인**할 것(예: SAM은 Apache-2.0, Depth Anything은 변형별로 라이선스 상이 — 상업·재배포 가능 변형 선택). README에 모델 라이선스 고지 필요.

---

## 3. `.cgb` 포맷 스펙 (v0.1)

JSON. 새로운 바이너리 포맷을 발명하지 않는다.

```json
{
  "format": "cgb",
  "version": "0.1",
  "metadata": {
    "generator": "CubeGB v0.1",
    "source_image": "optional/path_or_null",
    "created_at": "ISO8601",
    "up_axis": "Y"
  },
  "units": "meter",
  "primitives": [
    {
      "id": "p0",
      "name": "seat",
      "type": "cube",
      "transform": {
        "position": [0.0, 0.4, 0.0],
        "rotation_euler": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0]
      },
      "params": { "size": [0.5, 0.08, 0.5] },
      "material": { "color": [0.6, 0.4, 0.2], "name": "wood" },
      "parent": null
    }
  ],
  "operations": []
}
```

**규약**
- 기본 형상 치수는 `params`에 담고, `transform`은 씬 내 배치(위치/회전)에 사용. 비균일 스케일도 허용하되 가급적 크기는 `params`로 인코딩하고 `scale`은 `[1,1,1]` 유지(편집성↑).
- `rotation_euler`: 라디안, XYZ 순서.
- `parent`: 계층 구조용 id 참조(없으면 null).
- `operations`: CSG 결합용(union/difference/intersection). **v0.1에서는 빈 배열로 두고 미구현**, v0.2+에서 도입.

**프리미티브 타입별 `params`** (v0.1 필수: cube, sphere, cylinder, cone):
- `cube`: `{ "size": [x, y, z] }`
- `sphere`: `{ "radius": r, "segments": 16 }`
- `cylinder`: `{ "radius": r, "height": h, "segments": 16 }`
- `cone`: `{ "radius": r, "height": h, "segments": 16 }`
- (v0.2+ 확장) `capsule`, `torus`, `truncated_cone`

**산출물**: 위 스키마의 JSON Schema 정의 파일(`cgb/schema.json`), 직렬화/역직렬화 + 검증 함수(`cgb/io.py`), 샘플 파일 2~3개(`samples/*.cgb`, 예: 의자, 탁자, 단순 건물).

---

## 4. 권장 리포지토리 구조

```
cubegb/
├── cgb/                  # 포맷: 스키마, IO, 검증
│   ├── schema.json
│   ├── io.py
│   └── validate.py
├── viewer/               # three.js 웹 뷰어 (단일 HTML)
│   └── index.html
├── bake/                 # .cgb -> glTF/OBJ/FBX
│   └── baker.py
├── blender_addon/        # cubegb 임포터 애드온
│   └── cubegb_import.py
├── recognition/          # 세분화 + 깊이 + 프리미티브 피팅
│   ├── segment.py
│   ├── depth.py
│   └── fit.py
├── comfyui_nodes/        # ComfyUI 커스텀 노드
│   └── __init__.py
├── samples/              # 샘플 .cgb + 테스트 이미지
├── tests/
├── README.md
└── LICENSE
```

---

## 5. 개발 단계 (Phase)

각 Phase는 **검증 체크포인트**를 통과해야 다음으로 진행합니다.

---

### Phase 0 — 프로젝트 셋업 & `.cgb` 포맷 정의

**목표**: 리포 골격과 원본 포맷을 확정한다.

**산출물**
- 위 4번의 리포 구조 생성, 가상환경/의존성(`requirements.txt`, `pyproject.toml`).
- `cgb/schema.json` (3번 스펙 기반 JSON Schema).
- `cgb/io.py`: `.cgb` load/save, `cgb/validate.py`: 스키마 검증.
- `samples/`에 손으로 작성한 유효 `.cgb` 샘플 2개 이상(의자, 탁자 등).
- 기본 `tests/`로 검증 통과 확인.

**검증 체크포인트** ✅
- 샘플 `.cgb`가 `validate.py`를 통과한다.
- load → save → load 라운드트립이 동일 데이터를 보존한다.

---

### Phase 1 — 뷰어 (three.js)

**목표**: `.cgb`를 시각적으로 확인할 수 있게 한다. (이후 모든 결과물의 눈 역할)

**산출물**
- `viewer/index.html`: 단일 HTML(three.js CDN). `.cgb` 파일을 드래그앤드롭 또는 파일 선택으로 로드.
- 각 프리미티브 타입을 three.js 지오메트리로 렌더, transform 적용, material color 반영.
- OrbitControls(회전/줌), 그리드/축 표시, 프리미티브별 이름/색 구분.

**검증 체크포인트** ✅
- Phase 0의 샘플 `.cgb`를 로드하면 의자/탁자 형태가 화면에 올바르게 보인다.
- 회전·줌이 동작한다.

---

### Phase 2 — 메쉬 베이커 (`.cgb` → glTF/OBJ)

**목표**: 원본을 표준 메쉬 포맷으로 구워낸다. (파생물 1)

**산출물**
- `bake/baker.py`: `.cgb` 입력 → 각 프리미티브를 메쉬로 인스턴스화 → 하나의 씬으로 결합.
- 출력: **glTF/GLB(1순위)**, OBJ. 각 프리미티브는 **이름 붙은 개별 노드/오브젝트**로 보존(계층·트랜스폼·머티리얼 유지).
- 테셀레이션(분할 수) 파라미터로 **저폴리 유지**(예: cylinder segments 기본 16, 조절 가능).
- CLI: `python -m bake.baker input.cgb --format glb --out out.glb`

**검증 체크포인트** ✅
- 생성된 `.glb`를 Blender(또는 Phase 1 뷰어/온라인 glTF 뷰어)에서 열면 원본과 동일한 형태로 보인다.
- 오브젝트가 이름별로 분리되어 있고 폴리곤 수가 낮다.

---

### Phase 3 — Blender 임포터 애드온

**목표**: `.cgb`를 Blender에서 **편집 가능한 네이티브 프리미티브**로 복원한다. (CubeGB의 킬러 기능)

**산출물**
- `blender_addon/cubegb_import.py`: Blender 애드온. `File > Import > CubeGB (.cgb)` 메뉴 추가.
- `.cgb`의 각 프리미티브를 `bpy.ops.mesh.primitive_*_add` 류로 **실제 Blender 프리미티브 오브젝트**로 생성(메쉬로 굽지 않음). 이름·트랜스폼·계층·색 반영.
- 좌표계/업축 변환 처리(`.cgb`는 Y-up → Blender는 Z-up).

**검증 체크포인트** ✅
- 애드온 설치 후 샘플 `.cgb` 임포트 시, Blender 아웃라이너에 이름 붙은 프리미티브 오브젝트들이 생성된다.
- 생성된 큐브/실린더를 **그대로 잡고 스케일·이동·모디파이어 적용**이 가능하다(편집성 유지 확인).

> **여기까지가 다운스트림 골격.** 인식이 없어도 손으로 만든 `.cgb`로 전 파이프라인(뷰어·베이커·Blender)이 동작해야 함.

---

### Phase 4 — 인식 파이프라인 A: 세분화 + 깊이

**목표**: 이미지에서 부분별 점군(point cloud)을 만든다.

**산출물**
- `recognition/segment.py`: SAM으로 입력 이미지를 영역 세분화 → 마스크 목록.
- `recognition/depth.py`: Depth Anything V2(또는 MiDaS)로 픽셀별 깊이맵 추정.
- 세그먼트 마스크 + 깊이맵을 결합해 카메라 역투영 → **세그먼트별 3D 점군** 생성. 디버그용 점군 시각화(Open3D) 또는 저장(`.ply`).

**검증 체크포인트** ✅
- 테스트 이미지(예: 의자 사진) 입력 시, 세그먼트별로 분리된 3D 점군이 생성되고 시각화로 형태가 식별된다.

> 단일 이미지는 가려진 뒷면 정보가 없음 → 점군은 보이는 면 위주임을 전제로 다음 Phase에서 보완.

---

### Phase 5 — 인식 파이프라인 B: 프리미티브 피팅 & 정규화 (핵심 난이도)

**목표**: 점군을 프리미티브로 치환하고 `.cgb`를 출력한다. (이미지 → `.cgb` 엔드투엔드 완성)

**산출물**
- `recognition/fit.py`:
  - 각 세그먼트 점군에 대해 **후보 프리미티브 피팅**(RANSAC / 최소제곱 / scipy 최적화). 큐브·실린더·콘·스피어 중 잔차(residual)가 가장 낮은 타입 선택.
  - **자세 정규화**: 주성분 분석(PCA) 등으로 주축 추정 후, 가급적 **월드 축에 정렬**.
  - **대칭 복원**: 대상이 대칭이면 보이는 면 기준으로 가려진 면 추정(예: 깊이 방향 두께 가정).
  - 결과를 `.cgb`로 직렬화(Phase 0 IO 사용).
- CLI: `python -m recognition.fit image.jpg --out result.cgb`

**검증 체크포인트** ✅
- 테스트 이미지 입력 → `.cgb` 출력 → Phase 1 뷰어/ Phase 3 Blender에서 열었을 때, 원본 사물의 **블록아웃으로 인식 가능한 형태**가 나온다.
- 프리미티브가 대체로 축에 정렬되어 있다(비뚤어진 큐브 최소화).

> 품질이 완벽할 필요는 없음. "디자이너가 0에서 만드는 것보다 빠른 출발점"이면 1차 성공.

---

### Phase 6 — ComfyUI 커스텀 노드 패키징

**목표**: 전체를 ComfyUI 노드로 묶어 배포 가능하게 한다.

**산출물**
- `comfyui_nodes/`: 다음 노드들 제공
  - `CubeGB Generate` (이미지 입력 → `.cgb` 데이터)
  - `CubeGB Save` (`.cgb` 파일 저장)
  - `CubeGB Bake` (`.cgb` → glb/obj 익스포트)
  - `CubeGB Preview` (노드 내 미리보기 — 가능하면)
- ComfyUI 커스텀 노드 규약 준수(`NODE_CLASS_MAPPINGS` 등), `README`에 ComfyUI Manager/Registry 설치 안내.
- 의존성·모델 가중치 다운로드 안내.

**검증 체크포인트** ✅
- ComfyUI에 노드가 로드되고, 이미지 입력 → `.cgb` 생성 → 저장/익스포트 워크플로우가 동작한다.

---

### Phase 7 — (선택) 정제 & 확장

**목표**: 품질·범위 향상. 시간 여유 시 진행.

**후보 작업**
- 미분 가능 렌더링(PyTorch3D 등)으로 프리미티브 파라미터를 입력 이미지에 맞춰 미세조정(render-and-compare).
- 프리미티브 타입 확장(capsule, torus, truncated_cone), `operations`(CSG 결합) 구현.
- FBX 익스포트 추가.
- 프리미티브 자동 병합/단순화(중복·미세 프리미티브 정리).

---

## 6. 개발 시 주의사항 (에이전트 전달용)

- **반드시 Phase 순서대로**. Phase 0~3(다운스트림)을 먼저 완성해 손으로 만든 `.cgb`로 검증한 뒤 Phase 4부터 인식을 붙일 것.
- 각 Phase는 **독립 실행/검증 가능**해야 함. Phase 경계에서 멈추고 사람 확인을 받을 것.
- 인식(Phase 4~5)은 완벽을 추구하지 말 것. "쓸 만한 블록아웃 출발점"이 기준.
- 모든 메쉬 출력은 **저폴리 유지**가 기본값.
- `.cgb`는 사람이 읽고 git diff 가능한 상태를 유지(가독성 있는 JSON).
- 사전학습 모델 라이선스를 확인하고 README에 고지.

## 7. 1차 완성(MVP) 정의

> Phase 0 ~ 6 통과 = MVP.
> "이미지 한 장 → ComfyUI에서 `.cgb` 생성 → Blender에서 편집 가능한 프리미티브로 임포트" 가 끝에서 끝까지 동작하면 1차 목표 달성.
