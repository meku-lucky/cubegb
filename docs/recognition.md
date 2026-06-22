# 인식 파이프라인 (이미지 → `.cgb`)

> **한국어** · [English](recognition.en.md)

인식 파이프라인은 이미지 한 장으로 `.cgb`를 채웁니다. CubeGB의 AI 부분이며, 정밀
복원이 아니라 의도적으로 **"쓸 만한 블록아웃(good-enough blockout)"** 을 목표로 합니다 —
0에서 모델링하는 것보다 빠른 출발점입니다.

```
이미지 ─► 세분화(SAM) ─► 깊이(Depth Anything V2) ─► 역투영 ─►
        세그먼트별 점군 ─► 프리미티브 피팅 + 자세 정규화 ─► .cgb
```

단계 (`recognition/`):

| 파일 | 단계 | 하는 일 |
|---|---|---|
| `segment.py` | Phase 4 | SAM 자동 마스크 생성 → 주요 영역 마스크 |
| `depth.py`   | Phase 4 | Depth Anything V2(MiDaS 폴백) → 깊이맵; 핀홀 역투영 → 세그먼트별 3D 점군 |
| `fit.py`     | Phase 5 | PCA 자세 정규화 + 월드축 정렬; 잔차 최소 기준으로 cube/cylinder/cone/sphere 피팅; 가림/두께 보정; `.cgb` 저장 |

> 아래 명령어는 macOS / Linux 기준입니다. Windows에서 다른 부분은 **🪟 Windows** 표시와
> 함께 따로 안내합니다.

## 의존성 설치

```bash
pip install -r requirements.txt -r requirements-recognition.txt
```

(이 명령어는 Windows에서도 동일합니다.)

- **GPU**: NVIDIA GPU가 있으면 CUDA, Apple Silicon은 MPS를 자동 사용합니다. 없으면
  CPU로도 동작하지만 느립니다.

## 모델 가중치 (별도 다운로드)

파이썬 의존성에는 모델 체크포인트가 **포함되지 않습니다**.

### SAM (Segment Anything) — Apache-2.0

아래 중 하나를 받으세요(정확도 ↔ 속도·용량):

| 모델 | 파일 | 크기 | URL |
|---|---|---|---|
| `vit_h` | `sam_vit_h_4b8939.pth` | ~2.4GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| `vit_l` | `sam_vit_l_0b3195.pth` | ~1.2GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| `vit_b` | `sam_vit_b_01ec64.pth` | ~375MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

**macOS / Linux:**

```bash
mkdir -p models
curl -L -o models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**🪟 Windows** — Windows 10/11에는 `curl`이 기본 내장돼 있지만, **PowerShell에서는 `curl`이
`Invoke-WebRequest`의 별칭**이라 위 옵션이 통하지 않습니다. `curl.exe`를 명시하세요.

- PowerShell:

  ```powershell
  mkdir models
  curl.exe -L -o models\sam_vit_h_4b8939.pth `
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

  (또는 순수 PowerShell 방식:
  `Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -OutFile models\sam_vit_h_4b8939.pth`)

- 명령 프롬프트(cmd):

  ```bat
  mkdir models
  curl -L -o models\sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

### Depth Anything V2 — 자동 다운로드 (할 일 없음)

`transformers`가 첫 실행 때 Hugging Face에서 자동으로 받습니다
(`depth-anything/Depth-Anything-V2-Small-hf`). 별도 작업이 없습니다. 라이선스는
**변형별로 다르니** 재배포·상업적 사용 전 확인하세요.
[Depth Anything V2 저장소](https://github.com/DepthAnything/Depth-Anything-V2) 참고.

### MiDaS (선택, 깊이 폴백) — MIT

`torch.hub`로 자동 로드됩니다.

> CubeGB 본체 코드는 MIT이지만, 각 모델의 라이선스 준수 책임은 사용자에게 있습니다.
> [README의 모델 표](../README.md#모델--데이터-라이선스)도 참고하세요.

## 실행

### 체크포인트 경로를 환경변수로 지정 (권장)

**macOS / Linux (bash/zsh):**

```bash
export CUBEGB_SAM_CHECKPOINT="$PWD/models/sam_vit_h_4b8939.pth"
```

**🪟 Windows:**

- PowerShell:

  ```powershell
  $env:CUBEGB_SAM_CHECKPOINT = "$PWD\models\sam_vit_h_4b8939.pth"
  ```

- 명령 프롬프트(cmd):

  ```bat
  set CUBEGB_SAM_CHECKPOINT=%CD%\models\sam_vit_h_4b8939.pth
  ```

> cmd/PowerShell에서 `set`/`$env:`로 지정한 값은 **그 터미널 세션에서만** 유효합니다.
> 영구 지정은 시스템 환경변수 설정(또는 PowerShell 프로필)을 사용하세요.

### CLI

**macOS / Linux:**

```bash
python -m recognition.fit IMAGE.jpg \
  --sam-checkpoint models/sam_vit_h_4b8939.pth \
  --sam-model-type vit_h \
  --out result.cgb \
  [--depth-checkpoint ...] [--depth-backend {auto,depth_anything_v2,midas}] \
  [--device cuda|cpu] [--max-segments N] [--fov 55] [--target-size 1.5]
```

**🪟 Windows** — 위처럼 줄 끝에 `\`를 쓰면 동작하지 않습니다. **한 줄로 입력**하거나,
줄바꿈 문자를 cmd는 `^`, PowerShell은 백틱(`` ` ``)으로 바꾸세요. 경로 구분자도
`models\sam_vit_h_4b8939.pth`처럼 역슬래시를 쓰면 됩니다. 예(PowerShell, 한 줄):

```powershell
python -m recognition.fit IMAGE.jpg --sam-checkpoint models\sam_vit_h_4b8939.pth --sam-model-type vit_h --out result.cgb
```

`--sam-model-type`은 받은 체크포인트와 맞춰야 합니다(`vit_h`/`vit_l`/`vit_b`).

결과 `.cgb`는 스키마에 유효하며 [뷰어](viewer.md)에서 열거나, [베이크](baker.md)하거나,
[Blender로 임포트](blender-addon.md)할 수 있습니다. GUI에서 한 번에 하려면
[CubeGB Studio](studio.md)를 쓰세요.

## 문제 해결 (Troubleshooting)

- **Apple Silicon (MPS):** SAM의 자동 마스크 생성기는 `float64` 텐서를 만드는데 MPS가
  이를 거부합니다(`Cannot convert a MPS Tensor to float64`). 그래서 CubeGB는 **SAM을
  자동으로 CPU에서** 실행합니다(한 줄 경고가 표시됨). 깊이 모델은 MPS를 그대로 씁니다.
  `vit_h`는 CPU에서 느리니 빠른 반복엔 `vit_b`를 권장합니다.
- **`open3d` 설치 실패** (아주 최신 파이썬, 예: 3.14): 선택적 `.ply` 디버그 내보내기에만
  쓰여 생성에는 불필요합니다 — 건너뛰거나, 점군 디버깅이 필요하면 Python 3.10–3.12를
  쓰세요.
- **느리거나 메모리 부족:** `vit_h`가 가장 큰 SAM 모델입니다. `vit_l`/`vit_b`로 낮추거나,
  `--max-segments`를 줄이거나, 입력 이미지를 축소하세요.

## 규약 & 가정

- **월드 좌표계:** 파이프라인은 CubeGB의 Y-up·오른손·미터 좌표계로 점을 출력해
  베이커/뷰어/애드온과 일관됩니다.
- **스케일 모호성:** 단일 이미지는 절대 크기를 알 수 없어 점군을 적당한 바운딩 박스로
  정규화합니다(`--target-size`, 기본 ~1.5m).
- **가림(occlusion):** 보이는 앞면만 있으므로, 피팅된 프리미티브의 깊이 방향 두께를
  대칭/두께 휴리스틱으로 채우고 중심을 뒤로 밀어 종잇장처럼 얇아지지 않게 합니다.
  가려진 형상은 *추정*이지 실측이 아닙니다.

모델 가중치나 무거운 의존성이 없으면 모듈 임포트는 되지만, 실제 실행 시 명확하고
실행 가능한 오류 메시지를 냅니다.
