# 드론 영상 실시간 소형 객체 탐지 (DJI Neo · Parrot Mambo FPV)

드론 영상, 웹캠, 영상 파일 또는 네트워크 스트림의 프레임을 받아 번들된 **FFCA-YOLO/TS-RPST** 모델로 소형 객체를 탐지하고, 결과를 OpenCV 창에 실시간으로 표시합니다.

| 입력 | 실행 파일 | 영상 경로 |
| --- | --- | --- |
| DJI Neo | `dji_neo_camera_inference.py` | Android DJI Fly 화면을 USB(scrcpy)로 미러링해 캡처 |
| Parrot Mambo FPV | `mambo_camera_inference.py` | PC가 Mambo Wi-Fi에 접속해 RTSP/RTP 영상을 직접 수신 |
| 웹캠·파일·스트림 | `python -m dji_neo` | `--source`로 지정한 OpenCV 입력 |

> 모든 경로는 **영상 수신과 탐지만** 합니다. 비행 명령을 보내거나 자동 비행하는 기능은 없습니다. 비행은 각 기체의 앱으로 조종합니다.

## 주요 기능

- 웹캠, 로컬 영상 파일, RTSP 등 OpenCV가 지원하는 입력 사용
- 저장소에 포함된 AI-TOD 기반 FFCA-YOLO 가중치로 추론
- 신뢰도 필터링 및 NMS(Non-Maximum Suppression)
- 원본 프레임 좌표로 바운딩 박스 복원
- `person`, `vehicle` 탐지 결과를 화면에 표시
- Python 코드에서 재사용할 수 있는 탐지기 API 제공

기본 모델의 전체 클래스는 다음과 같습니다.

`airplane`, `bridge`, `storage-tank`, `ship`, `swimming-pool`, `vehicle`, `person`, `wind-mill`

CLI는 이 중 `person`, `vehicle`, `car`를 요청하도록 작성되어 있습니다. 기본 AI-TOD 클래스 목록에는 `car`가 없으므로 실제 기본 모델에서 표시되는 대상은 `person`과 `vehicle`입니다.

## 프로젝트 구조

```text
dji_neo/
├── __init__.py                 # 공개 타입(Detection) 노출
├── __main__.py                 # python -m dji_neo 진입점
├── neo_detect.py               # CLI 옵션, 영상 입력, 화면 표시 루프
├── dji_neo_camera_inference.py # DJI Neo: scrcpy 화면 캡처(또는 RTMP) → 탐지
├── DJI_NEO_SETUP.md            # DJI Neo + Android + scrcpy 상세 설정
├── mambo_camera_inference.py   # Mambo FPV: RTSP/RTP 수신 → 탐지 + 추적 ID/궤적
├── mambo_stream_probe.py       # Mambo 스트림 진단(RTP 손실, 프레이밍, 디코드)
├── tests/                      # Mambo 수신·추적 단위 테스트
├── detectors.py                # 공통 Detection 형식으로 변환하는 탐지기 어댑터
├── ffca_detector.py            # 모델 로드, 전처리, 추론, NMS, 좌표 복원
├── air2s_types.py              # Detection 데이터 클래스
├── environment.yml             # Conda 환경 정의
├── requirements_air2s_windows.txt # 이전 Air2S 코드의 잔여 의존성 파일
├── inference.txt               # 이전 Air2S 실행 예시(현재 CLI와 호환되지 않음)
├── docs/
│   └── simulation_evidence/    # 이전 Tello 폐루프 시뮬레이션 자료
└── ffca_yolo/
    ├── weights/best.pt         # 기본 TS-RPST 체크포인트
    ├── data/AITOD.yaml         # 클래스 이름 및 데이터셋 설정
    ├── models/                 # YOLOv5 및 FFCA 계층/모델 정의
    └── utils/                  # 전처리, NMS, 좌표 변환 등 추론 유틸리티
```

실행 시 데이터 흐름은 다음과 같습니다.

```text
VideoCapture → RepositoryYoloDetector → FFCAYoloDetector
             → letterbox/정규화 → 모델 추론 → NMS
             → 원본 좌표 복원 → 대상 클래스 필터 → 화면 표시
```

`ffca_yolo/`는 독립 애플리케이션이라기보다 현재 프로젝트가 내부적으로 사용하는 번들 런타임입니다. 학습 스크립트와 AI-TOD 데이터셋은 포함되어 있지 않습니다.

## 요구 사항

- Python 3.12(제공된 Conda 설정 기준)
- PyTorch 및 torchvision
- OpenCV
- NumPy, pandas, PyYAML, tqdm, matplotlib, seaborn, psutil
- 화면 출력을 사용할 수 있는 데스크톱 환경
- GPU 실행 시 PyTorch가 인식하는 CUDA 환경

`dji-sdk-python`은 현재 실행 경로에서 import되지 않으므로 실시간 탐지만 사용할 때는 필요하지 않습니다.

## 설치

저장소 폴더에서 Conda 환경을 생성합니다.

```powershell
conda env create -f environment.yml
conda activate dji-air2s
```

제공된 `environment.yml`은 CPU 환경을 기준으로 `cpuonly`를 지정합니다. 이 환경에서는 아래 실행 예시처럼 `--device cpu`를 사용하십시오. CUDA GPU를 사용하려면 `cpuonly`를 제거하고 시스템 CUDA 버전에 맞는 PyTorch를 별도로 설치해야 합니다.

설치 확인:

```powershell
python -c "import cv2, torch; print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 실행 방법

이 저장소는 패키지 내부의 상대 import를 사용하며 별도의 설치 설정 파일(`pyproject.toml` 또는 `setup.py`)은 없습니다. 따라서 **`dji_neo` 폴더의 상위 디렉터리에서** 모듈로 실행합니다.

```powershell
cd ..
python -m dji_neo --source 0 --device cpu
```

탐지 창에서 `q`를 누르면 종료됩니다.

### 입력 소스 예시

웹캠 번호:

```powershell
python -m dji_neo --source 0 --device cpu
```

로컬 영상 파일:

```powershell
python -m dji_neo --source "C:\videos\flight.mp4" --device cpu
```

RTSP 스트림:

```powershell
python -m dji_neo --source "rtsp://user:password@192.168.0.10:554/stream" --device cpu
```

CUDA 첫 번째 GPU:

```powershell
python -m dji_neo --source 0 --device 0
```

드론 영상은 아래의 [DJI Neo](#dji-neo-실시간-추론-android--scrcpy)와 [Parrot Mambo FPV](#parrot-mambo-fpv-실시간-탐지추적) 전용 스크립트를 사용합니다.

### CLI 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--weights` | `ffca_yolo/weights/best.pt` | 사용할 PyTorch 체크포인트 경로 |
| `--data` | `ffca_yolo/data/AITOD.yaml` | 클래스 이름을 포함한 데이터 YAML 경로 |
| `--source` | `0` | 웹캠 번호, 영상 파일 경로 또는 스트림 URL |
| `--device` | `0` | CUDA 장치 번호(예: `0`) 또는 `cpu` |

현재 CLI에서 이미지 크기와 임계값은 각각 `640`, confidence `0.2`, IoU `0.45`로 고정되어 있습니다. 입력 영상은 화면에만 표시되며 탐지 결과나 출력 영상은 파일로 저장되지 않습니다.

## Python API 사용

다른 코드에서 모델을 직접 사용할 수도 있습니다. 아래 코드는 `dji_neo`의 상위 디렉터리에서 실행하는 것을 전제로 합니다.

```python
import cv2

from dji_neo.detectors import RepositoryYoloDetector

detector = RepositoryYoloDetector(
    weights="dji_neo/ffca_yolo/weights/best.pt",
    data="dji_neo/ffca_yolo/data/AITOD.yaml",
    target_labels=["person", "vehicle"],
    imgsz=640,
    confidence=0.2,
    iou=0.45,
    device="cpu",
)
detector.load()

frame = cv2.imread("sample.jpg")
if frame is None:
    raise FileNotFoundError("sample.jpg")

for detection in detector.detect(frame):
    print(
        detection.class_id,
        detection.label,
        detection.confidence,
        detection.xyxy,
        detection.center,
        detection.area,
    )

print("inference (ms):", detector.last_inference_ms)
print("NMS (ms):", detector.last_nms_ms)
```

`detect()`는 `Detection` 객체의 리스트를 반환합니다.

| 필드/속성 | 의미 |
| --- | --- |
| `class_id` | 모델의 정수 클래스 ID |
| `label` | 클래스 이름 |
| `confidence` | 탐지 신뢰도 |
| `xyxy` | 원본 프레임 기준 `(x1, y1, x2, y2)` 픽셀 좌표 |
| `center` | 바운딩 박스 중심 좌표 |
| `area` | 바운딩 박스 면적 |

`detectors.py`의 `ColorTargetDetector`는 HSV 범위로 주황색 물체를 찾는 간단한 시뮬레이션용 탐지기입니다. FFCA-YOLO 모델을 사용하지 않으며 현재 CLI에서는 호출되지 않습니다.

## 모델 교체

같은 런타임과 호환되는 체크포인트를 준비한 뒤 `--weights`로 지정합니다. 클래스 구성이 다르면 해당 체크포인트에 맞는 데이터 YAML도 `--data`로 함께 지정해야 합니다.

```powershell
python -m dji_neo `
  --weights "C:\models\custom.pt" `
  --data "C:\models\custom.yaml" `
  --source "C:\videos\input.mp4" `
  --device cpu
```

기본 가중치 또는 데이터 YAML 파일이 없으면 모델 로딩 단계에서 `FileNotFoundError`가 발생합니다.

## DJI Neo 실시간 추론 (Android + scrcpy)

```text
DJI Neo ─(Wi-Fi)→ Android 폰(DJI Fly로 조종) ─(USB + ADB)→ PC(scrcpy 창 캡처 → FFCA-YOLO)
```

PC는 기체에 직접 접속하지 않고 스마트폰의 DJI Fly 화면을 읽습니다. scrcpy는 `--no-control`로 실행되므로 PC 입력이 폰으로 전달되지 않습니다. 설치와 폰 설정은 [DJI_NEO_SETUP.md](DJI_NEO_SETUP.md)를 따릅니다(scrcpy·adb 설치, USB 디버깅, `adb devices`가 `device`인지 확인).

```powershell
python .\dji_neo_camera_inference.py `
  --weights ".\ffca_yolo\weights\best.pt" `
  --data ".\data\AITOD.yaml" `
  --input-mode scrcpy `
  --scrcpy-max-size 1280 `
  --device cpu `
  --classes 0 `
  --conf-thres 0.25 `
  --select-crop `
  --view-img
```

- `--select-crop`: scrcpy 창 크기가 안정된 뒤 선택 창이 뜹니다. **좌우 검은 띠를 뺀 카메라 영상 전체**를 드래그하고 Enter를 누릅니다. 선택 후 창 크기가 바뀌면 같은 비율로 영역을 맞춥니다.
- 로그에 출력된 좌표로 다음부터는 `--scrcpy-crop X Y W H`를 지정할 수 있습니다. 이 값은 scrcpy 창 크기 기준이므로 폰 방향이나 `--scrcpy-max-size`를 바꾸면 다시 선택해야 합니다.
- 녹화 버튼·배터리 등 DJI Fly UI는 영상 위에 겹쳐 있어 함께 캡처됩니다.
- PyTorch가 CUDA를 인식하지 못하면 `--device cpu`를 사용하고 `--half`는 빼십시오.

## Parrot Mambo FPV 실시간 탐지·추적

```text
Mambo FPV ─(Wi-Fi, rtsp://192.168.99.1/media/stream2)→ PC(RTP/UDP 수신 → FFmpeg 디코드 → FFCA-YOLO → IoU 추적)
```

준비:

1. FFmpeg 설치: `winget install --id Gyan.FFmpeg.Essentials --exact --source winget`
2. PC를 Mambo Wi-Fi(`Mambo_XXXXXX`)에 연결하고 IP가 `192.168.99.x`인지 확인합니다(`ipconfig`).
3. 휴대폰 FreeFlight Mini 앱은 꺼 두는 것을 권장합니다. 같은 Wi-Fi 대역폭을 나눠 쓰면 손실이 늘었습니다.
4. 처음 실행 시 Windows 방화벽이 Python의 UDP 수신 허용을 물으면 허용합니다.

```powershell
# 영상만 확인(YOLO 미로드)
python mambo_camera_inference.py --preview-only

# 탐지 + 추적 ID/궤적 (클래스 0 airplane, 6 person 등)
python mambo_camera_inference.py --weights ffca_yolo/weights/best.pt --device cpu --imgsz 320 --classes 6
```

- 기본 수신 경로는 `--backend rtp`(UDP)입니다. Mambo는 SDP로 알려 주는 SPS/PPS(Main/CABAC)가 실제 스트림(High/CAVLC)과 달라 OpenCV·VLC로 받으면 H.264 오류가 반복됩니다. `rtp` 경로는 스트림 안의 SPS/PPS만 사용합니다. 시험 결과 같은 캡처에서 오류가 314줄에서 0줄로 줄었습니다.
- Mambo의 RTSP-over-TCP는 부하가 걸리면 프레이밍이 깨져 UDP가 기본입니다. `--backend opencv`/`vlc`도 비교용으로 남아 있습니다.
- 영상이 끊겨도 창은 닫히지 않고 대기 화면을 표시하며 자동으로 재연결합니다. 5초마다 `RTP stats`(패킷, 손실, 디코드 프레임, 오류)를 출력합니다.
- 추적은 클래스별 IoU 매칭입니다. 외형 특징·움직임 예측이 없어 가림·교차·빠른 카메라 이동 시 ID가 바뀔 수 있습니다.
- `q`/`Esc`/창 닫기로 종료합니다. `--save-video`로 결과를 저장할 수 있습니다.

스트림 진단(다른 영상 클라이언트를 모두 닫고 단독 실행):

```powershell
python mambo_stream_probe.py --seconds 60 --transport udp --dump outputs/mambo_probe.h264
```

RTP 손실, TCP 프레이밍 오류, 타임스탬프, IDR 간격, 수신 데이터의 오프라인 디코드 결과를 따로 보고합니다.

단위 테스트: `python -m unittest discover -s tests -p "test_mambo_*.py"`

## TEST
<img width="263" height="257" alt="dji_neo" src="https://github.com/user-attachments/assets/5237759a-3490-4a77-9a54-6bc30c0c5995" />

## 시연용
<img width="632" height="791" alt="drone2" src="https://github.com/user-attachments/assets/ab988f18-5b13-4063-a3db-cdfd79f1928d" />



## 문제 해결

- `No module named 'cv2'`: Conda 환경을 활성화했는지 확인하고 OpenCV를 설치합니다.
- `attempted relative import with no known parent package`: `python neo_detect.py`로 직접 실행하지 말고 상위 폴더에서 `python -m dji_neo`를 사용합니다.
- CUDA 장치 관련 오류: CPU 환경에서는 `--device cpu`를 지정합니다. GPU 사용 시 `torch.cuda.is_available()`이 `True`인지 확인합니다.
- 비디오 소스를 열 수 없음: 웹캠 번호, 파일 경로, URL, 인증 정보 및 방화벽 상태를 확인합니다.
- 프레임을 받아올 수 없음: 영상이 끝났거나 스트림 연결이 끊어진 상태입니다.
- 탐지가 표시되지 않음: 기본 CLI는 `person`과 `vehicle`에 해당하는 결과만 표시하며 confidence 임계값은 `0.2`입니다.
- Mambo `timed out` / ping 실패: 드론 재부팅 후 PC IP가 `169.254.x.x`이면 DHCP를 못 받은 것입니다. `ipconfig /renew "Wi-Fi"` 후 다시 실행합니다. 그래도 안 되면 드론 배터리와 부팅 상태를 확인하고 Wi-Fi를 다시 연결합니다.
- Mambo `no RTP packets`: RTSP 응답은 오지만 드론이 영상을 보내지 않는 상태입니다. 드론 전원을 껐다 켜고 배터리 잔량을 확인합니다.
- DJI Neo 결과가 화면 일부만 보임: `--scrcpy-crop`이 현재 scrcpy 창 크기와 맞지 않는 경우입니다. `--select-crop`으로 다시 선택합니다.
- DJI Neo 창이 대기 중: scrcpy가 종료됐을 수 있습니다. `adb devices`에서 폰이 `device`인지, 폰 화면 잠금이 풀려 있는지 확인합니다.

## 현재 저장소의 참고 사항

- `inference.txt`의 `dji_air2s observe` 명령은 제거된 이전 CLI를 대상으로 하므로 현재 실행 방법으로 사용하면 안 됩니다.
- `requirements_air2s_windows.txt`와 `docs/simulation_evidence/`는 이전 Air2S/Tello 작업에서 남은 자료이며 현재 실시간 탐지 실행에는 관여하지 않습니다.
- 저장소에는 `__pycache__/*.pyc` 파일이 일부 추적되어 있지만 실행에 필요한 소스 파일은 아닙니다.
