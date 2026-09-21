# DJI Neo + Android 스마트폰 + Windows 추론 PC

이 프로젝트의 기본 입력은 `scrcpy`로 Windows에 미러링한 **Android DJI Fly 화면**입니다.
따라서 RC-N3가 없어도 다음 구성이 가능합니다.

```text
DJI Neo ──(Wi-Fi)──> Android 스마트폰(DJI Fly로 비행/조종)
                         │
                         └──(USB + ADB)──> Windows 추론 PC
                                               │
                                      scrcpy 창 캡처 → FFCA-YOLO
```

중요한 점은 PC가 Neo를 직접 조종하거나 DJI의 기체 스트림을 직접 받는 구조가 아니라는 것입니다.
스마트폰이 DJI Fly에서 Neo를 계속 조종하고, PC는 스마트폰 화면을 읽어 추론합니다. `scrcpy`는 이
실행 경로에서 `--no-control`로 시작하므로 PC 마우스/키보드 입력이 스마트폰으로 전달되지 않습니다.

## 가능 여부와 제한

- **스마트폰 조종:** 가능. DJI Neo의 Mobile App Control은 DJI Fly와 Wi-Fi로 연결하고 앱의 가상
  조이스틱으로 수동 조종합니다.
- **PC 화면 수신:** 가능. Android USB 디버깅과 `scrcpy`를 사용해 DJI Fly 화면을 Windows에
  표시하고, 이 프로그램이 해당 창의 client area를 캡처합니다.
- **추론 영상:** DJI Fly 화면 전체(비행 UI, 지도, 버튼 포함)를 캡처합니다. 카메라 영역만 쓰려면
  `--select-crop`으로 한 번 선택하거나 `--scrcpy-crop X Y WIDTH HEIGHT`를 지정하십시오.
- **직접 기체 API/원본 영상:** 이 경로에서는 제공하지 않습니다. 원본 O4 영상이나 텔레메트리를
  DJI Neo에서 PC로 직접 받는 공식 PC API로 사용하지 않습니다.
- **비행 안전:** Mobile App Control은 RC-N3와 동작 조건이 다릅니다. 시험 비행은 시야 확보,
  개방 공간, 낮은 고도에서 하고 DJI Fly에 표시되는 제한/경고를 따르십시오.

## 1. Windows PC 준비

PowerShell에서 프로젝트 폴더를 연 뒤 Python 의존성을 설치합니다.

```powershell
python -m pip install -r requirements_fixed.txt
```

`requirements_fixed.txt`는 현재 저장소에 있는 고정 버전 목록입니다. PyTorch/CUDA는 PC의
CUDA 버전에 맞는 기존 프로젝트 환경을 유지하십시오. 이미 모델 추론 환경이 준비되어 있다면
`python -m pip install mss==10.2.0`만 추가해도 됩니다.

그 다음 공식 scrcpy 저장소의 Windows 배포판을 설치하고 PATH를 확인합니다.

```powershell
winget install --exact Genymobile.scrcpy
scrcpy --version
adb version
```

`winget`을 사용할 수 없으면 [scrcpy 공식 Windows 문서](https://github.com/Genymobile/scrcpy/blob/master/doc/windows.md)의
배포 파일을 설치한 뒤 `--scrcpy-path C:\path\to\scrcpy.exe`로 실행 파일을 직접 지정할 수 있습니다.

## 2. Android 스마트폰 준비

1. Android에서 **개발자 옵션**을 활성화하고 **USB 디버깅**을 켭니다.
2. USB 데이터 케이블로 스마트폰과 Windows PC를 연결합니다(충전 전용 케이블은 안 됩니다).
3. 스마트폰에 표시되는 “USB 디버깅을 허용하시겠습니까?”에서 허용합니다.
4. PC에서 아래 명령으로 상태가 `device`인지 확인합니다.

```powershell
adb devices
```

여러 Android 장치가 보이면 `adb devices`에 표시된 serial을 아래 실행 명령의
`--scrcpy-serial SERIAL`에 넣습니다.

## 3. Neo를 스마트폰으로 연결/조종

1. Neo의 전원을 켭니다.
2. 스마트폰의 Bluetooth, Wi-Fi, 위치 서비스를 켭니다.
3. DJI Fly에서 `Connection Guide` → `DJI Neo` → `Connect via Mobile Device`를 선택합니다.
4. `GO FLY` 화면의 `Controls`에서 `Manual Control`을 선택합니다.
5. 비행과 카메라 조작은 스마트폰의 DJI Fly에서 수행합니다.

## 4. 추론 실행

먼저 아래처럼 실행합니다. 프로그램이 `DJI Neo Phone (scrcpy)`라는 창을 띄우고, 그 화면을
캡처해 추론 결과를 별도 OpenCV 창에 표시합니다.

```powershell
python dji_neo_camera_inference.py `
  --weights "ffca_yolo\weights\best.pt" `
  --input-mode scrcpy `
  --device 0 `
  --half `
  --select-crop `
  --view-img
```

`--device cpu`를 사용하면 CUDA 없이도 실행할 수 있습니다. GPU가 없거나 PyTorch가 CUDA를
인식하지 못하면 `--half`를 빼십시오.

처음 `--select-crop`을 사용하면 첫 Android 화면에서 DJI Fly의 **카메라 영상 부분만** 드래그하고
Enter/Space를 누릅니다. 프로그램이 선택 좌표를 로그에 출력하므로 다음 실행부터는 다음처럼
자동 지정할 수 있습니다.

```powershell
python dji_neo_camera_inference.py `
  --weights "ffca_yolo\weights\best.pt" `
  --input-mode scrcpy `
  --scrcpy-crop 0 120 960 540 `
  --device 0 `
  --half `
  --view-img
```

화면 비율/해상도는 스마트폰과 DJI Fly 버전에 따라 달라지므로 위 crop 숫자는 예시입니다.
`--scrcpy-max-size 1280`처럼 미러링 해상도를 높이면 작은 객체 탐지에 유리하지만 캡처/추론 부하가
증가합니다. 기본값은 `960`, `30 FPS`입니다.

### 이미 scrcpy를 직접 실행한 경우

프로그램이 scrcpy를 새로 띄우지 않고 기존 창을 캡처하게 할 수도 있습니다.

```powershell
scrcpy --no-control --no-audio --window-title="DJI Neo Phone (scrcpy)"
python dji_neo_camera_inference.py `
  --weights "ffca_yolo\weights\best.pt" `
  --input-mode scrcpy `
  --scrcpy-no-launch `
  --view-img
```

## 자주 발생하는 문제

- `scrcpy was not found`: `scrcpy --version`이 동작하는지 확인하거나 `--scrcpy-path`에
  `scrcpy.exe` 절대 경로를 지정합니다.
- `adb devices`가 `unauthorized`: 스마트폰 잠금을 해제하고 USB 디버깅 승인 대화상자를 다시
  허용한 뒤 `adb kill-server; adb start-server`를 실행합니다.
- `Waiting for scrcpy window`: 창 제목이 맞는지 확인하고, 기존 창을 연결할 때는
  `--scrcpy-no-launch`를 사용하지 않은 상태로 먼저 자동 실행을 시도합니다.
- `The scrcpy window is minimized`: 창을 복원하십시오. 캡처 대상 창은 최소화하면 안 됩니다.
- 화면이 UI만 보이거나 카메라 영역이 잘못됨: `--select-crop`을 다시 실행해 DJI Fly의
  live-view 영역만 선택합니다. `--scrcpy-allow-occlusion`을 사용하면 다른 창이 scrcpy를
  가릴 수 있으므로 기본값(항상 위)을 권장합니다.
- 첫 프레임 타임아웃: DJI Fly에서 Neo 연결과 `GO FLY`/실시간 화면을 먼저 확인한 뒤 케이블,
  USB 디버깅, 화면 잠금 상태를 확인합니다.

## 참고: RC-N3 + RTMP 경로

RC-N3가 있고 DJI Fly 버전/기체 조합에서 RTMP 라이브 스트리밍 메뉴가 제공될 때만 아래 경로를
사용할 수 있습니다. RC-N3가 없는 현재 구성에서는 `scrcpy` 경로를 사용하십시오.

```powershell
python dji_neo_camera_inference.py `
  --weights "ffca_yolo\weights\best.pt" `
  --input-mode listen `
  --rtmp-listen-url rtmp://0.0.0.0:1935/live/neo `
  --device 0 `
  --half `
  --view-img
```

이 경우에만 DJI Fly의 RTMP 주소에 `rtmp://<PC_LAN_IP>:1935/live/neo`를 입력합니다. RTMP
수신 포트는 인터넷에 노출하지 말고 신뢰할 수 있는 LAN으로 제한하십시오.
