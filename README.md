feat(planning, control): 전역 경로 생성 및 제동 로직 개선

주요 변경 사항:

전역 경로 생성 간격 파라미터화:

기존에 사용되지 않던 racing_mode.waypoint_spacing 파라미터를 전역 경로(Global Path)의 샘플링 간격으로 재활용합니다. (generateGlobalPath 함수)

이를 통해 웨이포인트 밀도를 조절하여 목표 속도의 잦은 변경 문제를 완화하고, 주행 안정성을 향상시킵니다.

config.yaml의 해당 파라미터 주석을 수정하여 용도를 명확히 했습니다.

'요청 시 제동(Brake on Demand)' 로직 도입:

가속(PID)과 제동(조건부 로직) 제어를 분리하여, 불필요한 감속을 줄이고 에너지 효율성을 높입니다.

PID 제어기는 이제 가속(Throttle > 0)만 담당하며, 목표 속도 초과 시 Coasting(관성 주행)을 유도합니다.

브레이크는 racing_mode에서만 작동하며, 전방 경로의 곡률이 brake_trigger_curvature 임계값을 넘고 현재 속도가 해당 커브 목표 속도보다 빠를 때만 brake_application_gain에 따라 개입합니다.

관련 파라미터 (enable_brake_on_demand, trigger_curvature, application_gain)를 config.yaml의 control.racing_mode.BrakeLogic 섹션에 추가하고 C++ 구조체(ControlParams) 및 로직(run 함수)을 업데이트했습니다.

조건부 복잡도 로직 실행:

generateGlobalPath 함수를 수정하여 planning.trajectory_generation.racing_mode.complexity_logic.enable 파라미터 값에 따라 속도 프로파일링 방식을 선택할 수 있도록 변경했습니다.

enable: true (기본값): 기존의 복잡도 점수 기반 속도 계산 수행

enable: false: 단순 곡률 기반 속도 계산 수행 (브레이크 기능 강화에 따른 옵션)
