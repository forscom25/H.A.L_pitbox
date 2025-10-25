feat(control): 조향 lock 해결을 위한 하이브리드 제동 로직 도입

기존의 'Brake on Demand' 로직은 PID 속도 제어와 분리되어 있어,
제동이 활성화될 때 스로틀이 강제로 0이 되었습니다.
이로 인해 조향 중 속도 제어가 불가능해지는 '조향 lock' 현상이 발생했습니다.

이 커밋은 문제를 해결하기 위해 두 가지 주요 사항을 수정합니다.

PID 제어 통합 (조향 Lock 해결)

longitudinal_controller_ (PID)가 0.0 ~ max_throttle 범위 대신, -max_brake ~ max_throttle의 연속적인 값을 출력하도록 변경했습니다.

PID 출력이 양수(+)이면 '스로틀'로, 음수(-)이면 '브레이크'로 매핑됩니다.

이를 통해 제동 중에도 PID가 목표 속도에 따라 미세하게 속도를 조절할 수 있게 되어, 조향 안정성을 확보합니다.

선제적 목표 속도 설정 (공격적 제동)

final_target_speed가 단순히 현재 지점의 속도만 따르지 않도록 수정했습니다.

다가오는 로컬 경로(trajectory_points_) 전체를 스캔하여, 경로 상의 가장 낮은 목표 속도 (가장 급한 커브)를 찾습니다.

이 '선제적(Proactive) 속도'와 '현재 조향각 기반 반응형(Reactive) 속도' 중 더 낮은(보수적인) 값을 최종 PID 목표 속도로 설정합니다.

이를 통해 급격한 커브 진입 전에 PID가 자연스럽게 강력한 제동(- 값)을 출력하여 선제적으로 감속합니다.

기타 수정:

formula_autonomous_system.cpp: 더 이상 사용되지 않는 enable_brake_on_demand 관련 로직을 제거했습니다.

config.yaml: enable_brake_on_demand, brake_trigger_curvature 등 불필요해진 파라미터의 정리가 필요합니다.
