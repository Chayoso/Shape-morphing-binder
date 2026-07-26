"""
에피소드별 파라미터 스케줄 유틸리티

초기 안정화 → 후기 디테일 전략을 위한 piecewise linear schedule
"""

from typing import List, Tuple, Union, Optional


def parse_episode_schedule(
    schedule: List[List[Union[int, float]]],
    current_episode: int
) -> float:
    """
    Piecewise linear 스케줄에서 현재 에피소드의 값을 가져옴.
    
    Format: [[start_ep, end_ep, value], ...]
    
    Args:
        schedule: [[start, end, value], ...] 형식의 리스트
        current_episode: 현재 에피소드 번호
    
    Returns:
        현재 에피소드에 해당하는 값 (선형 보간)
    
    Example:
        >>> schedule = [[0, 5, 0.05], [5, 10, 0.03], [10, 999, 0.01]]
        >>> parse_episode_schedule(schedule, 3)  # ep 3
        0.05  # 0-5 구간
        
        >>> parse_episode_schedule(schedule, 7)  # ep 7
        0.038  # 5-10 구간, 선형 보간
        
        >>> parse_episode_schedule(schedule, 15)  # ep 15
        0.01  # 10+ 구간
    """
    if not schedule or len(schedule) == 0:
        raise ValueError("Schedule is empty")
    
    # 해당 구간 찾기
    for entry in schedule:
        start_ep, end_ep, value = entry
        if start_ep <= current_episode < end_ep:
            # 구간 내 → 다음 구간과 선형 보간 체크
            for next_entry in schedule:
                next_start, next_end, next_value = next_entry
                if next_start == end_ep:
                    # 다음 구간이 연속됨 → 선형 보간
                    progress = (current_episode - start_ep) / max(end_ep - start_ep, 1)
                    return value + (next_value - value) * progress
            
            # 다음 구간 없음 → 현재 값 반환
            return value
    
    # 마지막 구간 이후 → 마지막 값 반환
    return schedule[-1][2]


def apply_schedule_to_cfg(
    cfg: dict,
    schedule_cfg: Optional[dict],
    current_episode: int,
    verbose: bool = False
) -> dict:
    """
    Config에 정의된 스케줄을 현재 에피소드에 맞게 적용.
    
    Args:
        cfg: 원본 설정 dict (수정됨)
        schedule_cfg: 스케줄 설정 (예: cfg['schedule'])
        current_episode: 현재 에피소드
        verbose: 로그 출력 여부
    
    Returns:
        업데이트된 cfg (in-place 수정)
    
    Example:
        >>> cfg = {'lambda_lap': 0.05}
        >>> schedule = {'lambda_lap_ep': [[0, 5, 0.05], [5, 10, 0.03], [10, 999, 0.01]]}
        >>> apply_schedule_to_cfg(cfg, schedule, 7)
        {'lambda_lap': 0.038}  # 업데이트됨
    """
    if schedule_cfg is None or len(schedule_cfg) == 0:
        return cfg
    
    updated_params = []
    
    for key, schedule_list in schedule_cfg.items():
        if not key.endswith('_ep'):
            continue
        
        # 파라미터 이름 추출 (예: 'lambda_lap_ep' → 'lambda_lap')
        param_name = key[:-3]
        
        try:
            value = parse_episode_schedule(schedule_list, current_episode)
            cfg[param_name] = value
            updated_params.append(f"{param_name}={value:.4f}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to parse schedule for {param_name}: {e}")
    
    if verbose and updated_params:
        print(f"[Schedule] ep={current_episode}: " + ", ".join(updated_params))
    
    return cfg


__all__ = [
    'parse_episode_schedule',
    'apply_schedule_to_cfg',
]

