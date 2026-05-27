"""Budget policy with max cycle protection to prevent infinite loops."""

import time


class BudgetPolicy:
    def __init__(self, max_cycles: int = 20, max_oracle_calls: int = 1000, max_wall_time_s: int = 3600):
        self.max_cycles = max_cycles
        self.max_oracle_calls = max_oracle_calls
        self.max_wall_time_s = max_wall_time_s
        self.cycle_count = 0
        self.oracle_calls = 0
        self.start_time = None

    def can_continue(self) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
        if self.cycle_count >= self.max_cycles:
            return False
        if self.oracle_calls >= self.max_oracle_calls:
            return False
        if time.time() - self.start_time > self.max_wall_time_s:
            return False
        return True

    def record_cycle(self) -> None:
        self.cycle_count += 1

    def record_oracle_call(self) -> None:
        self.oracle_calls += 1
