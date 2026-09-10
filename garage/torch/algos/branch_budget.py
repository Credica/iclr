"""Remaining sequence budget, in the caller's explicit step clock."""


def remaining_branch_steps(checkpoint, task_count, steps_per_task, task_step=0):
    start_task = int(checkpoint['seq_idx']) + (0 if task_step > 0 else 1)
    if not 0 <= start_task < task_count or not 0 <= task_step < steps_per_task:
        raise ValueError('Branch task index or local progress exceeds the sequence budget')
    return (task_count - start_task) * steps_per_task - task_step
