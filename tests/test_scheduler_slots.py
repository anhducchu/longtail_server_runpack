from runner.gpu import GPU, plan_slots


def make_gpu(index, free_mb):
    return GPU(index=index, name="mock", memory_free_mb=free_mb)


def test_plan_slots_auto_small_gpu_gets_one_slot():
    gpus = [make_gpu(0, 4 * 1024)]  # 4 GB free: below headroom + one task's estimate
    slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12, headroom_gb=4, ceiling=4)
    assert slots == {0: 1}


def test_plan_slots_auto_large_gpu_hits_ceiling():
    gpus = [make_gpu(0, 80 * 1024)]  # 80 GB free: math says 6 slots, capped at the ceiling
    slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12, headroom_gb=4, ceiling=4)
    assert slots == {0: 4}


def test_plan_slots_explicit_override_ignores_memory():
    gpus = [make_gpu(0, 4 * 1024)]
    slots = plan_slots(gpus, tasks_per_gpu=2, task_memory_gb=12, headroom_gb=4, ceiling=4)
    assert slots == {0: 2}


def test_plan_slots_never_below_one():
    gpus = [make_gpu(0, 512)]  # essentially no free memory
    slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12, headroom_gb=4, ceiling=4)
    assert slots == {0: 1}


def test_plan_slots_is_per_gpu():
    gpus = [make_gpu(0, 4 * 1024), make_gpu(1, 80 * 1024)]
    slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12, headroom_gb=4, ceiling=4)
    assert slots == {0: 1, 1: 4}


def test_plan_slots_capacity_is_stable_as_our_own_tasks_fill_the_gpu():
    """An 80 GB card must still report 4 slots once 3 of our tasks are resident.

    Measuring raw free memory would shrink the baseline as we fill the card and
    stall the GPU one slot below its real capacity, never reaching the ceiling.
    """
    for running in range(0, 5):
        free_gb = 80 - running * 12
        gpus = [make_gpu(0, free_gb * 1024)]
        slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12,
                           headroom_gb=4, ceiling=4, running_on={0: running})
        assert slots == {0: 4}, f"running={running} free={free_gb}GB gave {slots}"


def test_plan_slots_does_not_reclaim_a_foreign_tenants_memory():
    # 40 GB card, 20 GB held by someone else, none of it ours to add back.
    gpus = [make_gpu(0, 20 * 1024)]
    slots = plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12,
                       headroom_gb=4, ceiling=4, running_on={0: 0})
    assert slots == {0: 1}


def test_plan_slots_running_on_defaults_to_empty():
    gpus = [make_gpu(0, 80 * 1024)]
    assert plan_slots(gpus, tasks_per_gpu="auto", task_memory_gb=12,
                      headroom_gb=4, ceiling=4) == {0: 4}
