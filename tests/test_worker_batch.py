from runner.config import Task
from runner.worker import resolve_batch_size


def make_task(metric_protocol, batch_size=64, oom_sizes=(128, 96, 64, 48, 32)):
    return Task(
        id="x", campaign="c", stage="s", adapter="ccua", method="ddpm", seed=0, priority=1,
        dataset={}, train={"batch_size": batch_size}, eval={"metric_protocol": metric_protocol},
        method_config={}, repository={}, runtime={}, retry={"oom_batch_sizes": list(oom_sizes)},
    )


def test_native_contract_never_changes_batch_on_retry():
    task = make_task("native_cifar_v1")
    assert resolve_batch_size(task, attempt=1) is None
    assert resolve_batch_size(task, attempt=2) is None
    assert resolve_batch_size(task, attempt=3) is None


def test_non_contract_first_attempt_uses_default():
    task = make_task("other_protocol")
    assert resolve_batch_size(task, attempt=1) is None


def test_non_contract_retry_clamps_to_configured_batch():
    task = make_task("other_protocol", batch_size=64)
    # oom_batch_sizes[1] == 96, larger than the campaign's own batch of 64:
    # a retry must never grow past the fairness contract's own budget.
    assert resolve_batch_size(task, attempt=2) == 64


def test_non_contract_retry_can_shrink_batch():
    task = make_task("other_protocol", batch_size=128)
    # oom_batch_sizes[1] == 96, below the configured batch: use it as-is.
    assert resolve_batch_size(task, attempt=2) == 96
