from codeinsight.infrastructure.runtime_policy import DevelopmentPolicy


def test_development_switches_are_off_by_default() -> None:
    policy = DevelopmentPolicy.from_environment({})

    assert policy.enabled is False
    assert policy.auto_approve_changes is False
    assert policy.skip_sandbox_validation is False


def test_development_switches_can_be_enabled_for_local_debugging() -> None:
    policy = DevelopmentPolicy.from_environment(
        {
            "CODEINSIGHT_ENV": "development",
            "CODEINSIGHT_DEV_MODE": "1",
            "CODEINSIGHT_DEV_AUTO_APPROVE": "true",
            "CODEINSIGHT_DEV_SKIP_SANDBOX_VALIDATION": "on",
        }
    )

    assert policy.as_dict() == {
        "environment": "development",
        "enabled": True,
        "auto_approve_changes": True,
        "skip_sandbox_validation": True,
    }


def test_production_always_disables_development_switches() -> None:
    policy = DevelopmentPolicy.from_environment(
        {
            "CODEINSIGHT_ENV": "production",
            "CODEINSIGHT_DEV_MODE": "1",
            "CODEINSIGHT_DEV_AUTO_APPROVE": "1",
            "CODEINSIGHT_DEV_SKIP_SANDBOX_VALIDATION": "1",
        }
    )

    assert policy.enabled is False
    assert policy.auto_approve_changes is False
    assert policy.skip_sandbox_validation is False
