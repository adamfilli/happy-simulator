"""Deployment components for auto-scaling and deployment strategies."""

from happysimulator.components.deployment.auto_scaler import (
    AutoScaler,
    AutoScalerStats,
    QueueDepthScaling,
    ScalingEvent,
    ScalingPolicy,
    StepScaling,
    TargetUtilization,
)
from happysimulator.components.deployment.rolling_deployer import (
    DeploymentState,
    RollingDeployer,
    RollingDeployerStats,
)
from happysimulator.components.deployment.canary_deployer import (
    CanaryDeployer,
    CanaryDeployerStats,
    CanaryStage,
    CanaryState,
    ErrorRateEvaluator,
    LatencyEvaluator,
    MetricEvaluator,
)

__all__ = [
    # Auto Scaler
    "AutoScaler",
    "AutoScalerStats",
    "ScalingPolicy",
    "TargetUtilization",
    "StepScaling",
    "QueueDepthScaling",
    "ScalingEvent",
    # Rolling Deployer
    "RollingDeployer",
    "RollingDeployerStats",
    "DeploymentState",
    # Canary Deployer
    "CanaryDeployer",
    "CanaryDeployerStats",
    "CanaryStage",
    "CanaryState",
    "MetricEvaluator",
    "ErrorRateEvaluator",
    "LatencyEvaluator",
]
