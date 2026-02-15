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
from happysimulator.components.deployment.canary_deployer import (
    CanaryDeployer,
    CanaryDeployerStats,
    CanaryStage,
    CanaryState,
    ErrorRateEvaluator,
    LatencyEvaluator,
    MetricEvaluator,
)
from happysimulator.components.deployment.rolling_deployer import (
    DeploymentState,
    RollingDeployer,
    RollingDeployerStats,
)

__all__ = [
    # Auto Scaler
    "AutoScaler",
    "AutoScalerStats",
    # Canary Deployer
    "CanaryDeployer",
    "CanaryDeployerStats",
    "CanaryStage",
    "CanaryState",
    "DeploymentState",
    "ErrorRateEvaluator",
    "LatencyEvaluator",
    "MetricEvaluator",
    "QueueDepthScaling",
    # Rolling Deployer
    "RollingDeployer",
    "RollingDeployerStats",
    "ScalingEvent",
    "ScalingPolicy",
    "StepScaling",
    "TargetUtilization",
]
