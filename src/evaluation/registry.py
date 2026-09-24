"""
AaroSense - MLflow Model Registry Integration (Milestone 2).

Handles automated model promotion after a model passes the evaluation gate:

  - Registers a trained model into the MLflow Model Registry.
  - Tags the version as ``Staging`` (gate passed) or ``Production``
    (explicitly promoted after further validation).
  - Archives stale versions of the same model that are superseded.
  - Provides a ``promote_to_production`` helper for manual promotion workflows.

Registry naming convention:
    ``AaroSense-<Season>-<ModelName>``
    e.g., ``AaroSense-Winter-LightGBM``
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger("AaroSense.Registry")


def _registry_model_name(season: str, model_name: str) -> str:
    """
    Build the canonical MLflow Registry model name.

    Convention: ``AaroSense-<Season>-<ModelName>``
    e.g., ``AaroSense-Winter-LightGBM``
    """
    return f"AaroSense-{season.capitalize()}-{model_name.capitalize()}"


class ModelRegistryManager:
    """
    Manages model lifecycle in the MLflow Model Registry for AaroSense.

    Responsibilities:
      - Register new model versions from existing MLflow runs.
      - Transition versions to ``Staging`` after gate passes.
      - Promote the best Staging model to ``Production``.
      - Archive superseded versions.

    Args:
        tracking_uri: MLflow tracking server URI.
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlruns/mlflow.db") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        logger.info("ModelRegistryManager initialised with URI: %s", tracking_uri)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(
        self,
        run_id: str,
        season: str,
        model_name: str,
        artifact_path: str = "sklearn_model",
    ) -> ModelVersion:
        """
        Register a model artifact from an MLflow run into the Registry.

        The artifact at ``runs:/<run_id>/<artifact_path>`` is registered
        under the canonical name and immediately transitioned to ``Staging``
        pending human or automated gate approval.

        Args:
            run_id:        MLflow run ID that produced the model artifact.
            season:        Season identifier (e.g., ``Winter``).
            model_name:    Algorithm name (e.g., ``lightgbm``).
            artifact_path: Artifact subdirectory within the run.

        Returns:
            The newly created ``ModelVersion`` object.
        """
        registry_name = _registry_model_name(season, model_name)
        model_uri = f"runs:/{run_id}/{artifact_path}"

        logger.info(
            "Registering model: URI='%s' → Registry='%s'",
            model_uri,
            registry_name,
        )

        # mlflow.register_model creates the registered model if it doesn't exist
        version: ModelVersion = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name,
            tags={
                "season": season,
                "algorithm": model_name,
                "project": "AaroSense",
            },
        )

        logger.info(
            "Registered '%s' version %s (run_id=%s).",
            registry_name,
            version.version,
            run_id,
        )
        return version

    # ------------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------------

    def transition_to_staging(
        self,
        season: str,
        model_name: str,
        version: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Transition a model version to the ``Staging`` stage.

        Staging models have passed the automated gate and are ready for
        integration and shadow-testing before production deployment.

        Args:
            season:      Season identifier.
            model_name:  Algorithm identifier.
            version:     Registry version number (string).
            description: Optional description to attach to the version.
        """
        registry_name = _registry_model_name(season, model_name)
        desc = description or f"Passed automated model gate — promoted to Staging."

        self.client.update_model_version(
            name=registry_name,
            version=version,
            description=desc,
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="stage",
            value="Staging",
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="gate_status",
            value="passed",
        )

        logger.info(
            "Transitioned '%s' v%s → Staging ✅",
            registry_name,
            version,
        )

    def transition_to_production(
        self,
        season: str,
        model_name: str,
        version: str,
        description: Optional[str] = None,
        archive_existing: bool = True,
    ) -> None:
        """
        Promote a Staging model version to ``Production``.

        Optionally archives all other versions of the same model
        (preventing stale production models from being loaded).

        Args:
            season:           Season identifier.
            model_name:       Algorithm identifier.
            version:          Registry version number to promote.
            description:      Optional promotion description.
            archive_existing: If True, archive all other non-archived versions.
        """
        registry_name = _registry_model_name(season, model_name)
        desc = description or "Manually promoted to Production after validation."

        if archive_existing:
            self._archive_other_versions(registry_name, exclude_version=version)

        self.client.update_model_version(
            name=registry_name,
            version=version,
            description=desc,
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="stage",
            value="Production",
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="gate_status",
            value="production",
        )

        logger.info(
            "Promoted '%s' v%s → Production 🚀",
            registry_name,
            version,
        )

    def reject_model(
        self,
        season: str,
        model_name: str,
        version: str,
        reason: str = "Failed evaluation gate.",
    ) -> None:
        """
        Tag a model version as rejected and archive it.

        Rejected models remain in the Registry for audit trails but are
        clearly marked to prevent accidental loading in inference.

        Args:
            season:     Season identifier.
            model_name: Algorithm identifier.
            version:    Registry version to reject.
            reason:     Human-readable rejection reason.
        """
        registry_name = _registry_model_name(season, model_name)

        self.client.update_model_version(
            name=registry_name,
            version=version,
            description=f"REJECTED: {reason}",
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="stage",
            value="Archived",
        )
        self.client.set_model_version_tag(
            name=registry_name,
            version=version,
            key="gate_status",
            value="rejected",
        )

        logger.warning(
            "Model '%s' v%s rejected and archived. Reason: %s",
            registry_name,
            version,
            reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _archive_other_versions(
        self, registry_name: str, exclude_version: str
    ) -> None:
        """Archive all versions except the one being promoted."""
        try:
            all_versions = self.client.search_model_versions(
                filter_string=f"name='{registry_name}'"
            )
            for mv in all_versions:
                if mv.version != str(exclude_version):
                    self.client.set_model_version_tag(
                        name=registry_name,
                        version=mv.version,
                        key="stage",
                        value="Archived",
                    )
                    logger.debug(
                        "Archived '%s' v%s (superseded by v%s).",
                        registry_name,
                        mv.version,
                        exclude_version,
                    )
        except Exception as exc:
            logger.warning("Could not archive old versions of '%s': %s", registry_name, exc)

    def get_latest_staging_uri(self, season: str, model_name: str) -> Optional[str]:
        """
        Retrieve the MLflow model URI for the latest Staging version.

        Args:
            season:     Season identifier.
            model_name: Algorithm identifier.

        Returns:
            Model URI string, or ``None`` if no Staging version exists.
        """
        registry_name = _registry_model_name(season, model_name)
        try:
            versions = self.client.search_model_versions(
                filter_string=f"name='{registry_name}'"
            )
            staging = [
                v for v in versions
                if v.tags.get("stage") == "Staging"
            ]
            if not staging:
                return None
            latest = max(staging, key=lambda v: int(v.version))
            return f"models:/{registry_name}/{latest.version}"
        except Exception as exc:
            logger.warning("Could not retrieve Staging URI for '%s': %s", registry_name, exc)
            return None
