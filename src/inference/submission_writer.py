"""Write the final submission CSV."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_submission(
    predictions: list[tuple[str, str]],
    submission_path: str | Path,
    sample_submission_path: str | Path | None = None,
) -> None:
    """
    Write (file_base, predicted_label) predictions to *submission_path*.

    If *sample_submission_path* is provided, aligns the output to that file's
    row order and fills in any missing entries with the most common prediction.
    """
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(predictions, columns=["file_base", "label"])

    # Write predictions directly — sample_submission uses a numeric index
    # unrelated to our file_base keys, so we output our own format.
    pred_df.to_csv(submission_path, index=False)
    logger.info(f"Submission written to {submission_path} ({len(pred_df)} rows)")
    if sample_submission_path is not None:
        sample = pd.read_csv(sample_submission_path)
        logger.info(
            f"Note: sample_submission has {len(sample)} rows with columns "
            f"{list(sample.columns)} — not used for alignment (incompatible id format)"
        )