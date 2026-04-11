"""Write the final submission CSV."""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

ID_DIGITS = 4  # zero-padding width → 0001, 0002, …


def write_submission(
    predictions: list[tuple[str, str]],
    submission_path: str | Path,
    sample_submission_path: str | Path | None = None,
) -> None:
    """
    Write predictions to *submission_path* in the required (id, genre) format.

    Args:
        predictions:            list of (file_base, predicted_label) tuples
                                produced by run_inference().
        submission_path:        where to write the final CSV.
        sample_submission_path: optional path to sample_submission.csv —
                                used only to verify row count alignment.
    """
    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output dataframe: drop file_base, add zero-padded sequential id
    out_df = pd.DataFrame({
        "id":    [str(i).zfill(ID_DIGITS) for i in range(1, len(predictions) + 1)],
        "genre": [label for _, label in predictions],
    })

    # Optional: verify alignment with sample_submission
    if sample_submission_path is not None:
        sample = pd.read_csv(sample_submission_path)
        if len(sample) != len(out_df):
            logger.warning(
                f"Row count mismatch: sample_submission has {len(sample)} rows "
                f"but we have {len(out_df)} predictions."
            )
        else:
            logger.info(f"Row count matches sample_submission: {len(out_df)} rows ✓")

    out_df.to_csv(submission_path, index=False)
    logger.info(f"Submission written to {submission_path} ({len(out_df)} rows)")
    logger.info(f"ID range: {out_df['id'].iloc[0]} → {out_df['id'].iloc[-1]}")
    logger.info(f"Genre distribution:\n{out_df['genre'].value_counts().to_string()}")