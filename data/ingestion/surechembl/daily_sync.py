"""SureChEMBL daily incremental patent sync.

Downloads new patents from SureChEMBL FTP/API and indexes them
for the FTO (Freedom-to-Operate) search engine.

Status: PLACEHOLDER — requires SureChEMBL access credentials.
To implement: obtain credentials from https://www.surechembl.org/
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def sync_patents(
    output_dir: str | Path,
    *,
    days_back: int = 1,
    api_key: str | None = None,
) -> dict:
    """Sync new patents from SureChEMBL.

    Args:
        output_dir: Directory to store downloaded patent data
        days_back: Number of days to sync (1 = today only)
        api_key: SureChEMBL API key (if None, uses env var SURECHEMBL_API_KEY)

    Returns:
        dict with keys: n_new, n_updated, errors
    """
    logger.warning(
        "SureChEMBL sync is a placeholder. Requires valid API credentials."
    )
    # When implemented:
    # 1. Connect to SureChEMBL FTP/API
    # 2. Download delta since last sync marker
    # 3. Parse patent XML (USPTO, EPO, WIPO formats)
    # 4. Extract chemical structures (Markush + specific)
    # 5. Store in data/processed/patents/{YYYY-MM-DD}/

    return {
        "n_new": 0,
        "n_updated": 0,
        "errors": ["PLACEHOLDER: SureChEMBL credentials not configured"],
    }
