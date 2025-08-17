import json
import logging
from pathlib import Path
from typing import Union
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def save_to_file(
    result: Union[dict, BaseModel],
    output_dir: str,
    filename_prefix: str
) -> None:
    """
    Save a RAG response object (pydantic model or dict) to a timestamped JSON file.

    Args:
        result: The response object (must have `timestamp` attribute).
        output_dir: Path to output directory.
        filename_prefix: Prefix for the saved file name (e.g., 'naive_rag').
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(result, dict):
            result_dict = result
            timestamp = result.get("timestamp", datetime.now()).strftime('%Y%m%d_%H%M%S')
        else:
            result_dict = result.dict()
            timestamp = result.timestamp.strftime('%Y%m%d_%H%M%S')
            result_dict["timestamp"] = result.timestamp.isoformat()

        filename = f"{filename_prefix}_{timestamp}.json"
        file_path = output_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {file_path}")

    except Exception as e:
        logger.warning(f"Failed to save results: {e}")
