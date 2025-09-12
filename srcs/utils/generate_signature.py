from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

from srcs.utils.common import get_logger

logger = get_logger(__name__)


class SignatureGenerator:
    """Generates SHA1 signature for the artifacts directory."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def _calculate_sha1(self, file_path: Path) -> str:
        sha1_hash = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha1_hash.update(chunk)
        return sha1_hash.hexdigest()

    def _create_artifacts_zip(self, output_zip: Path) -> None:
        logger.info(f"Creating zip of artifacts directory: {self.artifacts_dir}")

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(self.artifacts_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.artifacts_dir)
                    zipf.write(file_path, arcname)

    def generate(
        self, output_file: Optional[Path] = None, zip_file: Optional[Path] = None
    ) -> bool:
        if not self.artifacts_dir.exists():
            logger.error(f"Artifacts directory not found: {self.artifacts_dir}")
            return False

        if output_file is None:
            output_file = Path(__file__).parent.parent.parent / "signature.txt"

        if zip_file is None:
            zip_file = Path(__file__).parent.parent.parent / "artifacts.zip"

        try:
            self._create_artifacts_zip(zip_file)

            logger.info("Calculating SHA1 hash...")
            sha1_hash = self._calculate_sha1(zip_file)

            with open(output_file, "w") as f:
                f.write(f"{sha1_hash}\n")

            logger.info("Signature generated successfully!")
            logger.info(f"SHA1 hash: {sha1_hash}")
            logger.info(f"Signature saved to: {output_file}")
            logger.info(f"Zip file created at: {zip_file}")

            return True

        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return False


def main() -> int:
    script_dir = Path(__file__).parent.parent.parent
    artifacts_dir = script_dir / "artifacts"

    generator = SignatureGenerator(artifacts_dir)

    if generator.generate():
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
