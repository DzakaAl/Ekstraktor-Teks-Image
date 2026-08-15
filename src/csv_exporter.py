import os
import csv
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("paddleocr-table-to-csv")


class CSVExporter:
    """Exports extracted table DataFrames to structured CSV files."""

    def export_single(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Export a single DataFrame to CSV preserving string formats and leading zeros.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Ensure all columns are string type to prevent pandas from casting zero-prefixed numbers to int/float
        df_str = df.astype(str)

        df_str.to_csv(
            output_path,
            index=False,
            encoding='utf-8-sig',
            quoting=csv.QUOTE_MINIMAL
        )
        logger.info(f"Successfully exported CSV to: {output_path}")
        return output_path

    def export_batch(
        self,
        dfs_with_sources: List[Tuple[str, pd.DataFrame]],
        output_path: str
    ) -> str:
        """
        Export multiple table DataFrames from batch processing into a single unified CSV.
        Adds 'source_file' column to indicate the originating image.
        """
        if not dfs_with_sources:
            logger.warning("No DataFrames provided for batch export.")
            empty_df = pd.DataFrame(columns=["source_file"])
            return self.export_single(empty_df, output_path)

        combined_dfs = []

        for source_filename, df in dfs_with_sources:
            if df.empty:
                continue

            df_copy = df.astype(str).copy()

            # Insert 'source_file' column at the beginning
            df_copy.insert(0, 'source_file', source_filename)
            combined_dfs.append(df_copy)

        if not combined_dfs:
            logger.warning("All batch DataFrames were empty.")
            empty_df = pd.DataFrame(columns=["source_file"])
            return self.export_single(empty_df, output_path)

        # Concatenate all DataFrames aligning by header names
        batch_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
        batch_df = batch_df.fillna("")

        return self.export_single(batch_df, output_path)
