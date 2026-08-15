import unittest
import os
import tempfile
import pandas as pd
from src.csv_exporter import CSVExporter


class TestCSVExporter(unittest.TestCase):
    def test_csv_export_leading_zero(self):
        exporter = CSVExporter()
        df = pd.DataFrame({
            "No": ["1", "2"],
            "Nama": ["Budi", "Andi"],
            "Telepon": ["08123456789", "08987654321"]
        })

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            exporter.export_single(df, tmp_path)
            with open(tmp_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            self.assertIn("08123456789", content)
            self.assertIn("08987654321", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_batch_csv_export(self):
        exporter = CSVExporter()
        df1 = pd.DataFrame({
            "No": ["1"],
            "Nama": ["Budi"]
        })
        df2 = pd.DataFrame({
            "No": ["2"],
            "Nama": ["Andi"]
        })

        batch_data = [
            ("table1.jpg", df1),
            ("table2.jpg", df2)
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            exporter.export_batch(batch_data, tmp_path)
            res_df = pd.read_csv(tmp_path, dtype=str)
            self.assertIn("source_file", res_df.columns)
            self.assertEqual(list(res_df["source_file"]), ["table1.jpg", "table2.jpg"])
            self.assertEqual(list(res_df["Nama"]), ["Budi", "Andi"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
