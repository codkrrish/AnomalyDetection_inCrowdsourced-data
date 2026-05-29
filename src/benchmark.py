# src/benchmark.py

from src.config import BENCHMARKS


class BenchmarkValidator:

    def __init__(self, df):
        self.df = df.copy()

    def validate(self):
        """
        Flag entries whose baseSalary_USD falls outside known industry
        benchmarks for their (jobFamily_clean, level_group, region) tuple.
        """
        flags = []

        for _, row in self.df.iterrows():
            key = (
                row["jobFamily_clean"],
                row["level_group"],
                row["region"],
            )
            salary = row["baseSalary_USD"]

            if key in BENCHMARKS:
                low = BENCHMARKS[key]["min"]
                high = BENCHMARKS[key]["max"]
                flags.append(salary < low or salary > high)
            else:
                flags.append(False)

        self.df["flag_benchmark"] = flags

        return self.df
