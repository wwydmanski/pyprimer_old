from pyprimer.modules.PPC import PPC
from pyprimer.utils.sequence import PCRPrimer
from pyprimer.utils.sequence import Sequence
import pandas as pd

data_dir = "/bi/aim/scratch/afrolova/COVID19/github/pyprimer/data"

def test_PPC_calculation():
    test_primer = PCRPrimer("directory")
    test_primer.DescribePrimers(f"{data_dir}/primers")
    test_sequence = Sequence("fasta")
    test_sequence.DescribeSequences(f"{data_dir}/merged.fasta")
    
    test_pcr = PPC(test_primer.dataframe, test_sequence.dataframe.sample(100, random_state=42))
    test_pcr.analyse_primers(nCores=8, deletions=1, insertions=0, substitutions=2)
    test_pcr.summary.to_csv("tests/test_summary_df.csv", index = False)

    assert pd.read_csv("tests/test_summary_df.csv").equals(pd.read_csv("tests/goal_summary_df.csv"))