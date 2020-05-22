from pyprimer.utils.sequence import PCRPrimer, Sequence, READ_MODES
from pyprimer.utils.sequence import Sequence
from pyprimer.modules.PPC import PPC

data_dir = "/bi/aim/scratch/afrolova/COVID19/github/pyprimer/data"

if __name__ == "__main__":
    test_primer = PCRPrimer(READ_MODES.DIRECTORY)
    test_primer.DescribePrimers(f"{data_dir}/primers")
    test_sequence = Sequence(READ_MODES.FASTA)
    test_sequence.DescribeSequences(f"{data_dir}/merged.fasta")
    
    test_pcr = PPC(test_primer.dataframe, test_sequence.dataframe.sample(100))
    summary = test_pcr.analyse_primers(nCores=8, deletions=1, insertions=0, substitutions=2)
    summary.to_csv("summary_df.csv", index = False)