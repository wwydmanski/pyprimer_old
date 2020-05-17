from utils.sequence import PCRPrimer
from utils.sequence import Sequence
from benchmark import PCR

if __name__ == "__main__":
    test_primer = PCRPrimer("directory")
    test_primer.DescribePrimers("./data/primers")
    test_sequence = Sequence("fasta")
    test_sequence.DescribeSequences("./data/merged_short.fasta")
    
    test_pcr = PCR(test_primer.dataframe, test_sequence.dataframe)
    test_pcr.analyse_primers(nCores=2)
    test_pcr.summary.to_csv("summary_df.csv", index = False)