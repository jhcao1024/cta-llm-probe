import retrieval_baselines as retrieval_baselines_stage
import multi_col_retrieval_coverage as multi_column_coverage

def main():
    print("=" * 60)
    print("RETRIEVAL-DIAGNOSTICS")
    print("=" * 60)

    retrieval_baselines_stage.run()
    print("-" * 60)
    multi_column_coverage.run()

    print("\n[Done] All results saved in retrieval-diagnostics/outputs/")

if __name__ == "__main__":
    main()
