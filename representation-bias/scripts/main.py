import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_embeddings
import label_viz
import column_viz
import proximity


if __name__ == "__main__":
    print("=" * 60)
    print("Representation-bias: Embedding Space Visualization")
    print("=" * 60)

    generate_embeddings.main()

    label_viz.run()

    column_viz.run()

    proximity.run()

    print("\n" + "=" * 60)
    print("Representation-bias complete.")
    print("=" * 60)
