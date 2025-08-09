import source
import argparse

try:
    import timetree  # built C++ extension via pybind11
except Exception:
    timetree = None

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")
parser.add_argument("--ttxt", type=str, default=None, help="Path to a metadata .txt file for TimeTree (optional)")
parser.add_argument("--tbin", type=str, default=None, help="Path to a binary .bin file to save/load TimeTree (optional)")
parser.add_argument("--query-ts", type=int, default=None, help="Timestamp to query in the TimeTree (optional)")
parser.add_argument("--threshold", type=int, default=1000, help="Query threshold for TimeTree (default: 1000)")

if __name__ == "__main__":
    args = parser.parse_args()

    source_instance = source.Source(args.source, force_reserialize=args.force_reserialize)

    # Optional: demonstrate TimeTree usage if extension is available and args provided
    if timetree is not None and (args.ttxt or args.tbin):
        tree = None
        if args.tbin and (args.ttxt is None):
            tree = timetree.TimeTree.load(args.tbin)
        elif args.ttxt:
            tree = timetree.TimeTree(args.ttxt)
            if args.tbin:
                try:
                    tree.save(args.tbin)
                except Exception:
                    pass
        if tree is not None:
            if args.query_ts is not None:
                res = tree.get(args.query_ts, args.threshold)
                if res:
                    print(f"TimeTree: closest to {args.query_ts} is {res['timestamp']} frameidx={res['frameidx']}")
                else:
                    print("TimeTree: no match within threshold")
            else:
                # Print quick stats
                try:
                    print(f"TimeTree: nodes={tree.nodes()} leaves={tree.leaves()} height={tree.height()}")
                except Exception:
                    pass
    elif (args.ttxt or args.tbin) and timetree is None:
        print("timetree extension not available. Build it first with: pip install -e .")