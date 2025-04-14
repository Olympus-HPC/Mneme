import argparse
from mneme.recorded_execution import RecordedExecution


def main():
    parser = argparse.ArgumentParser(description="Mneme auto-tuning tool")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON file containing the record DB of a kernel",
    )
    args = parser.parse_args()
    records = RecordedExecution.from_json(args.input)
    for k, v in records.items():
        print(k, v)
    print(records)
