import argparse
from mneme.recorded_execution import RecordedExecution
from mneme.device import get_device_arch


def main():
    # print(get_device_arch())
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

    records.link_llvm_modules()
