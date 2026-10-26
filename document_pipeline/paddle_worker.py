"""Private inherited-pipe entry point for the isolated Paddle runtime."""
from __future__ import annotations

import argparse
from multiprocessing.connection import Connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, type=int)
    args = parser.parse_args()
    connection = Connection(args.fd)
    request = connection.recv()
    from document_pipeline.config import load_config
    from document_pipeline.page_parallel import _worker_main
    cfg = load_config({"ocr": request["config"]}).ocr
    _worker_main(connection, cfg, request["worker"], request["parent_pid"])


if __name__ == "__main__":
    main()
