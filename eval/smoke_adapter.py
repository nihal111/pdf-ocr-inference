"""Dummy adapter for testing the evaluation tools. It is not a pipeline.

It only recognises eval/fixtures/synthetic.pdf and returns a hard-coded result.
"""
from pathlib import Path


def embed_pdf(path: str) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if Path(path).name != 'synthetic.pdf':
        raise NotImplementedError('The smoke adapter only handles synthetic.pdf')
    return {'ocr_text': 'Hello world\n',
            'chunks': [{'id': 'chunk-000', 'embedding': [1.0, 0.0, 0.0]}]}
