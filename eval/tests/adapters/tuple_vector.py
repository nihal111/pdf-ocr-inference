"""Returns a tuple embedding. JSON would quietly turn it into a list, so the
evaluator has to catch it before serialising."""


def embed_pdf(path):
    return {'ocr_text': 'Hello world\n',
            'chunks': [{'id': 'chunk-000', 'embedding': (1.0, 0.0, 0.0)}]}
