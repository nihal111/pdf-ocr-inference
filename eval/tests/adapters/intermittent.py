from eval.smoke_adapter import embed_pdf as good
calls = 0
def embed_pdf(path):
    global calls
    calls += 1
    value = good(path)
    if calls == 1:
        value['ocr_text'] = 'WRONG'
    return value
