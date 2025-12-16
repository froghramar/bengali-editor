from pygls.server import LanguageServer
from lsprotocol.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import unicodedata

MODEL_NAME = "bigscience/bloom-560m"

class BanglaLanguageServer(LanguageServer):
    pass

server = BanglaLanguageServer("bangla-lsp", "0.1")

# -------- Load model once --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# -------- Helpers --------
def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()

def should_autocomplete(text: str) -> bool:
    if len(text.split()) < 3:
        return False
    if text[-1] in ["।", "?", "!", "\n"]:
        return False
    return True

def predict_sentence(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    completion = generated[len(prompt):]

    if "।" in completion:
        completion = completion.split("।")[0] + "।"

    return completion.strip()

# -------- LSP Completion Handler --------
@server.feature("textDocument/completion")
def completions(params: CompletionParams):
    doc = server.workspace.get_document(params.text_document.uri)
    line = doc.lines[params.position.line]
    prefix = line[: params.position.character]

    prefix = normalize(prefix)

    if not should_autocomplete(prefix):
        return CompletionList(is_incomplete=True, items=[])

    completion_text = predict_sentence(prefix)
    if not completion_text:
        return CompletionList(is_incomplete=True, items=[])

    item = CompletionItem(
        label=completion_text,
        kind=CompletionItemKind.Text,
        insert_text=completion_text,
        detail="Bangla AI Autocomplete"
    )

    return CompletionList(
        is_incomplete=False,
        items=[item]
    )

if __name__ == "__main__":
    server.start_io()
