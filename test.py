import textract
text = textract.process("docs/image.png")

print(text)