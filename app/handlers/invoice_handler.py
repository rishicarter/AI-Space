"""
Invoice module handler: Get invoice details from IMG/PDF
"""
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

from pdf2image import convert_from_path
import easyocr
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

class InvoiceModel(BaseModel):
    invoice_number: str = Field(description="Invoice number")
    invoice_date: str = Field(description="Invoice date")
    invoice_items: list = Field(description="List of items identified from invoice")

def data_handler(file_path: str):
    with open(file_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        return pdf_base64
    return None

def invoice_read(file_path: str):
    data = data_handler(file_path)
    if data is None:
        print("Data conversion issue")
        return
    llm = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL"),
        temperature=0.3,
        reasoning={"effort": "none"}
    )
    # msgs = [
    #     SystemMessage(content="You are Warehouse operator. Your job is to read the invoices and extract the information in the expected format provided. If the required information is not present then keep the fields empty."),
    #     HumanMessage(content=[
    #         {"type": "text", "text": "Identify the details from the document."},
    #         {"type": "file", "base64": f"{data}", "mime_type": "application/pdf"}
    #     ])
    # ]
    msgs = [HumanMessage(
        content=[
            {"type": "text", "text": "Extract the text from the Image. Only output the text from the Image."},
            {
                "type": "image",
                "base64": data,
                "mime_type": "image/jpeg",
            },
        ]
    )]

    chain = llm.with_structured_output(schema=InvoiceModel, method="json_schema", strict=True)

    res = llm.invoke(msgs)

    if res:
        print(dict(res))

def openai_read(file_path):
    llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0.1,
            reasoning_effort="minimal",
            api_key=os.getenv("OPENAI_API_KEY") #ENV.OPENAI_API_KEY
        )
    # image_bytes = await image_file.download_as_bytearray()
    # base64_string = base64.b64encode(image_bytes).decode("utf-8")
    base64_string = data_handler(file_path)
    msgs = [
        # SystemMessage("Extract the text from the Image. Only output the text from the Image."),
        HumanMessage(content=[
            {"type": "text", "text": "Extract the text from the Image. Only output the text from the Image."},
            {"type": "image", "base64": base64_string, "mime_type": "image/jpeg"}
        ])
    ]
    res = llm.invoke(msgs)
    print(res)

def pdf_reader_with_easyocr(file_loc: str):
    reader = easyocr.Reader(['en'])

    pages = convert_from_path(file_loc)

    all_text = ""

    for _, page in enumerate(pages):

        raw_page = BytesIO()
        page.save(raw_page, format="JPEG")

        results = reader.readtext(raw_page.getvalue())
        for result in results:
            all_text += result[1] + "\n"

    print(all_text)
if __name__ == "__main__":
    # FILE_PATH = "/home/rishicarter/git/AI-Space/app/handlers/invoice1.pdf"
    FILE_PATH = "/home/rishicarter/git/AI-Space/app/handlers/sample.jpeg"
    invoice_read(file_path=FILE_PATH)
    # openai_read(file_path=FILE_PATH)