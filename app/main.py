from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from models import load_text_model, generate_text


class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("password")
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError("PW must be atleast 8 characters")
        if not any(c.isdigit() for c in value):
            raise ValueError("PW must have atleast 1 digit")
        if not any(c.isupper() for c in value):
            raise ValueError("PW must contain atleast 1 uppercase letter")
        return value


app = FastAPI()
# openai_client = OpenAI(
#     base_url=os.getenv("OPENROUTER_URL"),
#     api_key=os.getenv("OPENROUTER_API_KEY")
# )


@app.get("/")
def main():
    return {"msg": "Hello from ai-space!"}


@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    pipe = load_text_model()
    response_text = generate_text(pipe, prompt)
    return response_text

# @app.get("/chat")
# def chat_controller(prompt: str = "Inspire me!"):

#     message_history = [
#         {"role": "system", "content": "You are helpful assistant.
#  Be very concise!"},
#         {"role": "user", "content": prompt}
#     ]

#     response = openai_client.responses.create(
#         model=os.getenv("OPENROUTER_MODEL"),
#         input=message_history,
#         reasoning={"effort": "none"}
#     )

#     statement = response.output_text
#     return {"statement": statement}

# if __name__ == "__main__":
#     main()
