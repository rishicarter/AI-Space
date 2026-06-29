from utilities import calculate_llm_price
from dataclasses import dataclass

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

@dataclass
class LLMOutput:
    output_text: str
    

class RAGBase:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS,
                 prompt_template=USER_PROMPT_TEMPLATE, course="llm-zoomcamp", model="gpt-5.4-mini"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {"question": 2.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()
    
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=query,
            context=context
        )
        return prompt.strip()

    def llm(self, prompt):
        message_history = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=message_history
        )

        return response.output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt=prompt)
        return answer
    
class RAGHW:
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS,
                 prompt_template=USER_PROMPT_TEMPLATE, course="llm-zoomcamp", model="gpt-5.4-mini"):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        # boost_dict = {"question": 2.0, "section": 0.5}
        # filter_dict = {"course": self.course}

        return self.index.search(
            query,
            # boost_dict=boost_dict,
            # filter_dict=filter_dict,
            num_results=num_results
        )
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["content"])
            # lines.append("Q: " + doc["question"])
            # lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()
    
    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=query,
            context=context
        )
        return prompt.strip()

    def llm(self, prompt):
        message_history = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=message_history,
            reasoning={"effort": "none"}
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt=prompt)
        return answer

class RAGPgVectorCW(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        
        def vec_to_str(vector):
            return "[" + ",".join(str(x) for x in vector) + "]"
        
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]

class RAGPgVector(RAGHW):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        
        def vec_to_str(vector):
            return "[" + ",".join(str(x) for x in vector) + "]"
        
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]