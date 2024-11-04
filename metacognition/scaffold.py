
class Embedding:
    def __init__(self):
        pass

    def create_embedding(self, text):
        pass


class LLM:
    def __init__(self):
        pass

    def generate_response(query):
        pass


class BugInsertionModel(LLM):
    def __init__(self):
        pass

    def generate_program_description(program_under_test: str):
        pass


class BugLibrary:
    def __init__(self, model: BugInsertionModel, embedding: Embedding, top_k: int = 5):
        self.model = model
        self.embedding = embedding
        self.library = []

    def get_relevant_bugs(self, program_under_test: str):
        # write a description of the program (in technical, operation terms)
        program_description = self.model.generate_program_description(program_under_test)

        # embed the description with openai
        program_embedding = self.embedding.create_embedding(program_description)

        # find the top k bugs using ENN and return with exemplars
        pass


class BugInsertionEnvironment:
    def __init__(self, model: LLM):
        pass
