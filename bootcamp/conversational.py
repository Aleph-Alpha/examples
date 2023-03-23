# Lets try building a conversational bot from scratch here
# for that we need all of the previously used functionalities
from aleph_alpha_client import (
    Client,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
    SemanticRepresentation,
    Prompt,
    CompletionRequest,
    CompletionResponse,
)
from scipy import spatial
import numpy as np
import os
from dotenv import load_dotenv


class ConversationalAgent:
    def __init__(self, dataset: list[str], mode: str = "smalltalk"):
        self.client = Client(token=os.getenv("AA_TOKEN"))
        self.dataset = dataset
        self.embeddings = self.get_embeddings(self.dataset)
        self.mode = mode
        self.history = []

    def get_embeddings(self, dataset):
        # TODO This function will return the embeddings of the dataset as a list

        for text in dataset:
            # Hint: Use the client to get the embeddings of the dataset
            request = SemanticEmbeddingRequest()
            result = self.client.semantic_embedding(request, model="luminous-base")
            self.embeddings.append(None)

        return None

    def search(self, query):
        # TODO This function will return the most similar sentence in the dataset to the query

        # Hint: Use the cosine similarity function to find the most similar sentence
        # first embed the query using the client

        # then iterate over the dataset and find the most similar text

        # return the most similar text

        return None

    def generate_smalltalk(self, query):
        # TODO This function will return a smalltalk response to the query

        # Hint: Use the client to generate a smalltalk response

        samlltalk_prompt = Prompt.from_text(f"User: {query}\nBot:")

        request = CompletionRequest(
            prompt=samlltalk_prompt,
            maximum_tokens=100,
            temperature=0.7,
            stop_sequences=["\n"],
        )

        result = self.client.complete(request=request, model="luminous-extended")

        return result.completions[0].completion

    def generate_qa(self, query):
        # TODO This function will return a qa response to the query

        # Hint: First search for the correct text, then use the client to generate a qa response

        return None

    def converse(self, query):
        # TODO This function will return a response to the query

        # Hint: Use the mode to decide which function to call

        if self.mode == "smalltalk":
            return self.generate_smalltalk(query)

        elif self.mode == "qa":
            return self.generate_qa(query)

        else:
            return "Invalid mode"


dataset = [
    "Germany : Germany (, ), officially the Federal Republic of Germany, is a country in Central Europe. It is the second most populous country in Europe after Russia, and the most populous member state of the European Union. Germany is situated between the Baltic and North seas to the north, and the Alps to the south; it covers an area of , with a population of over 83 million within its 16 constituent states. Germany borders Denmark to the north, Poland and the Czech Republic to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. The nation's capital and largest city is Berlin, and its financial centre is Frankfurt; the largest urban area is the Ruhr.",
    "Bristol : Bristol () is a city, ceremonial county and unitary authority in England. Situated on the River Avon, it is bordered by the ceremonial counties of Gloucestershire, to the north; and Somerset, to the south. Bristol is the most populous city in South West England.\nThe wider Bristol Built-up Area is the eleventh most populous urban area in the United Kingdom.",
    "Heidelberg : Heidelberg () is a university town in the German state of Baden-W\u00fcrttemberg, situated on the river Neckar in south-west Germany. In the 2016 census, its population was 159,914, of which roughly a quarter consisted of students.",
    "France : France (), officially the French Republic (), is a transcontinental country spanning Western Europe and overseas regions and territories in the Americas and the Atlantic, Pacific and Indian Oceans. Its metropolitan area extends from the Rhine to the Atlantic Ocean and from the Mediterranean Sea to the English Channel and the North Sea; overseas territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic, the French West Indies, and many islands in Oceania and the Indian Ocean. Due to its several coastal territories, France has the largest exclusive economic zone in the world. France borders Belgium, Luxembourg, Germany, Switzerland, Monaco, Italy, Andorra and Spain in Europe, as well as the Netherlands, Suriname and Brazil in the Americas. Its eighteen integral regions (five of which are overseas) span a combined area of  and over 67 million people (). France is a unitary semi-presidential republic with its capital in Paris, the country's largest city and main cultural and commercial centre; other major urban areas include Marseille, Lyon, Toulouse, Lille, Bordeaux, and Nice.",
]
agent = ConversationalAgent(dataset, mode="qa")

agent.generate_smalltalk("Hello")
