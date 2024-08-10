import os
import csv
from langchain_community.llms import CTransformers
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA


def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="/content/mistral-7b-instruct-v0.1.Q2_K.gguf",
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0.3
    )
    return llm


def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path, max_ques):
    document_ques_gen, document_answer_gen = file_processing(file_path)
    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    You are an expert at creating questions based on coding materials and documentation.
    Your goal is to prepare a coder or programmer for their exam and coding tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at creating practice questions based on coding material and documentation.
    Your goal is to help a coder or programmer prepare for a coding test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt=PROMPT_QUESTIONS,
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    filtered_ques_list=[]
    while (len(filtered_ques_list)<=max_ques):
        ques = ques_gen_chain.run(document_ques_gen)
        ques_list = ques.split("\n")
        filtered_ques_list += [element for element in ques_list if element.endswith('?') or element.endswith('.')]
        print("Length   :",len(filtered_ques_list))

    llm_answer_gen = load_llm()
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list


def get_csv(file_path, max_pairs=None):
    answer_generation_chain, ques_list = llm_pipeline(file_path, max_pairs)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        # Determine the number of pairs to generate
        if max_pairs is None:
            pairs_to_generate = len(ques_list)
        else:
            pairs_to_generate = min(len(ques_list), max_pairs)

        # Generate QA pairs
        for i in range(pairs_to_generate):
            question = ques_list[i]
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])

    return output_file


file_path = "/content/your_pdf.pdf"  # Specify the path to your PDF file
output_csv = get_csv(file_path, max_pairs=10)  # Example usage with a maximum limit of 10 question-answer pairs
print("Generated QA pairs saved in CSV:", output_csv)
