#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline
import os
import argparse
from wizardlm_langchain.model import load_quantized_model

import accelerate
from langchain import PromptTemplate
from transformers import pipeline
from langchain.prompts import PromptTemplate

from wizardlm_langchain.helpers import (
    AttributeDict,
    convert_to_bytes,
    get_available_memory,
)
from wizardlm_langchain.model import load_quantized_model

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_name = os.environ.get('MODEL_NAME')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def create_gptq_llm_chain(model_path: str, model_name: str, n_ctx: str):
    args = {
        "wbits": 4,
        "groupsize": 128,
        "model_type": "llama",
        "model_dir": model_path,
    }

    model, tokenizer = load_quantized_model(model_name, args=AttributeDict(args))
    cpu_mem, gpu_mem_map = get_available_memory(convert_to_bytes("3GiB")) # TODO: tune
    print(f"Detected Memory: System={cpu_mem}, GPU(s)={gpu_mem_map}")

    max_memory = {**gpu_mem_map, "cpu": cpu_mem}

    device_map = accelerate.infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"]
    )

    model = accelerate.dispatch_model(
        model, device_map=device_map, offload_buffers=True
    )

    print(f"Memory footprint of model: {model.get_memory_footprint() / (1024 * 1024)}")

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=int(n_ctx),
        device_map=device_map,
    )

    local_llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return local_llm


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "GPTQ":
            llm = create_gptq_llm_chain(model_path=model_path, model_name=model_name, n_ctx=model_n_ctx)
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    
    prompt_template = """### Human: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    ### Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    #chain_type_kwargs = {"prompt": PROMPT, "stop": ["###"]}
    chain_type_kwargs = {"prompt": PROMPT}


    qa = RetrievalQA.from_chain_type(llm=llm, 
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     chain_type_kwargs=chain_type_kwargs, 
                                     return_source_documents= not args.hide_source)
    
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        if "###" in answer:
            answer = answer.split("###")[0]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
