"""
Utility for the Long Form Question Answering Model.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from datasets import load_dataset
import torch
from tqdm import tqdm
import time
import numpy as np
import re
import faiss
import os
# Code modified from the HuggingFace blog post by yjernite at
# https://yjernite.github.io/lfqa.html#generation
# And the associated utility functions at
# https://github.com/huggingface/notebooks/blob/main/longform-qa/lfqa_utils.py


class longFormQA:
    def __init__(self, ds_path, encode_length=512):
        pattern = r".+?/{0,1}([A-Za-z0-9_-]+)\.csv"
        match = re.search(pattern, ds_path)
        if match:
            self.ds_name = match.group(1)
        else:
            self.ds_name = ds_path.rsplit(".", 1)[0]
        self.encode_length = encode_length
        self.ds = load_dataset("csv", data_files=ds_path, sep="|")["train"]
        self.qdim = 128

        # Right now, we only have this implemented for GPU
        if not torch.cuda.is_available():
            raise NotImplementedError("There isn't support for CPU yet.")


    def load_model_weights(self, qa_model_name, s2s_model_name):
        """ Load the models and their weights from HuggingFace """
        t0 = time.time()

        # Question-Answering Model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_embedder = AutoModel.from_pretrained(qa_model_name
                          ).to("cuda")
        _ = self.qa_embedder.eval()

        # Sequence to sequence
        self.s2s_tokenizer = AutoTokenizer.from_pretrained(s2s_model_name)
        self.s2s_model = AutoModelForSeq2SeqLM.from_pretrained(s2s_model_name
                         ).to("cuda")
        _ = self.s2s_model.eval()

        print(f"{(time.time()-t0)/60:.2f} minutes to load model weights")


    def embed_passages_for_retrieval(self, passages):
        """ Embed data passages for retrieveal using a QA embedding model. """
        # Batch encode them using the QA tokenizer
        a_toks = self.qa_tokenizer.batch_encode_plus(
            passages,
            max_length=self.encode_length,
            truncation=True,
            padding="max_length")
        a_ids, a_mask = (
            torch.LongTensor(a_toks["input_ids"]).to("cuda"),
            torch.LongTensor(a_toks["attention_mask"]).to("cuda"),
        )
        # Embed the passages using the QA embedder
        with torch.no_grad():
            a_reps = self.qa_embedder.embed_answers(
                a_ids, a_mask).cpu().type(torch.float)
        return a_reps.numpy()


    def create_dense_index(self, batch_size=256):
        """ If a dense index file has not been created, create one. """
        # Create a path name using our parameters
        n_datapoints = self.ds.num_rows
        self.embedded_path = (f"{self.ds_name}_reps_32_{n_datapoints}_"
                              f"{batch_size}_{self.encode_length}.dat")
        if not os.path.isfile(self.embedded_path):
            t0 = time.time()
            # Create a memory map
            fp = np.memmap(self.embedded_path, dtype="float32", mode="w+",
                           shape=(n_datapoints, self.qdim))

            # Embed the passages in batches
            n_batches = int(np.ceil(n_datapoints / batch_size))
            for i in tqdm(range(n_batches)):
                start = i * batch_size
                end = (i + 1)*batch_size
                passages = [p for p in self.ds[start: end]["passage_text"]]
                reps = self.embed_passages_for_retrieval(passages)
                fp[start: end] = reps
            print(f"{(time.time()-t0)/60:.2f} minutes to create dense index")
        else:
            print("Dense index already created")
        print(f"\nDense index file:\n{self.embedded_path}\n")


    def embed_questions_for_retrieval(self, question_lst):
        """ Embed questions for retrieval using a QA embedding model. """
        # Tokenize the questions using the QA tokenizer
        q_toks = self.qa_tokenizer.batch_encode_plus(
            question_lst,
            max_length=self.qdim,
            padding="max_length")
        q_ids, q_mask = (
            torch.LongTensor(q_toks["input_ids"]).to("cuda"),
            torch.LongTensor(q_toks["attention_mask"]).to("cuda"),
        )
        # Embed the questions using the QA embedder
        with torch.no_grad():
            q_reps = self.qa_embedder.embed_questions(
                q_ids, q_mask).cpu().type(torch.float)
        return q_reps.numpy()


    def query_qa_dense_index(self, question, n_results=15, min_length=25):
        """ Query the dense index using the embedding model.  """
        # Embed the questions and create a FAISS index
        q_rep = self.embed_questions_for_retrieval([question])
        faiss_res = faiss.StandardGpuResources()
        passage_reps = np.memmap(self.embedded_path, dtype='float32', mode='r',
                                 shape=(self.ds.num_rows, self.qdim))
        index_flat = faiss.IndexFlatIP(self.qdim)
        gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, index_flat)
        gpu_index.add(passage_reps)
        gpu_index.add(passage_reps)

        # Search the index for answers to the quesiton
        D, I = gpu_index.search(q_rep, 2*n_results)
        res_passages = [self.ds[int(i)] for i in I[0] if i < self.ds.num_rows]

        # Create the results list
        res_list = [dict([(k, p[k]) for k in self.ds.column_names])
                    for p in res_passages]
        res_list = [res for res in res_list if len(res["passage_text"].split())
                    > min_length][:n_results]
        for r, sc in zip(res_list, D[0]):
            r["score"] = float(sc)

        # A supporting document is needed for the sequence to sequence model
        # Basically, it will use these to generate the final answer
        result_texts = [p["passage_text"] for p in res_passages]
        support_doc = "<P> " + " <P> ".join(result_texts)
        return support_doc, res_list


    def ask_a_question(self, question, max_question_len=1024,
                       min_answer_len=64, max_answer_len=512,
                       index_doc_size=15, n_beams=8):
        """
        Ask a question and get the answer from the sequence to sequence model.
        """
        assert isinstance(question, str), "Format your question as a string"
        # Query our dense index
        support_doc, res_list = self.query_qa_dense_index(question,
            n_results=index_doc_size, min_length=min_answer_len)
        question_doc = f"question: {question} context: {support_doc}"

        # Tokenize the question
        q_toks = self.s2s_tokenizer.batch_encode_plus(
            [question_doc],
            max_length=max_question_len,
            truncation=True,
            padding="max_length")
        q_ids, q_mask = (
            torch.LongTensor(q_toks["input_ids"]).to("cuda"),
            torch.LongTensor(q_toks["attention_mask"]).to("cuda"),
        )
        # Indicate that we want an answer
        a_toks = self.s2s_tokenizer.batch_encode_plus(
            ["A"],
            max_length=min(max_question_len, max_answer_len),
            truncation=True,
            padding="max_length")
        a_ids, a_mask = (
            torch.LongTensor(a_toks["input_ids"]).to("cuda"),
            torch.LongTensor(a_toks["attention_mask"]).to("cuda"),
        )

        # generate an answer with beam search
        gen_ids = self.s2s_model.generate(
            input_ids=q_ids,
            attention_mask=q_mask,
            min_length=min_answer_len,
            max_length=max_answer_len,
            num_beams=n_beams,
            do_sample=False,
            early_stopping=True,
            temperature=1.0,
            top_k=None,
            top_p=None,
            eos_token_id=self.s2s_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1, # only want 1 answer to question
            decoder_start_token_id=self.s2s_tokenizer.bos_token_id,
        )
        answer = self.s2s_tokenizer.decode(
            gen_ids[0], skip_special_tokens=True).strip()
        return answer


# Define some small helper functions
def wrap_print(s, n=80):
    """
    Wrap lines to 80 characters when printing since Google Colab
    doesn't wrap print statements
    """
    lines = [""]
    for word in re.split(r"\s+", s):
        if len(lines[-1]) + len(word) <= 79:
            lines[-1] += " " + word
        else:
            lines.append(word)
    for line in lines:
        print(line.strip())


def ask_questions(question_list, model_kwargs):
    """ Ask a list of questions, and have the answers printed out """
    for Q in question_list:
        print(f"Question:\n{Q}")
        A = lfqa.ask_a_question(Q, **model_kwargs)
        print("Answer:")
        wrap_print(A)
        print()
