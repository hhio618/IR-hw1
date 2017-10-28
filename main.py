import glob
import re
from collections import Counter, defaultdict
from collections import OrderedDict
from math import log10
import numpy as np

try:
    from lxml import etree

    print("running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree

        print("running with cElementTree on Python 2.5+")

    except ImportError:
        try:
            # Python 2.5
            import xml.etree.ElementTree as etree

            print("running with ElementTree on Python 2.5+")

        except ImportError:
            try:
                # normal cElementTree install
                import cElementTree as etree

                print("running with cElementTree")

            except ImportError:
                try:
                    # normal ElementTree install
                    import elementtree.ElementTree as etree

                    print("running with ElementTree")

                except ImportError:
                    print("Failed to import ElementTree from any known place")
try:
    import lucene

    lucene.initVM()
except ImportError:
    pass

import sys

import timeit


def safe_log(x):
    return np.log10(x.clip(min=0.1))


class TFIDFData(object):
    def __init__(self, row_feature, column_docs, tf_idf, idf):
        self.row_feature, self.column_docs, self.tf_idf, self.idf = row_feature, column_docs, tf_idf, idf


class IndexResult(object):
    def __init__(self, num_doc, min_text_size, min_doc_no, max_text_size, max_doc_no, mean_text_size):
        self.num_doc = num_doc
        self.min_text_size = min_text_size
        self.min_doc_no = min_doc_no
        self.max_text_size = max_text_size
        self.max_doc_no = max_doc_no
        self.mean_text_size = mean_text_size


class HamshahrySearcher(object):
    def __init__(self, xml_files):
        self.tokenizer_regex = "[\n\r\s\.\?!;؛@%-؟:\,\،\+\=\[\]\<\>\(\)»«\{\}]+"
        self.docs_tf_idf = defaultdict()
        self.doc_bow_dict = defaultdict()
        self.xml_files = xml_files
        # To be calculated
        self.tf_idf_data = None

    def index(self):
        num_doc = 0
        text_section_iter = 0
        min_text_size = sys.maxsize
        min_doc_no = 0
        max_text_size = -sys.maxsize
        max_doc_no = 0
        mean_text_size = 0.0

        for xml in xml_files:

            parser = etree.XMLParser(recover=True, strip_cdata=False)
            tree = etree.parse(xml, parser=parser)

            root = tree.getroot()
            docs = root.findall("DOC")
            count = len(docs)
            num_doc += count
            print("Doc processed: ", num_doc)
            for text in root.iter("TEXT"):

                BOW = self.pre_process("".join(text.xpath("text()")))
                doc_no = text.getparent().find("DOCNO").text
                # vocab update here
                self.update_vocab_dict(BOW, doc_no)

                # analytics works
                text_size = len(BOW)
                if text_size < min_text_size:
                    min_doc_no = doc_no
                    min_text_size = text_size
                elif text_size > max_text_size:
                    max_doc_no = doc_no
                    max_text_size = text_size
                # find mean text size
                mean_text_size = (mean_text_size * text_section_iter + text_size) / float(text_section_iter + 1)
                text_section_iter += 1
        print("Finish indexing!")
        print("Calculating tf-idf...")
        self.calculate_corpus_tf_idf(num_doc)
        return IndexResult(num_doc, min_text_size, min_doc_no, max_text_size, max_doc_no, mean_text_size)

    def pre_process(self, content):
        content = content.strip()
        # tanvin
        content = re.sub("[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652]+", "", content)
        content = re.sub("[0-9]+", " ", content)
        BOW = re.compile(self.tokenizer_regex).split(content)
        BOW = [item for item in BOW if len(item) > 1]
        return BOW

    def update_vocab_dict(self, BOW, doc_no):
        self.doc_bow_dict[doc_no] = Counter(BOW)
        for token in set(BOW):
            self.docs_tf_idf[token] = self.docs_tf_idf.get(token, 0) + 1

    def calculate_corpus_tf_idf(self, num_doc):
        row_feature = list(self.docs_tf_idf.keys())
        doc_freq = np.array(list(self.docs_tf_idf.values()))
        idf = np.log(num_doc / doc_freq)
        for DOCID in self.docs_tf_idf:
            for item in self.docs_tf_idf[DOCID]:
                self.docs_tf_idf[DOCID][item] = (1 + log10(self.docs_tf_idf[DOCID][item])) * idf[item]
        self.tf_idf_data = TFIDFData(row_feature, list(self.doc_bow_dict.keys()), self.docs_tf_idf, idf)
        print("TF-IDF calculated for all docs!")

    def calculate_query_vector(self, query):
        row_feature = self.tf_idf_data.row_feature
        tf = [0] * len(row_feature)
        for token in query.split():
            idx = row_feature.index(token)
            tf[idx] += 1

        tf = np.array(tf)
        query_vector = (1 + safe_log(tf)) * self.tf_idf_data.idf
        return query_vector

    def calculate_doc_vector(self, doc):
        row_feature = self.tf_idf_data.row_feature
        tf = [0] * len(row_feature)
        doc_tf_idf = self.tf_idf_data.tf_idf[doc]
        for token in doc_tf_idf:
            idx = row_feature.index(token)
            tf[idx] += doc_tf_idf[token]
        return np.array(tf)

    def search(self, query):
        # calculating query vector
        query_vector = self.calculate_query_vector(query)
        docs = np.array(self.tf_idf_data.column_docs)
        cosine_similarities = []
        for doc in docs:
            doc_vector = self.calculate_doc_vector(doc)
            norm = (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector, axis=1))
            cosine_similarity = doc_vector.dot(query_vector) / norm
            cosine_similarities.append(cosine_similarity)

        cosine_similarities = np.array(cosine_similarities)
        idx = cosine_similarities.argsort()[::-1]
        docs = docs[idx[:20]]
        cosine_similarity = cosine_similarities[idx[:20]]
        for i in range(20):
            print("Query result(%d): %s , Similarity: %g" % (i + 1, docs[i], cosine_similarity[i]))
        return docs, cosine_similarity


if __name__ == '__main__':
    path = "./data/*/*.xml"
    xml_files = glob.glob(path)
    # xml_files = ["HAM2-031201.xml"]
    searcher = HamshahrySearcher(xml_files)
    start_time = timeit.default_timer()
    index_result = searcher.index()
    print("Number of docs:", index_result.num_doc)
    print("min_text_size:", index_result.min_text_size)
    print("min_doc_no", index_result.min_doc_no)
    print("max_text_size", index_result.max_text_size)
    print("max_doc_no", index_result.max_doc_no)
    print("mean_text_size", index_result.mean_text_size)
    print("Vocab size", searcher.docs_tf_idf.__len__())
    print("Query:", 'بازار بزرگ تهران')
    searcher.search('بازار بزرگ تهران')
    print("Overall time elapsed: ", (timeit.default_timer() - start_time))
