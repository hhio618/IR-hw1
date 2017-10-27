#!/usr/bin/env python
import glob

unicode = str
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

import sys, os, lucene, threading, time
from datetime import datetime

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, StringField, TextField, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import RAMDirectory
from java.io import *
from java.util import *
from org.apache.lucene.util import BytesRef, BytesRefIterator
from org.apache.lucene.index import PostingsEnum

from org.apache.lucene.index import *
from org.apache.lucene.search import *
from org.apache.lucene.queryparser.classic import QueryParser


class IndexFiles(object):
    def __init__(self, root, analyzer):
        self.store = RAMDirectory()
        self.analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(self.store, config)
        self.numDocs = self.indexDocs(root, self.writer)
        self.writer.commit()
        self.writer.close()

    def indexDocs(self, root, writer):
        path = root + "/data/*/*.xml"
        # print(path)
        xml_files = glob.glob(path)
        # xml_files = ["HAM2-031201.xml"]
        numDocs = 0
        for xml in xml_files:
            try:
                parser = etree.XMLParser(recover=False, strip_cdata=False)
                tree = etree.parse(xml, parser=parser)

            except etree.XMLSyntaxError as e:
                parser = etree.XMLParser(recover=True, strip_cdata=False)
                tree = etree.parse(xml, parser=parser)

            root = tree.getroot()
            for text in root.iter("TEXT"):
                contents = "".join(text.xpath("text()")).strip()
                doc_no = text.getparent().find("DOCNO").text
                # print("adding", doc_no)
                try:
                    doc = Document()
                    doc.add(StringField("id", doc_no, Field.Store.YES))
                    if len(contents) > 0:
                        doc.add(TextField("contents", contents, Field.Store.YES))
                    else:
                        pass
                        # print("warning: no content in %s" % doc_no)
                    writer.addDocument(doc)
                    numDocs += 1
                except Exception as e:
                    print("Failed in indexDocs:", e)
        return numDocs


from org.apache.lucene.index import MultiFields
from org.apache.lucene.util import BytesRef, BytesRefIterator
import timeit

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    start_time = timeit.default_timer()
    try:
        index = IndexFiles(os.path.dirname(sys.argv[0]), StandardAnalyzer())

        index_reader = DirectoryReader.open(index.store)
        # get vocab size

        terms = MultiFields.getTerms(index_reader, 'contents')
        termEnum = terms.iterator()
        vocabCounter = 0
        for term in BytesRefIterator.cast_(termEnum):
            vocabCounter += 1
        print("Number of docs:", index_reader.numDocs())
        print("Vocab size:", vocabCounter)

        # print min, max, mean
        querystr = 'بازار بزرگ تهران'
        print("Query: ", querystr)
        q = QueryParser("contents", index.analyzer).parse(querystr)
        hitsPerPage = 20
        searcher = IndexSearcher(index_reader)
        docs = searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs
        for i, hit in enumerate(hits):
            docId = hit.doc
            score = hit.score
            d = searcher.doc(docId)
            print("Query result(%d): %s , Similarity: %g" % ((i + 1), d.get("id"), score))
        print("Overall time elapsed: ", (timeit.default_timer() - start_time))
    except Exception as e:
        print(e)
        raise e
