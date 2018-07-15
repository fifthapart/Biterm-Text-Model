# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:55:10 2018

@author: Ethan Beaman
"""

from itertools import combinations
from numpy import fromiter, zeros
from numpy.random import choice, randint
from collections import defaultdict, OrderedDict, Counter
from operator import itemgetter

from gensim.utils import simple_preprocess

class BitermModel:
    """
    Biterm model for small documents.
    Parameters are:
        text: a list of lists where each element is the tokens of a document
            gensim.utils.simple_tokenize is a good choice
            Note: best results will come from removing stopwords as well
        ntopics: the number of topics to infer
        alpha: dirichlet prior hyperparameter for topic distribution
        beta: dirichlet prior hyperparameter for word distribution
        niter: number of gibbs sampling steps
    """
    
    def __init__(self, text, ntopics=5, alpha=0.001, beta=0.001, niter=1):
        self.ntopics = ntopics
        self.alpha = alpha
        self.beta = beta
        self.biterms, self.nwords, self.vocab = self._fit_corpus(text)
        self.topics, self.topic_words = self._gibbs_sample(niter)
        self.text = text
        
    def _flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def _normalize(self, v):
        c = v.sum()
        return v / c
        
    def _ngrams(self, sequence, n):
        return zip(*[sequence[i:] for i in range(n)])

    def _skipgrams(self, sequence, n, k):
        grams = []
        for ngram in self._ngrams(sequence + [None]*k, n + k):
            head = ngram[:1]
            tail = ngram[1:]
            for skip_tail in combinations(tail, n - 1):
                if skip_tail[-1] is None:
                    continue
                grams.append(head + skip_tail)
        return grams

    def _fit_corpus(self, text):
        skip2grams = []
        biterms = []
        for doc in text:
            skip2doc = self._skipgrams(doc, 2, 1)
            skip2grams.extend(skip2doc)
            for skip in skip2doc:
                i, j = skip
                if i == j:
                    continue
                b = (i, j) if i < j else (j, i)
                biterms.append(b)
            
        nwords = sum(len(doc) for doc in text)
        vocab = frozenset(self._flatten(text))
        
        return biterms, nwords, vocab

    def _gibbs_sample(self, niter):
        a = self.alpha
        b = self.beta
        K = self.ntopics
        M = self.nwords
        V = self.vocab
        
        def z_posterior(n_z, n_wiz, n_wjz):
            p = (n_z + a)*(n_wiz + b)*(n_wjz + b)/((2*n_z + M*b + 1)*(2*n_z + M*b))
            return p
        
        def theta_z(z):
            n_b = sum(n_z.values())
            return (n_z[z] + a) / (n_b + K*a)
        
        def phi_kw(z, w):
            return (n_wz[w][z] + b) / (2*n_z[z] + M*b)
    
        n_z = defaultdict(int)
        n_wz = {word: defaultdict(int) for word in V}
        current_assignments = []
        for bi in self.biterms:
            wi, wj = bi
            z_init = randint(K)
            
            current_assignments.append((bi, z_init))
            n_z[z_init] += 1
            n_wz[wi][z_init] += 1
            n_wz[wj][z_init] += 1
        
        for _ in range(niter):
            for i, (bi, z) in enumerate(current_assignments):
                wi, wj = bi
    
                n_z[z] -= 1
                n_wz[wi][z] -= 1
                n_wz[wj][z] -= 1
                
                z_prop = fromiter((z_posterior(n_z[z], n_wz[wi][z], 
                                               n_wz[wj][z]) for z in range(K)), 
                                  float, K)
                z_probs = self._normalize(z_prop)
                z_new = choice(K, p=z_probs)
                
                n_z[z_new] += 1
                n_wz[wi][z_new] += 1
                n_wz[wj][z_new] += 1
                
                current_assignments[i] = (bi, z_new)
        
        topic_words = {z: {word: phi_kw(z, word) for word in V} for z in range(K)}
        topic_dist = fromiter((theta_z(z) for z in range(K)), float, K)
        
        return topic_dist, topic_words
        
    def get_topics(self):
        """ 
        the global topic distribution 
        returns a list of K topic probabilities
        """
        return self.topics
        
    def get_topic_words(self, n=5):
        """
        the word distributions per topic
        returns an ordered dict with 
        the top n most probable words of the topic
        """
        sort_words = []
        for z in range(self.ntopics):
            word_prob = sorted(self.topic_words[z].items(), 
                               key=itemgetter(1), reverse=True)
            sort_words.append(word_prob)
            
        topn = OrderedDict((z, dict(t[:n])) for z, t in enumerate(sort_words))
        return topn
    
    def infer_documents(self):
        
        def p_zkbi(z, wi, wj):
            p_z = self.topics[z]
            p_wiz = self.topic_words[z][wi]
            p_wjz = self.topic_words[z][wj]
            return p_z * p_wiz *p_wjz
        
        doc_probs = {}
        for d, doc in enumerate(self.text):
            doc_biterms, _, _ = self._fit_corpus([doc])
            
            bcounts = Counter(doc_biterms)
            nb = fromiter((bcounts[b] for b in doc_biterms), 
                          float, len(doc_biterms))
            p_bd = self._normalize(nb)
            
            p_zb = zeros((self.ntopics, len(doc_biterms)))
            for i, (wi, wj) in enumerate(doc_biterms):
                zb = fromiter((p_zkbi(z, wi, wj) for z in range(self.ntopics)), 
                              float, self.ntopics)
                p_zbi = self._normalize(zb)
                p_zb[:, i] = p_zbi
            
            p_zd = p_zb.dot(p_bd)
            doc_probs[d] = p_zd
            
        return doc_probs
    
if __name__ == '__main__':
    with open('C:/Users/Ethan Beaman/Documents/vet_forum.txt', encoding='utf-8') as f:
        text = f.read().lower().splitlines()
    with open('C:/Users/Ethan Beaman/Documents/stopwords.txt') as s:
        stop = frozenset(s.read().splitlines())
    
    def clean(txt):
        doc = simple_preprocess(txt)
        return [d for d in doc if d not in stop]
    
    POS = {'NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'}
    

    preprocessed = [clean(doc) for doc in text]
    topic = BitermModel(preprocessed, ntopics=10, niter=20)
    
       
    
    
    
    
    
    
    
    
    
