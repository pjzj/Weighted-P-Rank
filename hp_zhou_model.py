import os
import pickle
from model.hp_model.hp_preprocess import hp_preprocess
from model.params import params
import numpy as np
import copy
import time
class hp_zhou_model:
    def __init__(self,generate1=False):
        self.hp = hp_preprocess(generate=False)
        self.papers = self.hp.data['paper_information']
        self.authors = self.get_author_list()
        self.paper_s2i_dict = {v:k for k,v in enumerate([paper['id'] for paper in self.papers])}
        self.journal_s2i_dict,self.journal_list = self.get_journal_s2i_dict()
        self.author_s2i_dict = {v:k for k,v in enumerate(self.authors)}
        self.citations = self.hp.data['citations']
        self.paper_num = len(self.paper_s2i_dict)
        self.today = time.time()



        if generate1 == True:
            # self.get_paper_author_graph()
            # self.get_paper_journal_graph()
            self.word_s2i_dict,self.paper_semantics = self.get_word_unit_list()
            self.word_embed = self.load_glove_embedding()
            # self.get_normalized_timestramp()
            self.get_paper_citation_graph()

        else:
            print('reading the related matrix: word_s2i_dict, paper_sematics, word_embed, Gpa_matrix, Gpj_matrix, Gpp_matrix...')
            with open(os.path.join(params.hep_save_path, 'word_s2i_dict.pk'), mode='rb') as f:
                self.word_s2i_dict = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'paper_sematics.pk'), mode='rb') as f:
                self.paper_semantics = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'Gpa_matrix.pk'), mode='rb') as f:
                self.Gpa_matrix = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'Gpj_matrix.pk'), mode='rb') as f:
                self.Gpj_matrix = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'embed_matrix.pk'), mode='rb') as f:
                self.word_embed = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'Gpp_matrix.pk'), mode='rb') as f:
                self.Gpp_matrix = pickle.load(f)
            with open(os.path.join(params.hep_save_path, 'normalized_time.pk'), mode='rb') as f:
                self.normalized_time = pickle.load(f)

            print('read Gpa matrix, shape: {}'.format(self.Gpa_matrix.shape))
            print('read Gpj matrix, shape: {}'.format(self.Gpj_matrix.shape))
            print('read Gpp matrix, shape: {}'.format(self.Gpp_matrix.shape))
            print('read normalized time, shape: {}'.format(self.normalized_time.shape))

        # print()
    def get_author_list(self):
        author_list = []
        for paper in self.papers:
            authors = paper['author']
            for author in authors:
                if author not in author_list:
                    author_list.append(author)
        return author_list
    def get_normalized_timestramp(self):
        paper_timestramp = []
        for paper in self.papers:
            timeArray = time.strptime(str(paper['time']), "%Y-%m-%d %H:%M:%S")
            stramp_time = time.mktime(timeArray)
            paper_timestramp.append(stramp_time)
        erlist_time = min(paper_timestramp)
        longest_time = self.today - erlist_time
        normalized_time = np.array([np.exp(- params.rou * (self.today - publication_time) / longest_time) for publication_time in paper_timestramp])
        with open(os.path.join(params.hep_save_path,'normalized_time.pk'),mode='wb') as f:
            pickle.dump(normalized_time,f)
        print('generating normalized time, shape: {}...'.format(normalized_time.shape))

    def get_word_unit_list(self):
        word_list = []
        paper_sematics = {}
        for paper in self.papers:
            paper_id, title,abstract = paper['id'],paper['title'],paper['abs']
            title.extend(abstract)
            for word in title:
                if word not in word_list:
                    word_list.append(word)
            paper_sematics[paper_id] = title
        word_s2i_dict = {v:k for k,v in enumerate(word_list,1)}

        with open(os.path.join(params.hep_save_path,'word_s2i_dict.pk'),mode='wb') as f:
            pickle.dump(word_s2i_dict,f)
        with open(os.path.join(params.hep_save_path,'paper_sematics.pk'),mode='wb') as f:
            pickle.dump(paper_sematics,f)
        # print('generating word_s2i_dict and paper_sematics...')
        return word_s2i_dict,paper_sematics

    def get_journal_s2i_dict(self):
        journal_list = []
        for paper in self.papers:
            if paper['journal'] not in journal_list:
                journal_list.append(paper['journal'])
        return {v: k for k, v in enumerate(journal_list)},journal_list

    def get_paper_author_graph(self):
        '''
        Link Weighting in Paper-Author Graph (Gpa)
        :return: Gpa
        '''
        paper_num = len(self.papers)
        author_num = len(self.authors)

        Gpa_matrix = np.zeros(shape=(paper_num,author_num),dtype=np.float)

        for paper in self.papers:
            for author in self.authors:
                paper_id,paper_authors = paper['id'],paper['author']
                if author in paper_authors:
                    author_rank, paper_authors_num = paper_authors.index(author) + 1, len(paper_authors)
                    W_rj = (2 * paper_authors_num - author_rank + 2) / (paper_authors_num * (paper_authors_num + 1)) * 2 / 3
                    paper_cord,author_cord = self.paper_s2i_dict[paper_id],self.author_s2i_dict[author]
                    Gpa_matrix[paper_cord,author_cord] = W_rj
        with open(os.path.join(params.hep_save_path,'Gpa_matrix.pk'),mode='wb') as f:
            pickle.dump(Gpa_matrix,f)
        print('generating Gpa matrix, shape: {}...'.format(Gpa_matrix.shape))

    def get_paper_journal_graph(self):
        '''
        Link Weighting in Paper Journal Graph
        :return: Gpj
        '''
        paper_num = len(self.papers)
        journal_num = len(self.journal_s2i_dict)

        Gpj_matrix = np.zeros(shape=(paper_num,journal_num),dtype=np.float)

        for paper in self.papers:
            for journal in self.journal_s2i_dict.keys():
                paper_id, paper_journal = paper['id'], paper['journal']
                if journal == paper_journal:
                    paper_cord,journal_cord = self.paper_s2i_dict[paper_id],self.journal_s2i_dict[paper_journal]
                    Gpj_matrix[paper_cord,journal_cord] = 1.0
        with open(os.path.join(params.hep_save_path,'Gpj_matrix.pk'),mode='wb') as f:
            pickle.dump(Gpj_matrix,f)
        print('generating Gpj matrix, shape: {}...'.format(Gpj_matrix.shape))

    def get_paper_citation_graph(self):
        '''
        Link Weighting in Paper Citation Graph
        :return: Gp
        '''
        paper_num = len(self.papers)
        # sim1_matrix = np.zeros(shape=(paper_num,paper_num),dtype=np.float)
        sim1_matrix = [0.0]
        # sim2_matrix = np.zeros(shape=(paper_num,paper_num),dtype=np.float)
        sim2_matrix = [0.0]
        similarity_matrix = np.zeros(shape=(paper_num,paper_num),dtype=np.float) + 0.1
        for paper_i in self.papers:
            paper_i_id,paper_i_title = paper_i['id'],paper_i['title']
            for paper_j in self.papers:
                paper_j_id,paper_j_title = paper_j['id'],paper_j['title']
                sim1_ij = 0.0
                if paper_i_id != paper_j_id:
                    sents_i,sents_j = [],[]
                    for word_i in self.paper_semantics[paper_i_id]:
                        word_i_id = self.word_s2i_dict[word_i]
                        sents_i.append(self.word_embed[word_i_id])
                    for word_j in self.paper_semantics[paper_j_id]:
                        word_j_id = self.word_s2i_dict[word_j]
                        sents_j.append(self.word_embed[word_j_id])

                    sents_i_embed = list(np.mean(np.array(sents_i),axis=0))
                    sents_j_embed = list(np.mean(np.array(sents_j),axis=0)) # signature j,S --> embed_dim

                    sent_ij_rank = []
                    temp_embed = copy.deepcopy(sents_j_embed)
                    temp_embed.sort(reverse=True)
                    # get the rank r^i_j in signature j
                    for i,num in enumerate(sents_i_embed):
                        for j, value in enumerate(temp_embed):
                            if value < num:
                                rank_pos = j + 1
                                sent_ij_rank.append(rank_pos)
                                break
                    fenzi,fenmu = 0.0,0.0
                    for i,num in enumerate(sent_ij_rank,1):
                        fenzi += np.exp(1 / (num + num**2))
                        fenmu += np.exp(1.0 / (2 * i))
                    sim1_ij = np.tanh(params.alpha * fenzi / (params.beta * fenmu))
                    node_Pi,node_Pj,node_Pij = 0.0,0.0,0.0
                    for key,value in self.citations.items():
                        if paper_i_id == key:
                            node_Pi += len(value) # the number of links to
                        elif paper_i_id in value:
                            node_Pi += 1
                    for key, value in self.citations.items():
                        if paper_j_id == key:
                            node_Pj += len(value)  # the number of links to
                        elif paper_j_id in value:
                            node_Pj += 1
                    for key, value in self.citations.items():
                        if paper_i_id in value and paper_j_id in value:
                            node_Pij += 1
                    set_cititions_i,set_cititions_j = None,None
                    if paper_i_id in self.citations.keys():
                        set_cititions_i = set(self.citations[paper_i_id])
                    if paper_j_id in self.citations.keys():
                        set_cititions_j = set(self.citations[paper_j_id])
                    if set_cititions_i is not None and set_cititions_j is not None:
                        node_Pij += len(set_cititions_i.intersection(set_cititions_j))

                    paper_i_index, paper_j_index = self.paper_s2i_dict[paper_i_id],self.paper_s2i_dict[paper_j_id]

                    # print((paper_i_index, paper_j_index))
                    sim2_ij = node_Pij / np.sqrt(node_Pi * node_Pj + 1)

                    epsilon_1 = np.median(sim1_matrix)

                    # epsilon_1 = 0.2
                    sim1_matrix.append(sim1_ij)

                    lambda_1 = np.exp(params.mu * np.abs(sim1_ij - epsilon_1))

                    epsilon_2 = np.median(sim2_matrix)
                    # epsilon_2 = 0.2

                    sim2_matrix.append(sim2_ij)

                    lambda_2 = np.exp(params.mu * np.abs(sim2_ij - epsilon_2))

                    Wij = lambda_1 * sim1_ij + lambda_2 * sim2_ij

                    if paper_i_id in self.citations.keys() and paper_j_id in self.citations[paper_i_id]:
                        similarity_matrix[paper_i_index, paper_j_index] = Wij
                        # print(Wij)
        similarity_matrix_hang = np.tile(np.expand_dims(np.sum(similarity_matrix,axis=-1),axis=-1),reps=[1,paper_num])

        similarity_matrix = similarity_matrix / similarity_matrix_hang
        # print(similarity_matrix)
        # print(np.isnan(similarity_matrix))
        # similarity_matrix = np.where(np.isnan(similarity_matrix),x=np.zeros(shape=similarity_matrix.shape),y=similarity_matrix)

        Q = (1 - params.d) * np.linalg.pinv((1 - params.d * similarity_matrix))
        # print(Q)
        # print('----')
        Q_hang = np.tile(np.expand_dims(np.sum(abs(Q), axis=-1), axis=-1),reps=[1, paper_num])
        print(Q_hang)
        # Q = Q / Q_hang
        # print('***')
        # print(Q)
        # Q = np.where(np.isnan(Q), x=np.zeros(shape=Q.shape),y=Q)
        with open(os.path.join(params.hep_save_path,'Gpp_matrix.pk'),mode='wb') as f:
            pickle.dump(Q,f)
        print('generating Gpp matrix, shape: {}...'.format(Q.shape))

    def load_glove_embedding(self):
        len_words = len(self.word_s2i_dict) + 1
        embed_matrix = np.zeros(shape=(len_words,params.embed_dim),dtype=np.float)
        with open(params.embed_path,mode='r',encoding='utf-8') as f:
            for line in f.readlines():
                word_np = line.strip().split()
                word,word_embed = word_np[0],np.array(word_np[1:])
                if word in self.word_s2i_dict.keys():
                    embed_matrix[self.word_s2i_dict[word]] = word_embed
        with open(os.path.join(params.hep_save_path,'embed_matrix.pk'),mode='wb') as f:
            pickle.dump(embed_matrix,f)
        print('generating embed matrix, shape: {}...'.format(embed_matrix.shape))
        return embed_matrix

    def get_np_A_and_np_J(self):
        np_A = [1e-7 for _ in self.authors]
        np_J = [1e-7 for _ in self.journal_list]
        for paper in self.papers:
            paper_authors = paper['author']
            paper_journal = paper['journal']
            for paper_author in paper_authors:
                np_A[self.authors.index(paper_author)] += 1
            np_J[self.journal_list.index(paper_journal)] += 1
        return np.expand_dims(np.array(np_A),axis=-1),np.expand_dims(np.array(np_J),axis=-1)

    def execute_algorithm(self):
        initial_xv_paper = np.ones(shape=(self.paper_num,1)) / self.paper_num

        self.Gpj_matrix = self.Gpj_matrix / np.tile(np.expand_dims(np.sum(self.Gpj_matrix,axis=-1),axis=-1),reps=[1,self.Gpj_matrix.shape[1]])

        desired_error = 0.000001

        xv_paper = initial_xv_paper

        np_A,np_J = self.get_np_A_and_np_J()
        print('calculating the shape of np_A: {}'.format(np_A.shape))
        print('calculating the shape of np_J: {}'.format(np_J.shape))
        while True:
            xv_author = np.dot(self.Gpa_matrix.T,xv_paper) # paper_num * 1
            old_xv_paper = xv_paper

            xv_journal = np.dot(self.Gpj_matrix.T,xv_paper) # jornal_num * 1
            part1 = np.dot(self.Gpa_matrix,(xv_author / np_A))
            part2 = np.dot(self.Gpj_matrix,(xv_journal / np_J))
            v_pagerank = params.fai1 * part1 + params.fai2 * part2
            x = np.dot(self.Gpp_matrix,v_pagerank)
            xv_paper = params.gamma * x + params.delta * np.expand_dims(self.normalized_time,axis=-1) + (1 - params.gamma - params.delta) * 1 / self.paper_num
            error = np.linalg.norm(xv_paper - old_xv_paper)
            # print(error)
            if error < desired_error:
                print('xv_paper: {}'.format(xv_paper))
                print('xv_paper: {}'.format(xv_paper))
                print('xv_paper: {}'.format(xv_paper))
                break

if __name__ == '__main__':
    hp = hp_zhou_model(generate1=False)
    hp.execute_algorithm()

