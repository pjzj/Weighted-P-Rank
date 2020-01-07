import os
import re
import pickle
import json
from datetime import datetime
from model.params import params
class hp_preprocess:
    def __init__(self,generate=False):
        self.abs_path = self.load_base_path()
        if generate:
            self.get_paper_info()
        else:
            self.data = self.read_paper_info()
    def load_base_path(self):
        print('Load base path...')
        year_abs_path = {1992 + i: [] for i in range(12)}
        for year in params.hep_years:
            hep_year_path = os.path.join(params.hep_abstract_path, str(year))
            abstract_pathes = os.listdir(hep_year_path)
            for abs_path in abstract_pathes:
                abs_path = os.path.join(hep_year_path,abs_path)
                year_abs_path[year].append(abs_path)
        return year_abs_path

    def get_paper_info(self):
        print('get the information of the paper...')
        authors_list = []
        journal_list = []
        paper_id_list = []
        word_bag = []
        paper_list = []
        citations = {}
        with open(params.hep_citations_path,encoding='utf-8',mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) != 0:
                    new_cite,old_cite = line.split()
                    if new_cite not in citations.keys():
                        citations[new_cite] = [old_cite]
                    else:
                        citations[new_cite].append(old_cite)

        for year in params.hep_years:
            each_year_pathes = self.abs_path[year]
            for each_year_path in each_year_pathes:
                one_paper = {'id':0,'title':None,'author':[],'abs':None,'citations':[],'time':None,'journal':None}
                with open(each_year_path,encoding='utf-8',mode='r') as f:
                    author_flag = True
                    abs_index = 0
                    abs_words = ''
                    for line in f.readlines():
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        if line.startswith('Paper:'):
                            try:
                                one_paper['id'] = line.replace('Paper: ','').split('/')[-1]
                                if one_paper['id'] not in paper_id_list:
                                    paper_id_list.append(one_paper['id'])
                            except:
                                print(each_year_path)
                            if one_paper['id'] in params.invalide_paper:
                                break
                        elif line.startswith('Title:'):
                            one_paper['title'] = line.replace('Title: ','').split()
                        elif line.startswith('Authors:') and author_flag:
                            author_flag = False
                            authors = line.replace('Authors: ','')
                            # >= 2 authors, A, B and C || A, B, C
                            if ' and ' in authors or ', and ' in authors:
                                if ' and ' in authors:
                                    authors_comma = authors.split(' and ')
                                elif ', and ' in author:
                                    authors_comma = authors.split('and')
                                for author in authors_comma:
                                    author = author.strip().strip(',')
                                    if ',' in author:
                                        author_and = author.split(',')
                                        one_paper['author'].extend(author_and)
                                    else:
                                        one_paper['author'].append(author)
                            elif ',' in authors and ', and ' not in authors and ' and ':
                                one_paper['author'].extend(authors.split(','))
                            else:
                                one_paper['author'].append(authors.strip())
                            temp = one_paper['author']
                            one_paper['author'] = []
                            for author in temp:
                                if '(' in author:
                                    author = author[:author.index('(')]
                                if author.strip() == '':
                                    input('the excepted file id: {}'.format(one_paper['id']))
                                # one_paper['author'].append(author.strip() + '****' + str(one_paper['id']))
                                one_paper['author'].append(author.strip())
                            assert len(one_paper['author']) != 0

                            for author in one_paper['author']:
                                if author not in authors_list:
                                    authors_list.append(author)
                        elif line.startswith('Date:'):
                            time_list = line.replace('Date:','').split(',')[-1].strip().split()[:-2]
                            time_change = []
                            for time_point in time_list:
                                year_remove = {92:1992,93:1993,94:1994,95:1995,96:1996,97:1997,98:1998,99:1999}
                                year_list = list(year_remove.keys())
                                month_remove = {'jan':'01','feb':'02','mar':'03','apr':'04','may':'05','jun':'06','jul':'07','aug':'08','sep':'09',
                                                 'oct':'10','nov':'11','dec':'12'}
                                month_list = list(month_remove.keys())
                                week_remove = {'Sun':'Sunday','Mon':'Monday','Tue':'Tuesday','Wed':'Wednesday','Thu':'Thursday','Fri':'Friday','Sat':'Saturday'}
                                if '-' in time_point or '+' in time_point or '(' in time_point or 'MET' in time_point \
                                        or 'BSC' in time_point or time_point in week_remove.keys() or 'MEZ' in time_point or 'WET' in time_point or 'EST' \
                                        in time_point or 'am' in time_point and 'bm' in time_point or 'January' in time_point:
                                    continue
                                elif time_point.isdigit() and int(time_point) in year_list:
                                    time_point = year_remove[int(time_point)]
                                elif time_point.lower() in month_list:
                                    time_point = month_remove[time_point.lower()]
                                time_change.append(time_point)
                            if len(time_list) != 4 or time_list[0].isdigit() == False:
                                continue
                            else:
                                day = time_change[0]
                                if len(day) != 2:
                                    time_change[0] = '0' + day
                                if len(time_change[-1].split(':')) == 2:
                                    time_change[-1] += ':00'
                                try:
                                    str_time = str(time_change[2])+'-'+str(time_change[1])+'-'+str(time_change[0])+' '+str(time_change[-1])
                                    time_change = datetime.strptime(str_time,"%Y-%m-%d %H:%M:%S")
                                    one_paper['time'] = time_change
                                    if time_change is None:
                                        input('>>>')
                                except:
                                    continue
                                # print(one_paper['time'])
                        elif line.startswith('Journal-ref: '):
                            Journal_name = line.replace('Journal-ref: ','').split('. ')[0]
                            sigle_journals = ['Physica','Fizika','Annals of Physics','Chaos Solitons Fractals','Soryushiron Kenkyu',
                                              'Modern Physics Letters','Pramana','Semigroup Forum','Physics Letters','IJGT','Notices of the American Mathematical Society',
                                              'Gauge Theories, Applied Supersymmetry and Quantum Gravity II','JHEP','Nature','Compositio Mathematica','Published','Ciencia',
                                              'PRHEP','Entropy','World Scientific','25th Coral Gables Conference on High Energy Physics and Cosmology',
                                              'NATO ASI Series (Plenum)','Soryushiron kenkyu','Central European Journal of Physics','CCAST-WL workshop series',
                                              'Journal of Shaanxi Normal University (Natural Science Edition)','Brazilian Journal Physics']
                            single_journal_flag = True
                            for single_journal in sigle_journals:
                                if single_journal in Journal_name:
                                    if single_journal not in journal_list:
                                        journal_list.append(single_journal)
                                    one_paper['journal'] = single_journal
                                    single_journal_flag = False
                                    break
                            if Journal_name is not None and '.' not in Journal_name and \
                                    ':' not in Journal_name and \
                                    'Springer' not in Journal_name and \
                                    'in' not in Journal_name.lower() and \
                                    single_journal_flag == True:
                                jor = re.match('.+\.',line.replace('Journal-ref: ',''))

                                if jor is not None:
                                    single_journal = jor.group().replace(' ','').strip('.')
                                    if single_journal not in journal_list:
                                        journal_list.append(single_journal)
                                    one_paper['journal'] = single_journal
                            else:
                                if Journal_name not in journal_list:
                                    journal_list.append(Journal_name)
                                one_paper['journal'] = Journal_name
                        elif line.startswith('\\\\'):
                            abs_index += 1
                        if abs_index == 2:
                            abs_words += ' ' + line.strip()
                        elif abs_index == 3:
                            abs_words = re.sub(',|\.|"|!|\?|\d|\(|\)|\\\\|\$|-|\^|\{|\}|=|\/', '',abs_words)
                            words_list = abs_words.strip().split()
                            words = [word.strip().lower() for word in words_list if len(word) != 0]
                            one_paper['abs'] = words
                            for word in words:
                                if word not in word_bag:
                                    word_bag.append(word)
                            break
                if one_paper['journal'] == None or len(one_paper['abs']) == 0 or one_paper['time'] == None:
                    continue
                if one_paper['id'] in citations.keys():
                    reference = citations[one_paper['id']]
                    if len(reference) != 0:
                        one_paper['citations'].extend(reference)
                paper_list.append(one_paper)

                if len(paper_list) > 100:
                    break

        print('the number of the effective paper: ',len(paper_list))
        with open(os.path.join(params.hep_save_path,'paper_information.pk'),mode='wb') as f:
            pickle.dump(paper_list,f)
        print('save the related dictionaries of these paper...')

        author_s2i_dict = {v:k for k,v in enumerate(authors_list)}
        with open(os.path.join(params.hep_save_path,'author_s2i_dict.pk'),mode='wb') as f:
            pickle.dump(author_s2i_dict,f)
        with open(os.path.join(params.hep_save_path,'authors_list.json'),encoding='utf-8',mode='w') as f:
            json.dump(authors_list,f)

        word_s2i_dict = {v:k for k,v in enumerate(word_bag)}
        with open(os.path.join(params.hep_save_path,'word_s2i_dict.pk'),mode='wb') as f:
            pickle.dump(word_s2i_dict,f)
        with open(os.path.join(params.hep_save_path,'word_bag.json'),encoding='utf-8',mode='w') as f:
            json.dump(word_bag,f)

        word_id_s2i_dict = {v:k for k,v in enumerate(paper_id_list)}
        with open(os.path.join(params.hep_save_path,'word_id_s2i_dict.pk'),mode='wb') as f:
            pickle.dump(word_id_s2i_dict,f)
        with open(os.path.join(params.hep_save_path,'paper_id_list.json'),encoding='utf-8',mode='w') as f:
            json.dump(paper_id_list,f)

        journal_s2i_dict = {v:k for k,v in enumerate(journal_list)}
        with open(os.path.join(params.hep_save_path,'journal_s2i_dict.pk'),mode='wb') as f:
            pickle.dump(journal_s2i_dict,f)
        with open(os.path.join(params.hep_save_path,'journal_list.json'),encoding='utf-8',mode='w') as f:
            json.dump(journal_list,f)

        with open(os.path.join(params.hep_save_path,'citations.json'),encoding='utf-8',mode='w') as f:
            json.dump(citations,f)

    def read_paper_info(self):
        data = {}
        read_list = ['author_s2i_dict.pk','authors_list.json','journal_list.json','journal_s2i_dict.pk','paper_id_list.json',
                     'paper_information.pk','word_bag.json','word_id_s2i_dict.pk','word_s2i_dict.pk','citations.json']
        for file_name in read_list:
            file_path = os.path.join(params.hep_save_path,file_name)
            if 'pk' in file_name:
                data_value = self.load_pickle(file_path)
                data_key = file_name.replace('.pk','')
            else:
                data_value = self.load_json(file_path)
                data_key = file_name.replace('.json', '')
            data[data_key] = data_value
        return data

    def load_pickle(self,file_path):
        with open(file_path,mode='rb') as f:
            return pickle.load(f)
    def load_json(self,file_path):
        with open(file_path,encoding='utf-8',mode='r') as f:
            return json.load(f)

if __name__ == '__main__':
    hp_preprocess = hp_preprocess(generate=True)