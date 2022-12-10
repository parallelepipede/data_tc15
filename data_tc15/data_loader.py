#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paul Zanoncelli
"""

import os
import os.path as osp
import urllib 
import tarfile
from zipfile import ZipFile
from graph_files import load_dataset


import networkx as nx

import random 
from sys import maxsize


class DataLoader():
    
    def __init__(self,name,root = 'data',downloadAll = False,reload = False,mode = 'Networkx', option = None): # option : number, gender, letter
        self.name = name
        self.dir_name = "_".join(name.split("-"))
        self.root = root
        self.option = option 
        self.mode = mode
        if not osp.exists(self.root) :
            os.makedirs(self.root)
        self.url = "https://brunl01.users.greyc.fr/CHEMISTRY/"
        self.urliam = "https://iapr-tc15.greyc.fr/IAM/"
        self.downloadAll = downloadAll
        self.reload = reload
        self.list_database = {
            "Ace" : (self.url,"ACEDataset.tar"),
            "Acyclic" : (self.url,"Acyclic.tar.gz"),
            "Aids" : (self.urliam,"AIDS.zip"),
            "Alkane" : (self.url,"alkane_dataset.tar.gz"),
            "Chiral" : (self.url,"DatasetAcyclicChiral.tar"),
            "Coil_Del" : (self.urliam,"COIL-DEL.zip"),
            "Coil_Rag" : (self.urliam,"COIL-RAG.zip"),
            "Fingerprint" : (self.urliam,"Fingerprint.zip"),
            "Grec" : (self.urliam,"GREC.zip"),
            "Letter" : (self.urliam,"Letter.zip"),
            "Mao" : (self.url,"mao.tgz"),
            "Monoterpenoides" : (self.url,"monoterpenoides.tar.gz"),
            "Mutagenicity" : (self.urliam,"Mutagenicity.zip"),
            "Pah" : (self.url,"PAH.tar.gz"),
            "Protein" : (self.urliam,"Protein.zip"),
            "Ptc" : (self.url,"ptc.tgz"),
            "Steroid" : (self.url,"SteroidDataset.tar"),
            "Vitamin" : (self.url,"DatasetVitamin.tar"),
            "Web" : (self.urliam,"Web.zip")
        }
        
        self.data_to_use_in_datasets = {
            "Acyclic" : ("Acyclic/dataset_bps.ds"),
            "Aids" : ("AIDS_A.txt"),
            "Alkane" : ("Alkane/dataset.ds","Alkane/dataset_boiling_point_names.txt"),
            "Mao" : ("MAO/dataset.ds"),
            "Monoterpenoides" : ("monoterpenoides/dataset_10+.ds"), #('monoterpenoides/dataset.ds'),('monoterpenoides/dataset_9.ds'),('monoterpenoides/trainset_9.ds')

        }
        self.has_train_valid_test = {
            "Coil_Del" : ('COIL-DEL/data/test.cxl','COIL-DEL/data/train.cxl','COIL-DEL/data/valid.cxl'),
            "Coil_Rag" : ('COIL-RAG/data/test.cxl','COIL-RAG/data/train.cxl','COIL-RAG/data/valid.cxl'),
            "Fingerprint" : ('Fingerprint/data/test.cxl','Fingerprint/data/train.cxl','Fingerprint/data/valid.cxl'),
            "Grec" : ('GREC/data/test.cxl','GREC/data/train.cxl','GREC/data/valid.cxl'),
            "Letter" : {'HIGH' : ('Letter/HIGH/test.cxl','Letter/HIGH/train.cxl','Letter/HIGH/validation.cxl'),
                        'MED' : ('Letter/MED/test.cxl','Letter/MED/train.cxl','Letter/MED/validation.cxl'),
                        'LOW' : ('Letter/LOW/test.cxl','Letter/LOW/train.cxl','Letter/LOW/validation.cxl')
                       },
            "Mutagenicity" : ('Mutagenicity/data/test.cxl','Mutagenicity/data/train.cxl','Mutagenicity/data/validation.cxl'),
            "Pah" : ['PAH/testset_0.ds','PAH/trainset_0.ds'],
            "Protein" : ('Protein/data/test.cxl','Protein/data/train.cxl','Protein/data/valid.cxl'),
            "Web" : ('Web/data/test.cxl','Web/data/train.cxl','Web/data/valid.cxl')
        }
    
        if not self.name : 
            raise ValueError("No dataset entered"  )
        if self.name not in self.list_database:
            message = "Invalid Dataset name " + self.name
            message += '\n Available datasets are as follows : \n\n'
            
            message += '\n'.join(database for database in self.list_database)
            raise ValueError(message)
        if self.downloadAll : 
            print('Waiting...')
            for database in self.list_database : 
                self.write_archive_file(database)
            print('Finished')
        else:
            self.write_archive_file(self.name)
        self.max_for_letter = 0
        self.dataset = self.open_files()
        self.info_dataset = {
            'Ace' : "This dataset is not available yet",
            'Acyclic' : "This dataset isn't composed of valid, test, train dataset but one whole dataset \ndataloader = DataLoader('Acyclic,root = ...') \nGs,y,label_names = dataloader.dataset ",
            'Aids' : "This dataset is not available yet",
            'Alkane' : "This dataset isn't composed of valid, test, train dataset but one whole dataset \ndataloader = DataLoader('Alkane',root = ...) \nGs,y,label_names = dataloader.dataset ",
            'Chiral' : "This dataset is not available yet",
            "Coil_Del" : "This dataset has test,train,valid datasets. \ndataloader = DataLoader('Coil-Deg', root = ...). \ntest,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train \nGs_valid,y_valid,label_names_valid = valid",
            "Coil_Rag" : "This dataset has test,train,valid datasets. \ndataloader = DataLoader('Coil-Rag', root = ...). \ntest,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train\n Gs_valid,y_valid,label_names_valid = valid",
            "Fingerprint" : "This dataset has test,train,valid datasets. \ndataloader = DataLoader('Fingerprint', root = ...). \ntest,train,valid = dataloader.dataset. \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train\n Gs_valid,y_valid,label_names_valid = valid",
            "Grec" : "This dataset has test,train,valid datasets. Write dataloader = DataLoader('Grec', root = ...). \ntest,train,valid = dataloader.dataset. \nGs_test,y_test,label_names_test = test\n Gs_train,y_train,label_names_train = train\n Gs_valid,y_valid,label_names_valid = valid",
            "Letter" : "This dataset has test,train,valid datasets. Choose between high,low,med dataset. \ndataloader = DataLoader('Letter', root = ..., option = 'high') \ntest,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train \nGs_valid,y_valid,label_names_valid = valid",
            'Mao' : "This dataset isn't composed of valid, test, train dataset but one whole dataset \ndataloader = DataLoader('Mao',root= ...) \nGs,y,label_names = dataloader.dataset ",
            'Monoterpenoides': "This dataset isn't composed of valid, test, train dataset but one whole dataset\n Write dataloader = DataLoader('Monoterpenoides',root= ...) \nGs,y,label_names = dataloader.dataset ",
            'Mutagenicity' : "This dataset has test,train,valid datasets. \ndataloader = DataLoader('Mutagenicity', root = ...) \ntest,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test\n Gs_train,y_train,label_names_train = train \nGs_valid,y_valid,label_names_valid = valid",
            'Pah' : 'This dataset is composed of test and train datasets. '+ str(self.max_for_letter + 1) + ' datasets are available. \nChoose number between 0 and ' + str(self.max_for_letter) + "\ndataloader = DataLoader('Pah', root = ...,option = 0) \ntest,train = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train\n ",
            "Protein" : "This dataset has test,train,valid dataset. \ndataloader = DataLoader('Protein', root = ...) \n test,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train \nGs_valid,y_valid,label_names_valid = valid",
            "Ptc" : "This dataset has test and train datasets. Select gender between mm, fm, mr, fr. \ndataloader = DataLoader('Ptc',root = ...,option = 'mm') \ntest,train = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train_,y_train,label_names_train = train",
            "Steroid" : "This dataset is not available yet",
            'Vitamin' : "This dataset is not available yet",
            'Web' : "This dataset has test,train,valid datasets. \ndataloader = DataLoader('Web', root = ...) \n test,train,valid = dataloader.dataset \nGs_test,y_test,label_names_test = test \nGs_train,y_train,label_names_train = train \nGs_valid,y_valid,label_names_valid = valid",
        }
         
        if mode == "Pytorch":
            if self.name in self.data_to_use_in_datasets : 
                Gs,y,label_names = self.dataset 
                inputs,adjs,y = self.from_networkx_to_pytorch(Gs,y,label_names) 
                self.pytorch_dataset = inputs,adjs,y
            elif self.name == "Pah":
                self.pytorch_dataset = []
                test,train = self.dataset 
                Gs_test,y_test,label_names_test = test
                Gs = Gs_test
                Gs_train,y_train,label_names_train = train
                self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_test,y_test,label_names_test))
                self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_train,y_train,label_names_train))
            elif self.name in self.has_train_valid_test:
                self.pytorch_dataset = []
                test,train,valid = self.dataset
                Gs_test,y_test,label_names_test = test 
                Gs = Gs_test
                
                Gs_train,y_train,label_names_train = train 
                Gs_valid,y_valid,label_names_valid = valid
                self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_test,y_test,label_names_test))
                self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_train,y_train,label_names_train))
                self.pytorch_dataset.append(self.from_networkx_to_pytorch(Gs_valid,y_valid,label_names_valid))
            
    def download_file(self,url,filename):
        try :
            response = urllib.request.urlopen(url + filename)
        except urllib.error.HTTPError:
            print(filename + " not available or incorrect http link")
            return
        return response
    
    def write_archive_file(self,database):
        path = osp.join(self.root,database)
        url,filename = self.list_database[database]
        filename_dir = osp.join(path,filename)
        if not osp.exists(filename_dir) or self.reload:
            response = self.download_file(url,filename)
            if response is None : 
                return 
            if not osp.exists(path) :
                os.makedirs(path)
            with open(filename_dir,'wb') as outfile : 
                outfile.write(response.read())   
                
    def Dataset(self):
        if self.mode == "Tensorflow":
            return #something
        if self.mode == "Pytorch":
            return self.pytorch_dataset
        return self.dataset
        
    def info(self):
        print(self.info_dataset[self.name])
        
    def iter_load_dataset(self,data):
        results = []
        for datasets in data : 
            results.append(load_dataset(osp.join(self.root,self.name,datasets)))
        return results
    
    def load_dataset(self,list_files):
        if self.name == "Ptc":
            if type(self.option) != str or self.option.upper() not in ['FR','FM','MM','MR']:
                raise ValueError('option for Ptc dataset needs to be one of : \n fr fm mm mr')
            results = []
            results.append(load_dataset(osp.join(self.root,self.name,'PTC/Test',self.option + '.ds')))
            results.append(load_dataset(osp.join(self.root,self.name,'PTC/Train',self.option + '.ds')))
            return results
        if self.name == "Pah":
            maximum_sets = 0
            for file in list_files:
                if file.endswith('ds'):
                    maximum_sets = max(maximum_sets,int(file.split('_')[1].split('.')[0]))
            self.max_for_letter = maximum_sets
            if not type(self.option) == int or self.option > maximum_sets or self.option < 0: 
                raise ValueError('option needs to be an integer between 0 and ' + str(maximum_sets))
            data = self.has_train_valid_test["Pah"]
            data[0] = self.has_train_valid_test["Pah"][0].split('_')[0] + '_' + str(self.option) + '.ds'
            data[1] = self.has_train_valid_test["Pah"][1].split('_')[0] + '_' + str(self.option) + '.ds'
            return self.iter_load_dataset(data)
        if self.name == "Letter":
            if type(self.option) == str and self.option.upper() in self.has_train_valid_test["Letter"]:
                data = self.has_train_valid_test["Letter"][self.option.upper()]
            else:
                message = "The parameter for letter is incorrect choose between : "
                message += "\nhigh  med  low"
                raise ValueError(message)
            return self.iter_load_dataset(data)
        if self.name in self.has_train_valid_test : #common IAM dataset with train, valid and test
            data = self.has_train_valid_test[self.name]
            return self.iter_load_dataset(data)
        if self.name in self.data_to_use_in_datasets:  #common dataset without train,valid and test, only dataset.ds file
            data = self.data_to_use_in_datasets[self.name]
            if len(data) > 1 and data[0] in list_files and data[1] in list_files: #case for Alkane
                return load_dataset(osp.join(self.root,self.name,data[0]),filename_targets = osp.join(self.root,self.name,data[1]))
            if data in list_files:
                return load_dataset(osp.join(self.root,self.name,data))
        raise Exception("This dataset isn't supported yet ")
 
    def open_files(self):
        filename = self.list_database[self.name][1]
        path = osp.join(self.root,self.name)
        filename_archive = osp.join(path,filename)
        
        if filename.endswith('gz'):
            if tarfile.is_tarfile(filename_archive):
                with tarfile.open(filename_archive,"r:gz") as tar:
                    if self.reload: 
                        print(filename + " Downloaded")
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tar, path=path)
                    return self.load_dataset(tar.getnames())
        elif filename.endswith('.tar'):
            if tarfile.is_tarfile(filename_archive):
                with tarfile.open(filename_archive,"r:") as tar:
                    if self.reload : 
                        print(filename + " Downloaded")
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tar, path=path)
                    return self.load_dataset(tar.getnames())
        elif filename.endswith('.zip'):             
            with ZipFile(filename_archive,"r") as zip_ref:
                if self.reload : 
                    print(filename + " Downloaded")
                zip_ref.extractall(path)
                return self.load_dataset(zip_ref.namelist())
        raise Exception("This file type isn't supported yet")

    
    def build_dictionary(self,Gs,label_names):
        labels = set()
        sizes = set()
        for G in Gs : 
            for _,node in G.nodes(data = True): 
                labels.add(node[label_names['node_labels'][0]]) 
            sizes.add(G.order())
        label_dict = {}
        for i,label in enumerate(labels):
            label_dict[label] = [0.]*len(labels)
            label_dict[label][i] = 1.
        return label_dict
    
    def from_networkx_to_pytorch(self,Gs,y,label_names):
        from torch import tensor,Tensor,eye
        from torch.nn.functional import pad
        #exemple for MAO: atom_to_onehot = {'C': [1., 0., 0.], 'N': [0., 1., 0.], 'O': [0., 0., 1.]}
        # code from https://github.com/bgauzere/pygnn/blob/master/utils.py
        
        atom_to_onehot = self.build_dictionary(Gs,label_names)
        
        max_size = 30
        adjs = []
        inputs = []
        for i, G in enumerate(Gs):
            I = eye(G.order(), G.order())
            #A = torch.Tensor(nx.adjacency_matrix(G).todense())
            #A = torch.Tensor(nx.to_numpy_matrix(G))
            if len(label_names['edge_labels']):
                A = tensor(nx.to_scipy_sparse_matrix(G,dtype = int,weight = label_names['edge_labels'][0]).todense(),dtype = int)
            else:
                A = tensor(nx.to_scipy_sparse_matrix(G,dtype = int).todense(),dtype = int)  #what do we use for IAM datasets (they don't have bond_type or event label) ?
            adj = pad(A, pad=(0, max_size-G.order(), 0, max_size-G.order()))  #add I now ? if yes : F.pad(A + I,pad = (...))
            adjs.append(adj)

            f_0 = []
            for _, label in G.nodes(data=True):
                #print("sdfsd ",_,label)
                cur_label = atom_to_onehot[label[label_names['node_labels'][0]]].copy()

                f_0.append(cur_label)
            X = pad(Tensor(f_0), pad=(0, 0, 0, max_size-G.order()))
            inputs.append(X)
        return inputs,adjs,y
            
    def from_pytorch_to_tensorflow(self,batch_size):
        seed = random.randrange(maxsize)
        random.seed(seed)
        tf_inputs = random.sample(self.pytorch_dataset[0],batch_size)
        random.seed(seed)
        tf_y = random.sample(self.pytorch_dataset[2],batch_size)    
    
   
    
dataloader = DataLoader('Mao',root = "database")
dataloader.info()
Gs,y,label_names = dataloader.Dataset()


