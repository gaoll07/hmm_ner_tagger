#!/usr/bin/python
# 10708 HW2 Question 3
# define the function that will be used in hw2_main.py
# Author: Lee Gao, lilig@andrew.cmu.edu
import sys
import numpy as np

##### Define functions #####
# define data loading function: loads train/dev/test
def data_loader(name):
    # import training data
    fileName = name
    
    sequence = []
    sequences = []
    #Make the first start symbol
    sequence.append(('<START>','<START>','<START>','<START>'))
    
    for line in open(fileName,'r'):
        line = line.strip()
        if line:
            arr = line.split(' ')
            sequence.append((arr[0],arr[1],arr[2],arr[3]))
        else:
            #Make Stop symbol
            sequence.append(('<STOP>','<STOP>','<STOP>','<STOP>'))
            
            #decode
            #print 'Next sequence is:', sequence
            '''
            if len(sequence)>2:
                annotatedSeq = decode(sequence, tagList, gazeteerDict, weightsDict)
                writeToFile(sequence, annotatedSeq, fa)
            '''
            #raw_input()
            sequences.append(sequence)
            sequence = []
            #Make Start Symbol
            sequence.append(('<START>','<START>','<START>','<START>'))
    
    #Make final stop symbol
    sequence.append(('<STOP>','<STOP>','<STOP>','<STOP>'))
    sequences.append(sequence)
    return sequences

# define trainsition matrix generating function
def transition(training,states):
    state_no = len(states) # number of states
    # generate consecutive tags
    conse_tags = list()
    for sent in training:
        item_no = len(sent)
        for i in xrange(item_no-1):
            item1 = sent[i]
            item2 = sent[i+1]
            conse_tags.append([item1[3],item2[3]])
    
    single_count = np.zeros(state_no)
    double_count = np.zeros((state_no,state_no))
    for tag in conse_tags:
        for i,state in enumerate(states):
            if tag[0] == state:
                single_count[i] += 1
                for j,state2 in enumerate(states):
                    if tag[1] == state2:
                        double_count[i,j] += 1
                        
    single_count_transpose = np.transpose(np.tile(single_count,(state_no,1)))
    A = np.divide(double_count[:9,],single_count_transpose[:9,])
    A = np.concatenate((A,np.array([[0,0,0,0,0,0,0,0,0,1]])),axis=0)
    return A

# define emission matrix generating function
def emission(training,states,words):
    state_no = len(states)
    word_no = len(words)
    
    tag_count = np.zeros(state_no)
    word_count = np.zeros([state_no, word_no])
    
    tag_array = []
    word_array = []
    for sent in training:
        for item in sent:
            tag_array.append(item[0])
            word_array.append(item[3])
            
    for t,tag in enumerate(states):
        tag_count[t] = tag_array.count(tag)
        tag_idx = [i for i,ta in enumerate(tag_array) if ta == tag]
        for w,word in enumerate(words):
            word_temp = [wd for wi,wd in enumerate(word_array) if wi in tag_idx]
            word_count[t,w] = word_temp.count(word)
            
    tag_count_transpose = np.transpose(np.tile(tag_count,(word_no,1)))
    B = np.divide(word_count, tag_count_transpose)
    return B

# define HMM NER tag assigning function that uses Viterbi algorithm 
def hmm_tagger(data_name,A,B,states,words):
    # load data
    data = data_loader(data_name)
    #data = data[:10]
    state_no = len(states)

    # viterbi algorithm
    i_state_all = []
    for sent in data:
        sent_len = len(sent)
        i_state = np.zeros(sent_len)
        i_state[0] = 0
        delta = np.zeros([sent_len,state_no])
        phi = np.zeros([sent_len,state_no])
        delta[0,0] = 1
        t = 1
        for item in sent[1:]:
            try:
                w = words.index(item[0])
                for j in xrange(state_no):
                    temp = np.multiply(delta[t-1],A[:,j])
                    delta[t,j] = np.max(temp) * B[j,w]
                    phi[t,j] = np.argmax(temp)
                t += 1
            except:
                for j in xrange(state_no):
                    temp = np.multiply(delta[t-1],A[:,j])
                    delta[t,j] = np.max(temp)
                    phi[t,j] = np.argmax(temp)
                t += 1
        # termination
        P_star = np.max(delta[sent_len-1,])
        i_state[sent_len-1] = np.argmax(delta[sent_len-1,])
        for t in xrange(sent_len-2,-1,-1):
            i_state[t] = phi[t+1,i_state[t+1]]
        i_state_all.append(i_state)
    
    save_file = data_name + '-1'
    with open(save_file, 'w+') as f:
        for s,sent in enumerate(data):
            for t,item in enumerate(sent):
                if item[0] != '<START>' and item[0] != '<STOP>':
                    f.write(' '.join(item + (states[int(i_state_all[s][t])],))+'\n')
            f.write('\n')
    f.closed

# define shape function
def shape(w):
    s = ''
    wl = list(w)
    for i in wl:
        if i.islower():
            s += 'a'
        elif i.isupper():
            s += 'A'
        elif i.isdigit():
            s += 'd'
        else:
            s += i
    return s

# generating feature weights
def feat_weight(feats, weights):
    if len(feats) == 11:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,tag,gaz,tag_pre = feats
    if len(feats) == 12:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,tag,gaz,tag_pre = feats
    if len(feats) == 13:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,tag,gaz,tag_pre = feats
    if len(feats) == 14:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,pre3,tag,gaz,tag_pre = feats
    if len(feats) == 15:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,pre3,pre4,tag,gaz,tag_pre = feats
    
    # generate features            
    f0 = 'CAPi=' + cap + ':Ti=' + tag
    f1 = 'GAZi=' + gaz + ':Ti=' + tag
    f2 = 'Oi+1=' + word_post.lower() + ':Ti=' + tag
    f3 = 'Oi+1=' + word_post.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f4 = 'Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f5 = 'Oi-1=' + word_pre.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f6 = 'Oi=' + word.lower() + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f7 = 'Oi=' + word.lower() + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f8 = 'Oi=' + word.lower() + ':Pi+1=' + pos_post + ':Ti=' + tag
    f9 = 'Oi=' + word.lower() + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f10 = 'Oi=' + word.lower() + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f11 = 'Oi=' + word.lower() + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f12 = 'Oi=' + word.lower() + ':Ti=' + tag
    f13 = 'Oi=' + word.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f14 = 'Oi=' + word.lower() + ':Wi+1=' + word_post + ':Ti=' + tag
    f15 = 'Oi=' + word.lower() + ':Wi-1=' + word_pre + ':Ti=' + tag
    f16 = 'POSi=' + str(iw) + ':Ti=' + tag
    f17 = 'Pi+1=' + pos_post + ':Ti=' + tag
    f18 = 'Pi+1=' + pos_post + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f19 = 'Pi-1=' + pos_pre + ':Ti=' + tag
    f20 = 'Pi-1=' + pos_pre + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f21 = 'Pi=' + pos + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f22 = 'Pi=' + pos + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f23 = 'Pi=' + pos + ':Pi+1=' + pos_post + ':Ti=' + tag
    f24 = 'Pi=' + pos + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f25 = 'Pi=' + pos + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f26 = 'Pi=' + pos + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f27 = 'Pi=' + pos + ':Ti=' + tag
    f28 = 'Pi=' + pos + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f29 = 'Pi=' + pos + ':Wi+1=' + word_post + ':Ti=' + tag
    f30 = 'Pi=' + pos + ':Wi-1=' + word_pre + ':Ti=' + tag
    f31 = 'Si+1=' + shape(word_post) + ':Ti=' + tag
    f32 = 'Si+1=' + shape(word_post) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f33 = 'Si-1=' + shape(word_pre) + ':Ti=' + tag
    f34 = 'Si-1=' + shape(word_pre) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f35 = 'Si=' + shape(word) + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f36 = 'Si=' + shape(word) + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f37 = 'Si=' + shape(word) + ':Pi+1=' + pos_post + ':Ti=' + tag
    f38 = 'Si=' + shape(word) + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f39 = 'Si=' + shape(word) + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f40 = 'Si=' + shape(word) + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f41 = 'Si=' + shape(word) + ':Ti=' + tag
    f42 = 'Si=' + shape(word) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f43 = 'Si=' + shape(word) + ':Wi+1=' + word_post + ':Ti=' + tag
    f44 = 'Si=' + shape(word) + ':Wi-1=' + word_pre + ':Ti=' + tag
    f45 = 'Ti-1=' + tag_pre + ':Ti=' + tag
    f46 = 'Wi+1=' +  word_post+ ':Ti=' + tag
    f47 = 'Wi+1=' + word_post + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f48 = 'Wi-1=' + word_pre + ':Ti=' + tag
    f49 = 'Wi-1=' + word_pre + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f50 = 'Wi=' + word + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f51 = 'Wi=' + word + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f52 = 'Wi=' + word + ':Pi+1=' + pos_post + ':Ti=' + tag
    f53 = 'Wi=' + word + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f54 = 'Wi=' + word + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f55 = 'Wi=' + word + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f56 = 'Wi=' + word + ':Ti=' + tag
    f57 = 'Wi=' + word + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f58 = 'Wi=' + word + ':Wi+1=' + word_post + ':Ti=' + tag
    f59 = 'Wi=' + word + ':Wi-1=' + word_pre + ':Ti=' + tag
    features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, 
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, 
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, 
                f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, 
                f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, 
                f51, f52, f53, f54, f55, f56, f57, f58, f59]
    if len(feats) >= 12:
        f60 = 'PREi=' + pre1 + ':Ti=' + tag
        features.append(f60)
    if len(feats) >= 13:
        f61 = 'PREi=' + pre2 + ':Ti=' + tag
        features.append(f61)
    if len(feats) >= 14:
        f62 = 'PREi=' + pre3 + ':Ti=' + tag
        features.append(f62)
    if len(feats) >= 15:
        f63 = 'PREi=' + pre4 + ':Ti=' + tag
        features.append(f63)
        
    weight = 0
    for feat in features:
        try:
            weight += weights[feat]
        except:
            pass
    
    return weight

# define linear model NER tagger that uses Viterbi algorithm
def linear_tagger(data_name,tags,weights):
    tag_no = len(tags)
    # load data
    data = data_loader(data_name)
    #data = data[:5]
    # load gazetteer
    gazetteer = []
    with open('/home/lilig/Dropbox/project/hw2_data/gazetteer.txt','r+') as fobj:
        for line in fobj:
            gazetteer.append(line.strip())
    fobj.closed
    
    # viterbi algorithm
    i_state_all = []
    for sent in data:
        sent_len = len(sent)
        i_state = np.zeros(sent_len)
        i_state[0] = 0
        delta = np.zeros([sent_len,tag_no])
        phi = np.zeros([sent_len,tag_no])
        # initialization
        delta[0,0] = 0.9
        delta[0,1] = 0.1

        # recursion
        for iw in xrange(1,len(sent)):
            word = sent[iw][0]
            pos = sent[iw][1]
            if iw > 1: 
                word_pre = sent[iw-1][0]
                pos_pre = sent[iw-1][1]
            else:
                word_pre = '<NULL>'
                pos_pre = '<NULL>'
            if iw < len(sent)-1:
                word_post = sent[iw+1][0]
                pos_post = sent[iw+1][1]
            else:
                word_post = '<NULL>'
                pos_post = '<NULL>'
            cap = str(word[0].isupper())
            feats = [iw+1,word,word_pre,word_post,pos,pos_pre,pos_post,cap]
            if len(word) > 1:
                pre1 = word[:1]
                feats.append(pre1)
            if len(word) > 2:
                pre2 = word[:2]
                feats.append(pre2)
            if len(word) > 3:
                pre3 = word[:3]
                feats.append(pre3)
            if len(word) > 4:
                pre4 = word[:4]
                feats.append(pre4)
            # loop over tags
            weight = np.zeros([tag_no,tag_no])
            for j,tag in enumerate(tags):
                if len(tag) > 2:
                    gaz = str(' '.join([tag[2:],word]) in gazetteer)
                else:
                    gaz = str(' '.join([tag,word]) in gazetteer)
                for jp,tag_pre in enumerate(tags):
                    weight[jp,j] = feat_weight(feats + [tag,gaz,tag_pre],weights)
                temp = np.add(delta[iw-1],weight[:,j])
                delta[iw,j] = np.max(temp)
                phi[iw,j] = np.argmax(temp)

        # termination
        P_star = np.max(delta[sent_len-1,])
        i_state[sent_len-1] = np.argmax(delta[sent_len-1,])
        for t in xrange(sent_len-2,-1,-1):
            i_state[t] = phi[t+1,i_state[t+1]]
        i_state_all.append(i_state)
    
    save_file = data_name + '-2'
    with open(save_file, 'w+') as f:
        for s,sent in enumerate(data):
            for t,item in enumerate(sent):
                if item[0] != '<START>' and item[0] != '<STOP>':
                    f.write(' '.join(item + (tags[int(i_state_all[s][t])],))+'\n')
            f.write('\n')
    f.closed

# define the mapping function phi(x,y) in structural perception
def phi_gen(feats):
    if len(feats) == 11:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,tag,gaz,tag_pre = feats
    if len(feats) == 12:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,tag,gaz,tag_pre = feats
    if len(feats) == 13:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,tag,gaz,tag_pre = feats
    if len(feats) == 14:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,pre3,tag,gaz,tag_pre = feats
    if len(feats) == 15:
        iw,word,word_pre,word_post,pos,pos_pre,pos_post,cap,pre1,pre2,pre3,pre4,tag,gaz,tag_pre = feats
    
    # generate features            
    f0 = 'CAPi=' + cap + ':Ti=' + tag
    f1 = 'GAZi=' + gaz + ':Ti=' + tag
    f2 = 'Oi+1=' + word_post.lower() + ':Ti=' + tag
    f3 = 'Oi+1=' + word_post.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f4 = 'Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f5 = 'Oi-1=' + word_pre.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f6 = 'Oi=' + word.lower() + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f7 = 'Oi=' + word.lower() + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f8 = 'Oi=' + word.lower() + ':Pi+1=' + pos_post + ':Ti=' + tag
    f9 = 'Oi=' + word.lower() + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f10 = 'Oi=' + word.lower() + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f11 = 'Oi=' + word.lower() + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f12 = 'Oi=' + word.lower() + ':Ti=' + tag
    f13 = 'Oi=' + word.lower() + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f14 = 'Oi=' + word.lower() + ':Wi+1=' + word_post + ':Ti=' + tag
    f15 = 'Oi=' + word.lower() + ':Wi-1=' + word_pre + ':Ti=' + tag
    f16 = 'POSi=' + str(iw) + ':Ti=' + tag
    f17 = 'Pi+1=' + pos_post + ':Ti=' + tag
    f18 = 'Pi+1=' + pos_post + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f19 = 'Pi-1=' + pos_pre + ':Ti=' + tag
    f20 = 'Pi-1=' + pos_pre + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f21 = 'Pi=' + pos + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f22 = 'Pi=' + pos + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f23 = 'Pi=' + pos + ':Pi+1=' + pos_post + ':Ti=' + tag
    f24 = 'Pi=' + pos + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f25 = 'Pi=' + pos + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f26 = 'Pi=' + pos + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f27 = 'Pi=' + pos + ':Ti=' + tag
    f28 = 'Pi=' + pos + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f29 = 'Pi=' + pos + ':Wi+1=' + word_post + ':Ti=' + tag
    f30 = 'Pi=' + pos + ':Wi-1=' + word_pre + ':Ti=' + tag
    f31 = 'Si+1=' + shape(word_post) + ':Ti=' + tag
    f32 = 'Si+1=' + shape(word_post) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f33 = 'Si-1=' + shape(word_pre) + ':Ti=' + tag
    f34 = 'Si-1=' + shape(word_pre) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f35 = 'Si=' + shape(word) + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f36 = 'Si=' + shape(word) + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f37 = 'Si=' + shape(word) + ':Pi+1=' + pos_post + ':Ti=' + tag
    f38 = 'Si=' + shape(word) + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f39 = 'Si=' + shape(word) + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f40 = 'Si=' + shape(word) + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f41 = 'Si=' + shape(word) + ':Ti=' + tag
    f42 = 'Si=' + shape(word) + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f43 = 'Si=' + shape(word) + ':Wi+1=' + word_post + ':Ti=' + tag
    f44 = 'Si=' + shape(word) + ':Wi-1=' + word_pre + ':Ti=' + tag
    f45 = 'Ti-1=' + tag_pre + ':Ti=' + tag
    f46 = 'Wi+1=' +  word_post+ ':Ti=' + tag
    f47 = 'Wi+1=' + word_post + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f48 = 'Wi-1=' + word_pre + ':Ti=' + tag
    f49 = 'Wi-1=' + word_pre + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f50 = 'Wi=' + word + ':Oi+1=' + word_post.lower() + ':Ti=' + tag
    f51 = 'Wi=' + word + ':Oi-1=' + word_pre.lower() + ':Ti=' + tag
    f52 = 'Wi=' + word + ':Pi+1=' + pos_post + ':Ti=' + tag
    f53 = 'Wi=' + word + ':Pi-1=' + pos_pre + ':Ti=' + tag
    f54 = 'Wi=' + word + ':Si+1=' + shape(word_post) + ':Ti=' + tag
    f55 = 'Wi=' + word + ':Si-1=' + shape(word_pre) + ':Ti=' + tag
    f56 = 'Wi=' + word + ':Ti=' + tag
    f57 = 'Wi=' + word + ':Ti-1=' + tag_pre + ':Ti=' + tag
    f58 = 'Wi=' + word + ':Wi+1=' + word_post + ':Ti=' + tag
    f59 = 'Wi=' + word + ':Wi-1=' + word_pre + ':Ti=' + tag
    features = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, 
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, 
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, 
                f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, 
                f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, 
                f51, f52, f53, f54, f55, f56, f57, f58, f59]
    if len(feats) >= 12:
        f60 = 'PREi=' + pre1 + ':Ti=' + tag
        features.append(f60)
    if len(feats) >= 13:
        f61 = 'PREi=' + pre2 + ':Ti=' + tag
        features.append(f61)
    if len(feats) >= 14:
        f62 = 'PREi=' + pre3 + ':Ti=' + tag
        features.append(f62)
    if len(feats) >= 15:
        f63 = 'PREi=' + pre4 + ':Ti=' + tag
        features.append(f63)
    phi = np.ones([len(features)])
    return [features,phi]

# define weight training function
import numpy as np
def weight_train(data,tags):
    tag_no = len(tags)
    # load gazetteer
    gazetteer = []
    with open('/home/lilig/Dropbox/project/hw2_data/gazetteer.txt','r+') as fobj:
        for line in fobj:
            gazetteer.append(line.strip())
    fobj.closed
    
    # number of iterations
    k = 100
    
    # update weights
    with open('weight_self_training', 'w+') as fobj:
        for sent in data:
            # recursion
            for iw in xrange(len(sent)):
                tag_true = sent[iw][3]
                word = sent[iw][0]
                pos = sent[iw][1]
                if iw > 0: 
                    word_pre = sent[iw-1][0]
                    pos_pre = sent[iw-1][1]
                    tag_pre = sent[iw-1][3]
                else:
                    word_pre = '<NULL>'
                    pos_pre = '<NULL>'
                    tag_pre = '<NULL>'
                if iw < len(sent)-1:
                    word_post = sent[iw+1][0]
                    pos_post = sent[iw+1][1]
                else:
                    word_post = '<NULL>'
                    pos_post = '<NULL>'
                cap = str(word[0].isupper())
                #print 'word:',word,'pos:',pos
                feats = [iw+1,word,word_pre,word_post,pos,pos_pre,pos_post,cap]
                if len(word) > 4:
                    pre1 = word[:1]
                    feats.append(pre1)
                    pre2 = word[:2]
                    feats.append(pre2)
                    pre3 = word[:3]
                    feats.append(pre3)
                    pre4 = word[:4]
                    feats.append(pre4)
                    weight = np.ones([64])
                elif len(word) == 4:
                    pre1 = word[:1]
                    feats.append(pre1)
                    pre2 = word[:2]
                    feats.append(pre2)
                    pre3 = word[:3]
                    feats.append(pre3)
                    weight = np.ones([63])
                elif len(word) == 3:
                    pre1 = word[:1]
                    feats.append(pre1)
                    pre2 = word[:2]
                    feats.append(pre2)
                    weight = np.ones([62])
                elif len(word) == 4:
                    pre1 = word[:1]
                    feats.append(pre1)
                    weight = np.ones([61])
                else:
                    weight = np.ones([60])
                # initilaize
                for ii in xrange(k):
                    opti = np.zeros([tag_no,1])
                    # loop over tags
                    for j,tag in enumerate(tags):
                        if len(tag) > 2:
                            gaz = str(' '.join([tag[2:],word]) in gazetteer)
                        else:
                            gaz = str(' '.join([tag,word]) in gazetteer)
                        # generate features
                        features,phi = phi_gen(feats + [tag,gaz,tag_pre])
                        if tag != tag_true:
                            phi = np.zeros([len(phi)])
                        #print weight,phi
                        opti[j] = np.dot(weight,phi)
                    tag_pdt = tags[np.argmax(opti)]
                    # update weight
                    c = 0.5
                    features_pdt,phi_pdt = phi_gen(feats + [tag_pdt,gaz,tag_pre])
                    features_true,phi_true = phi_gen(feats + [tag_true,gaz,tag_pre])
                    weight = np.add(weight,np.multiply(c,np.add(-phi_pdt,phi_true)))
    
        for i in xrange(len(features)):
            fobj.write(' '.join(features[i],weight[i])+'\n')
            fobj.write('\n')
    fobj.closed
