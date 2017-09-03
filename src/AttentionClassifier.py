#!/usr/bin/env python
#coding:utf-8

from chainer import Variable,optimizers,serializers,Chain
import chainer.links as L
import chainer.functions as F
import numpy as np

import codecs
xp = np
from util.NNCommon import *
import util.generators as gens
from util.vocabulary import Vocabulary
import random,os


class LSTM(L.NStepLSTM):
    def __init__(self, n_layer, in_size, out_size, dropout=0.5):
        n_layers = 1
        super(LSTM, self).__init__(n_layers, in_size, out_size, dropout)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == np:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype))

        hy, cy, ys = super(LSTM, self).__call__(self.hx, self.cx, xs)
        self.hx, self.cx = hy, cy
        return ys

class BiEncDecLSTM(Chain):

    def __init__(self,vocab_size,layer_size,in_size,out_size,class_size,drop_ratio=0.5,cudnn_flag=False):
        super(BiEncDecLSTM, self).__init__(
                embed = L.EmbedID(vocab_size,in_size),
                enc_f = LSTM(layer_size,in_size, out_size, dropout=drop_ratio),#, use_cudnn=cudnn_flag),
                enc_b = LSTM(layer_size,in_size, out_size, dropout=drop_ratio),#, use_cudnn=cudnn_flag),
                att_w1 = L.Linear(2*out_size,2*out_size),
                att_w2 = L.Linear(1,2*out_size),
                clssi = L.Linear(2*out_size,class_size)
        )
        self.in_size = in_size
        self.out_size= out_size

    def makeEmbedBatch(self,xs,reverse=False):
        if reverse:
            xs = [xp.asarray(x[::-1],dtype=xp.int32) for x in xs]
        elif not reverse:
            xs = [xp.asarray(x,dtype=xp.int32) for x in xs]
        section_pre = np.array([len(x) for x in xs[:-1]], dtype=np.int32)
        sections = np.cumsum(section_pre) # CuPy does not have cumsum()
        xs = F.split_axis(self.embed(F.concat(xs, axis=0)), sections, axis=0)
        return xs

    def callAndAtt(self,xs):#xsはword_idのlistのlist

        xs_f = self.makeEmbedBatch(xs)
        xs_b = self.makeEmbedBatch(xs,True)

        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f)
        ys_b = self.enc_b(xs_b)
        ys_bi = [F.concat((y_f,y_b[::-1]),axis=1) for y_f,y_b in zip(ys_f,ys_b)]
        y_att = [self.att_w2(np.ones((y_bi.data.shape[0],1),dtype=xp.float32))*F.tanh(self.att_w1(y_bi)) for y_bi in ys_bi]
        y_att = [F.softmax(F.reshape(F.sum(y_ce,axis=1),(1,y_ce.data.shape[0]))) for y_ce in y_att]

        y_conc = [F.transpose(F.concat([y_ce for ri in range(2*self.out_size)],axis=0)) for y_ce in y_att]
        h = F.concat([F.reshape(F.sum(y_ce*y_bi,axis=0),(1,2*self.out_size)) for y_ce,y_bi in zip(y_conc,ys_bi)],axis=0)
        y = self.clssi(h)

        return y,y_att

    def __call__(self,xs):
        y,att_w = self.callAndAtt(xs)
        return y


    def predict(self,xs,vocab):
        t = [1]*len(xs)#1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        xs = [x for x in xs]#1は<s>を指す。decには<s>から入れる。

        t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
        xs_f = self.makeEmbedBatch(xs)
        xs_b = self.makeEmbedBatch(xs,True)

        self.enc_f.reset_state()
        self.enc_b.reset_state()
        ys_f = self.enc_f(xs_f,train=False)
        ys_b = self.enc_b(xs_b,train=False)

        self.dec.hx = self.enc_f.hx
        self.dec.cx = self.enc_f.cx
        ys_d = self.dec(t,train=False)
        ys_w = [self.h2w(y) for y in ys_d]

        t = [(y_each.data[-1].argmax(0)) for y_each in ys_w]
        t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
        count_len=0
        while count_len<10:
            ys_d = self.dec(t,train=False)
            ys_w = [self.h2w(y) for y in ys_d]
            t = [(y_each.data[-1].argmax(0)) for y_each in ys_w]
            t = [self.embed(xp.array([t_each],dtype=xp.int32)) for t_each in t]
            count_len+=1

class Args():
    def __init__(self):
        self.source ="../data/serif8000_fixed.txt"
        self.target ="../data/tag.txt"
        self.source_tr ="../data/serif_tr8000_fixed.txt"
        self.target_tr ="../data/tag_tr.txt"
        # self.source_te ="../data/tweet.txt"
        self.source_te ="../data/serif_te8000_fixed.txt"
        self.target_te ="../data/tag_te.txt"
        self.epoch = 5
        word_set = set();tag_set = set()
        [[word_set.add(word) for word in word_arr] for word_arr in gens.word_list(self.source)]
        [tag_set.add(tag) for tag in codecs.open(self.target,"r",encoding="utf-8")]
        self.n_vocab = len(word_set)
        self.n_tag = len(tag_set)
        self.embed = 50
        self.hidden= 100
        self.layer = 1
        self.batchsize=30
        self.dropout = 0.5
        self.gpu = -1
        if self.gpu>=0:
            import cupy as xp
        self.gradclip = 5

def train(args):
    if os.path.exists("./model/vocab.bin"):
        src_vocab = Vocabulary.load("./model/vocab.bin")
    else:
        src_vocab = Vocabulary.new(gens.word_list(args.source), args.n_vocab)
        src_vocab.save('./model/vocab.bin')
    if os.path.exists("./model/tag.bin"):
        trg_tag = Vocabulary.load("./model/tag.bin")
    else:
        trg_tag = Vocabulary.new(gens.word_list(args.target), args.n_tag)
        trg_tag.save('./model/tag.bin')
    print("vocab_len:{}".format(src_vocab.__len__))
    print("tag_len:{}".format(trg_tag.__len__))
    encdec = BiEncDecLSTM(args.n_vocab,args.layer,args.embed,args.hidden,args.n_tag)
    optimizer = optimizers.Adam()
    optimizer.setup(encdec)

    for e_i in range(args.epoch):
        tt_list = [[src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source_tr)]
        tag_list = [trg_tag.stoi(tag[0]) for tag in gens.word_list(args.target_tr)]
        print("{}:{}".format(len(tt_list),len(tag_list)))
        assert len(tt_list)==len(tag_list)
        ind_arr = [ri for ri in range(len(tt_list))]
        random.shuffle(ind_arr)
        tt_now = (tt_list[ri] for ri in ind_arr)
        tag_now = (tag_list[ri] for ri in ind_arr)
        tt_gen = gens.batch(tt_now,args.batchsize)
        tag_gen = gens.batch(tag_now,args.batchsize)

        for tt,tag in zip(tt_gen,tag_gen):
            y_ws = encdec(tt)

            teac_arr = [src_vocab.itos(t) for t in tt[0]]
            pred_arr = [trg_tag.itos(y_each.data.argmax(0)) for y_each in y_ws]
            print("teach:{}:{}:{}".format(teac_arr,trg_tag.itos(tag[0]),pred_arr[0]))
            tag = xp.array(tag,dtype=xp.int32)
            loss = F.softmax_cross_entropy(y_ws,tag)

            encdec.cleargrads()
            loss.backward()
            optimizer.update()

            # loss.backward()
            # optimizer.target.cleargrads()
            # loss.backward()
            # loss.unchain_backward()
            # optimizer.update()

        serializers.save_npz('./model/attn_tag_model_{}.npz'.format(e_i), encdec)

def test(args,epoch):
    model_name = "./model/attn_tag_model_{}.npz".format(epoch)
    encdec = BiEncDecLSTM(args.n_vocab,args.layer,args.embed,args.hidden,args.n_tag)
    serializers.load_npz(model_name,encdec)
    src_vocab = Vocabulary.load("./model/vocab.bin")
    trg_tag= Vocabulary.load("./model/tag.bin")
    tt_now = ([src_vocab.stoi(char) for char in char_arr] for char_arr in gens.word_list(args.source_te))
    tag_now = (trg_tag.stoi(tag[0]) for tag in gens.word_list(args.target_te))
    tt_gen = gens.batch(tt_now,args.batchsize)
    tag_gen = gens.batch(tag_now,args.batchsize)
    correct_num = 0
    wrong_num = 0
    fw = codecs.open("./output/result_attn_tw{}.csv".format(epoch),"w",encoding="utf-8")
    fw.write("台詞,教師キャラ,予測キャラ,予測値,›単語\n")
    for tt,tag in zip(tt_gen,tag_gen):
        y,att_w = encdec.callAndAtt(tt)
        max_y = [max(F.softmax(F.reshape(y_each.data,(1,len(y_each.data)))).data[0]) for y_each in y]
        y = [y_each.data.argmax(0) for y_each in y]
        for tt_e,y_e,tag_e,max_y_e,att_w_e in zip(tt,y,tag,max_y,att_w):
            txt = ",".join([src_vocab.itos(id) for id in tt_e])
            tag_e = trg_tag.itos(tag_e);y_e = trg_tag.itos(y_e)
            att_ind = att_w_e.data.argmax()
            most_word = src_vocab.itos(tt_e[att_ind])
            fw.write("{}:{}:{}:{}:{}\n".format(txt,tag_e,y_e,max_y_e,most_word))
        correct_num+=len([1 for y_e,tag_e in zip(y,tag) if y_e==tag_e])
        wrong_num+=len([1 for y_e,tag_e in zip(y,tag) if y_e!=tag_e])
    print("epoch:{}".format(epoch))
    print(" correct:{}".format(correct_num))
    print(" wrong:{}".format(wrong_num))
    fw.write("correct{}\n".format(correct_num))
    fw.write("wrong:{}\n".format(wrong_num))
    fw.close()


if __name__=="__main__":
    args = Args()
    train(args)
    for ri in range(1,5):
        test(args,ri)



