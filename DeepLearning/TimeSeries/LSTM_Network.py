class LSTM(object):

    def __init__(self, nin, n_hidden, nout):
        rng = np.random.RandomState(1234)
        # cell input
        W_ug = np.asarray(rng.normal(size=(nin, n_hidden), scale= .01, loc = 0.0), dtype = theano.config.floatX)
        W_hg = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = 0.0), dtype = theano.config.floatX)
        b_g = np.zeros((n_hidden,), dtype=theano.config.floatX)
        # input gate equation
        W_ui = np.asarray(rng.normal(size=(nin, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        W_hi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        b_i = np.zeros((n_hidden,), dtype=theano.config.floatX)
        # forget gate equations
        W_uf = np.asarray(rng.normal(size=(nin, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        W_hf = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        b_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        # cell output gate equations
        W_uo = np.asarray(rng.normal(size=(nin, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        W_ho = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale =.01, loc=0.0), dtype = theano.config.floatX)
        b_o = np.zeros((n_hidden,), dtype=theano.config.floatX)
        # output layer
        W_hy = np.asarray(rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = theano.config.floatX)
        b_hy = np.zeros((nout,), dtype=theano.config.floatX)

        # cell input
        W_ug = theano.shared(W_ug, 'W_ug')
        W_hg = theano.shared(W_hg, 'W_hg')
        b_g = theano.shared(b_g, 'b_g')
        # input gate equation
        W_ui = theano.shared(W_ui, 'W_ui')
        W_hi = theano.shared(W_hi, 'W_hi')
        b_i = theano.shared(b_i, 'b_i')
        # forget gate equations
        W_uf = theano.shared(W_uf, 'W_uf')
        W_hf = theano.shared(W_hf, 'W_hf')
        b_f = theano.shared(b_f, 'b_f')
        # cell output gate equations
        W_uo = theano.shared(W_uo, 'W_uo')
        W_ho = theano.shared(W_ho, 'W_ho')
        b_o = theano.shared(b_o, 'b_o')
        # output layer
        W_hy = theano.shared(W_hy, 'W_hy')
        b_hy = theano.shared(b_hy, 'b_hy')

        self.activ1 = T.nnet.sigmoid
        self.activ2 = T.tanh
        
        lr = T.scalar()
        u = T.matrix()
        t = T.scalar()


        h0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        s0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))

        
        
        
        

        #theano.printing.debugprint([h0_tm1, u, W_hh, W_uh, W_hy, b_hh, b_hy], print_type=True)
        [h, s], _ = theano.scan(self.recurrent_fn, sequences = u,
                           outputs_info = [h0_tm1, s0_tm1],
                           non_sequences = [W_ug, W_hg, b_g, W_ui, W_hi,
                                            b_i, W_uf, W_hf, b_f, W_uo, W_ho, b_o, W_hy, b_hy])

        y = T.dot(h[-1], W_hy) + b_hy
        cost = ((t - y)**2).mean(axis=0).sum()
        update_params = [W_ug, W_hg, b_g, W_ui, W_hi, b_i,W_uf, W_hf, b_f, W_uo, W_ho, b_o, W_hy, b_hy]

        gW_ug, gW_hg, gb_g, gW_ui, gW_hi, gb_i,gW_uf, gW_hf, gb_f, gW_uo, gW_ho, gb_o, gW_hy, gb_hy\
        = T.grad(cost,update_params)
            

            
                 
            
        update = [(W_ug, W_ug - lr*gW_ug), 
                  (W_hg, W_hg - lr*gW_hg ), 
                  (b_g, b_g - lr*gb_g), 
                  (W_ui, W_ui - lr*gW_ui),
                  (W_hi, W_hi - lr*gW_hi), 
                  (b_i, b_i - lr*gb_i), 
                  (W_uf, W_uf - lr*gW_uf), 
                  (W_hf, W_hf - lr*gW_hf),
                  (b_f, b_f - lr*gb_f),
                  (W_uo, W_uo - lr*gW_uo), 
                  (W_ho, W_ho - lr*gW_ho), 
                  (b_o, b_o - lr*gb_o),
                  (W_hy, W_hy - lr*gW_hy), 
                  (b_hy, b_hy - lr*gb_hy)]
        
        #theano.printing.debugprint([h0_tm1], print_type=True)
        self.train_step = theano.function([u, t, lr], cost,
            on_unused_input='warn',
            updates=update,
            allow_input_downcast=True)
        
        
                
        self.predict_step = theano.function([u, t], outputs=[cost, y],
           on_unused_input='warn',
           allow_input_downcast=True)
        

    def recurrent_fn(self, u_t, h_tm1, s_tm1, W_ug, W_hg, b_g, W_ui, W_hi,
                                            b_i, W_uf, W_hf, b_f, W_uo, W_ho, b_o, W_hy, b_hy):
        g_t = self.activ2(T.dot(u_t, W_ug) + T.dot(h_tm1, W_hg) + b_g)
        i_t = self.activ1(T.dot(u_t, W_ui) + T.dot(h_tm1, W_hi) + b_i)
        f_t = self.activ1(T.dot(u_t, W_uf) + T.dot(h_tm1, W_hf) + b_f)
        o_t = self.activ1(T.dot(u_t, W_uo) + T.dot(h_tm1, W_ho) + b_o)
        s_t = g_t * i_t + s_tm1*f_t
        h_t = self.activ2(s_t)*o_t
        
        #h_t = self.activ2(T.dot(h_tm1, W_hh) + T.dot(u_t, W_uh) + b_hh)
        return [h_t, s_t]
    
