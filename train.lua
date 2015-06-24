require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'table_utils'
nngraph.setDebug(true)

rnn_size = 100
vocab_size = 3000 + 2

--train data 
sentences_ru = table.load('filtered_sentences_ru_rev')
sentences_en = table.load('filtered_sentences_en')
assert(#sentences_en == #sentences_ru)
n_data = #sentences_en

--encoder
input = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

encoder = nn.gModule({x, prev_c, prev_h}, {next_c, next_h})


--decoder
input = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(n_input, rnn_size)(input)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

prediction = nn.Linear(rnn_size, vocab_size)(next_h)
prediction = nn.LogSoftMax()(prediction)

decoder = nn.gModule({x, prev_c, prev_h}, {next_c, next_h, prediction})


--embedding layer
embed = Embedding(vocab_size, rnn_size)

criterion = nn.ClassNLLCriterion()

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(embed, encoder, decoder)
params:uniform(-0.08, 0.08)

seq_len = 30

-- make a bunch of clones, AFTER flattening, as that reallocates memory
embed_clones = model_utils.clone_many_times(embed, seq_len)
encoder_clones = model_utils.clone_many_times(encoder, seq_len)
decoder_clones = model_utils.clone_many_times(decoder, seq_len)
criterion_clones = model_utils.clone_many_times(criterion, seq_len)


-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_enc = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_c_dec = {[0]=torch.zeros(n_data, rnn_size)}
    lstm_h_dec = {[0]=torch.zeros(n_data, rnn_size)}
    x_error = {[0]=torch.rand(n_data, 28, 28)}
    x_prediction = {}
    loss_z = {}
    loss_x = {}
    canvas = {[0]=torch.rand(n_data, 28, 28)}
    x = {}
    patch = {}
    
    
    local loss = 0

    for t = 1, seq_length do
      e[t] = torch.randn(n_data, n_z)
      x[t] = features_input
      z[t], loss_z[t], lstm_c_enc[t], lstm_h_enc[t], patch[t] = unpack(encoder_clones[t]:forward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t], lstm_h_dec[t-1], ascending}))
      x_prediction[t], x_error[t], lstm_c_dec[t], lstm_h_dec[t], canvas[t], loss_x[t] = unpack(decoder_clones[t]:forward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1], ascending}))
      --print(patch[1]:gt(0.5))
      
      loss = loss + torch.mean(loss_z[t]) + torch.mean(loss_x[t])
    end
    loss = loss / seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_enc = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_c_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec1 = {[seq_length] = torch.zeros(n_data, rnn_size)}
    dlstm_h_dec2 = {[seq_length] = torch.zeros(n_data, rnn_size)}

    dx_error = {[seq_length] = torch.zeros(n_data, 28, 28)}
    dx_prediction = {}
    dloss_z = {}
    dloss_x = {}
    dcanvas = {[seq_length] = torch.zeros(n_data, 28, 28)}
    dz = {}
    dx1 = {}
    dx2 = {}
    de = {}
    dpatch = {}
    
    for t = seq_length,1,-1 do
      dloss_x[t] = torch.ones(n_data, 1)
      dloss_z[t] = torch.ones(n_data, 1)
      dx_prediction[t] = torch.zeros(n_data, 28, 28)
      dpatch[t] = torch.zeros(n_data, N, N)
      dx1[t], dz[t], dlstm_c_dec[t-1], dlstm_h_dec1[t-1], dcanvas[t-1], dascending1 = unpack(decoder_clones[t]:backward({x[t], z[t], lstm_c_dec[t-1], lstm_h_dec[t-1], canvas[t-1]}, {dx_prediction[t], dx_error[t], dlstm_c_dec[t], dlstm_h_dec[t], dcanvas[t], dloss_x[t]}))
      dx2[t], dx_error[t-1], dlstm_c_enc[t-1], dlstm_h_enc[t-1], de[t], dlstm_h_dec2[t-1], dascending2 = unpack(encoder_clones[t]:backward({x[t], x_error[t-1], lstm_c_enc[t-1], lstm_h_enc[t-1], e[t], lstm_h_dec[t-1], ascending}, {dz[t], dloss_z[t], dlstm_c_enc[t], dlstm_h_enc[t], dpatch[t]}))
      dlstm_h_dec[t-1] = dlstm_h_dec1[t-1] + dlstm_h_dec2[t-1]
    end

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end




-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dembeddings = {}                              -- d loss / d input embeddings
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {[opt.seq_length]=dfinalstate_h}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)

        -- backprop through LSTM timestep
        dembeddings[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {embeddings[t], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))

        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end




