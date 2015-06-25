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

seq_length = 30

-- make a bunch of clones, AFTER flattening, as that reallocates memory
embed_clones = model_utils.clone_many_times(embed, seq_len)
encoder_clones = model_utils.clone_many_times(encoder, seq_len)
decoder_clones = model_utils.clone_many_times(decoder, seq_len)
criterion_clones = model_utils.clone_many_times(criterion, seq_len)


x_raw_enc = sentences_ru
x_raw_dec = sentences_en
iteration_counter = 1

-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=torch.zeros(1, rnn_size)}
    lstm_h_enc = {[0]=torch.zeros(1, rnn_size)}
        
    local loss = 0
    
    x_enc = x_raw_enc[iteration_counter]
    for t = 1, #x_enc - 1 do
      lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x_enc[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}))
    end
    
    lstm_c_dec = {[0]=torch.zeros(1, rnn_size)}
    lstm_h_dec = {[0]=lstm_h_enc[#x_enc-1]}
    x_dec_prediction = {}
    
    x_dec = x_raw_dec[iteration_counter]     
    x_dec[0] = x_enc[#x_enc]
    for t = 1, #(x_raw_dec[iteration_counter]) - 1 do 
      lstm_c_dec[t], lstm_h_dec[t], x_dec_prediction[t] = unpack(decoder_clones[t]:forward({x_dec[t-1], lstm_c_dec[t-1], lstm_h_dec[t-1]}))
      loss_x = criterion_clones[t]:forward(x_dec_prediction[t], x_dec[t])
      loss = loss + loss_x
            
    end
    loss = loss / seq_length

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_dec = {[#(x_raw_dec[iteration_counter]) - 1] = torch.zeros(1, rnn_size)}
    dlstm_h_dec = {[#(x_raw_dec[iteration_counter]) - 1] = torch.zeros(1, rnn_size)}
    dloss_x = {}
    
    for t = #(x_raw_dec[iteration_counter]) - 1,1,-1 do
      dx_dec_prediction[t] = criterion_clones[t]:backward(x_dec_prediction[t], x_dec[t])
      dx_dec[t-1], dlstm_c_dec[t-1], dlstm_h_dec[t-1] = unpack(decoder_clones[t]:backward({x_dec[t-1], lstm_c_dec[t-1], lstm_h_dec[t-1]}, {lstm_c_dec[t], lstm_h_dec[t], x_dec_prediction[t]}))
    end
    
    dlstm_c_enc = {[#x_enc - 1] = torch.zeros(1, rnn_size)}
    dlstm_h_enc = {[#x_enc - 1] = dlstm_h_dec[0]}
        
    for t = #x_enc -1, 1, -1 do
      dx_enc[t], dlstm_c_enc[t-1], dlstm_h_enc[t-1] = unpack(encoder_clones[t]:backward({x_enc[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}, {lstm_c_enc[t], lstm_h_enc[t]}))
    end
      
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    iteration_counter = iteration_counter + 1
    if iteration_counter > #x_raw_enc then 
      iteration_counter = 1
    end

    return loss, grad_params
end






